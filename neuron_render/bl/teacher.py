"""Runs inside Blender: asks Cycles for radiance at a list of camera poses.

    blender -b scene.blend --python teacher.py -- --job job.json

Job modes
  teacher  the scene's own pixel filter with the job's sample count / denoising: training views, and
           (at a high sample count) the converged references used to score held-out views
  scene    the .blend's render settings untouched (what F12 produces), the "ordinary" baseline
"point_sample" collapses the pixel filter to the pixel centre (used to verify the compiled geometry).
A view may carry a frame number and material / light overrides; with "lightgroups" the job writes one
radiance image per light group ([groups, H, W, 3]) instead of the combined one. With "environment" every
object is hidden and the views record the world alone (see neuron_render/environment.py).

Each view is written as <name>.npy (float16 scene-linear RGB, top row first). Nothing is saved to the .blend.
"""
import argparse
import json
import os
import sys
import time

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix


def enable_gpu():
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('METAL', 'OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            return kind
    return None


def read_exr(path):
    inp = oiio.ImageInput.open(path)
    spec = inp.spec()
    a = np.asarray(inp.read_image(0, 0, 0, spec.nchannels, 'float')).reshape(spec.height, spec.width, spec.nchannels)
    inp.close()
    return a, list(spec.channelnames)


def channels(a, names, suffixes):
    idx = []
    for s in suffixes:
        hit = [i for i, n in enumerate(names) if n == s or n.endswith('.' + s)]
        if not hit:
            raise RuntimeError('EXR is missing channel %s (has %s)' % (s, names))
        idx.append(hit[0])
    return a[..., idx]


def principled(mat):
    if mat is None or not mat.use_nodes:
        return None
    return next((n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled'), None)


class MaterialState:
    """Sets the editable material parameters for one view, always starting from the file's own values."""

    def __init__(self, editable):
        self.nodes = {}
        for name in editable:
            node = principled(bpy.data.materials.get(name))
            if node is not None:
                self.nodes[name] = (node, tuple(node.inputs['Base Color'].default_value), float(node.inputs['Roughness'].default_value))

    def apply(self, overrides):
        for name, (node, base, rough) in self.nodes.items():
            o = overrides.get(name, {})
            node.inputs['Base Color'].default_value = (*o['base_color'], 1.0) if 'base_color' in o else base
            node.inputs['Roughness'].default_value = o.get('roughness', rough)


class LightState:
    """Sets lamp strength / colour and world strength for one view, always starting from the file's own values."""

    def __init__(self, scene):
        self.lamps = {o.name: (o.data, float(o.data.energy), tuple(o.data.color)) for o in scene.objects if o.type == 'LIGHT'}
        self.world = None
        w = scene.world
        if w is not None and w.use_nodes and w.node_tree:
            bg = next((n for n in w.node_tree.nodes if n.bl_idname == 'ShaderNodeBackground'), None)
            if bg is not None and not bg.inputs['Strength'].is_linked:
                self.world = (bg.inputs['Strength'], float(bg.inputs['Strength'].default_value))

    def apply(self, overrides, white=False):
        for name, (data, energy, color) in self.lamps.items():
            o = overrides.get(name, {})
            data.energy = o.get('energy', energy)
            data.color = (1.0, 1.0, 1.0) if white else tuple(o.get('color', color))
        if self.world is not None:
            sock, strength = self.world
            sock.default_value = overrides.get('World', {}).get('strength', strength)


def setup_lightgroups(scene, groups, out):
    """One Cycles light group per lamp, one for the world, one for glowing meshes; and a compositor graph that
    denoises each group's pass (Cycles itself only denoises the combined image) and writes them to one EXR."""
    vl = bpy.context.view_layer
    names = ['g%d' % i for i in range(len(groups))]
    for n in names:
        vl.lightgroups.add(name=n)
    for n, g in zip(names, groups):
        if g['kind'] == 'light':
            bpy.data.objects[g['name']].lightgroup = n
        elif g['kind'] == 'world':
            scene.world.lightgroup = n
        else:
            for ob in scene.objects:
                if ob.type == 'MESH' and any(m and principled(m) and principled(m).inputs['Emission Strength'].default_value > 0 for m in ob.data.materials):
                    ob.lightgroup = n
    vl.cycles.denoising_store_passes = True
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    tree.links.new(rl.outputs['Image'], tree.nodes.new('CompositorNodeComposite').inputs['Image'])
    fo = tree.nodes.new('CompositorNodeOutputFile')
    fo.base_path = os.path.join(out, '_groups_')
    fo.format.file_format, fo.format.color_depth, fo.format.exr_codec = 'OPEN_EXR_MULTILAYER', '32', 'NONE'
    fo.layer_slots.clear()
    for n in names:
        dn = tree.nodes.new('CompositorNodeDenoise')
        dn.use_hdr = True
        tree.links.new(rl.outputs['Combined_' + n], dn.inputs['Image'])
        tree.links.new(rl.outputs['Denoising Normal'], dn.inputs['Normal'])
        tree.links.new(rl.outputs['Denoising Albedo'], dn.inputs['Albedo'])
        fo.layer_slots.new(n)
        tree.links.new(dn.outputs['Image'], fo.inputs[-1])
    return names


def read_groups(out, names):
    import glob
    files = sorted(glob.glob(os.path.join(out, '_groups_*.exr')))
    a, ch = read_exr(files[-1])
    for f in files:
        os.remove(f)
    return np.stack([channels(a, ch, [n + '.R', n + '.G', n + '.B']) for n in names])


def setup_environment(scene, out):
    """Leave only the world to render. A world shader may show the camera something other than what it lights
    and reflects with (a Light Path node); then the camera is given a perfect mirror to look into, so that
    what it records is the world as a reflection ray sees it."""
    r, cyc = scene.render, scene.cycles
    r.film_transparent = False
    cyc.sample_clamp_direct = cyc.sample_clamp_indirect = 0.0
    cyc.blur_glossy = 0.0
    for ob in scene.objects:
        ob.hide_render = True
    w = scene.world
    mirror = bool(w and w.use_nodes and w.node_tree and any(n.bl_idname == 'ShaderNodeLightPath' for n in w.node_tree.nodes))
    data = bpy.data.cameras.new('NR_Env')
    data.type, data.sensor_fit, data.sensor_width, data.clip_start, data.clip_end = 'PERSP', 'AUTO', 36.0, 0.01, 1.0e6
    cam = bpy.data.objects.new('NR_Env', data)
    scene.collection.objects.link(cam)
    if mirror:
        cyc.max_bounces, cyc.glossy_bounces = max(cyc.max_bounces, 2), max(cyc.glossy_bounces, 2)
        me = bpy.data.meshes.new('NR_Mirror')
        me.from_pydata([(-50, -50, -1), (50, -50, -1), (50, 50, -1), (-50, 50, -1)], [], [(0, 1, 2, 3)])
        mat = bpy.data.materials.new('NR_Mirror')
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        gl = nt.nodes.new('ShaderNodeBsdfGlossy')
        gl.inputs['Color'].default_value, gl.inputs['Roughness'].default_value = (1.0, 1.0, 1.0, 1.0), 0.0
        nt.links.new(gl.outputs['BSDF'], nt.nodes.new('ShaderNodeOutputMaterial').inputs['Surface'])
        me.materials.append(mat)
        plane = bpy.data.objects.new('NR_Mirror', me)
        scene.collection.objects.link(plane)
        plane.parent = cam
    with open(os.path.join(out, 'environment.json'), 'w') as f:
        json.dump(dict(mirror=mirror), f)
    return cam


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    job = json.load(open(ap.parse_args(argv).job))
    out = job['out']
    os.makedirs(out, exist_ok=True)
    mode = job.get('mode', 'teacher')

    scene = bpy.context.scene
    r, cyc = scene.render, scene.cycles
    r.engine = 'CYCLES'
    device = 'CPU'
    if mode != 'scene':
        if job.get('device', 'GPU') != 'CPU' and enable_gpu():
            cyc.device = 'GPU'
        else:
            cyc.device = 'CPU'
        cyc.samples = int(job['samples'])
        thr = float(job.get('adaptive_threshold', 0.0))
        cyc.use_adaptive_sampling = thr > 0
        if thr > 0:
            cyc.adaptive_threshold = thr
        cyc.use_denoising = bool(job.get('denoise', True))
        if cyc.use_denoising:
            cyc.denoiser = 'OPENIMAGEDENOISE'
            try:
                cyc.denoising_use_gpu = cyc.device == 'GPU'
            except (AttributeError, TypeError):
                pass
        if job.get('point_sample'):
            cyc.pixel_filter_type = 'BLACKMAN_HARRIS'
            cyc.filter_width = 0.01
        r.use_compositing = False
        r.use_sequencer = False
        r.use_motion_blur = False
    else:
        # The file's own settings, optionally truncated: what Cycles has after `samples` samples.
        if 'samples' in job:
            cyc.samples = int(job['samples'])
        if 'denoise' in job:
            cyc.use_denoising = bool(job['denoise'])
    device = cyc.device
    r.resolution_x, r.resolution_y, r.resolution_percentage = int(job['width']), int(job['height']), 100
    r.use_border = False
    r.use_persistent_data = True
    want_pos = bool(job.get('passes', False))
    if want_pos:
        bpy.context.view_layer.use_pass_position = True
    r.image_settings.file_format = 'OPEN_EXR_MULTILAYER' if want_pos else 'OPEN_EXR'
    r.image_settings.color_mode = 'RGB'
    r.image_settings.color_depth = '32'
    r.image_settings.exr_codec = 'NONE'

    materials, lamps = MaterialState(job.get('editable', {})), LightState(scene)
    groups = None
    if job.get('lightgroups'):
        # Per-light targets: every lamp white at its own strength, so its colour stays free after the fit.
        cyc.use_denoising = False
        groups = setup_lightgroups(scene, job['lightgroups'], out)

    if job.get('environment'):
        cam = setup_environment(scene, out)
    else:
        # A constraint-free copy of the scene camera, so poses can be set directly.
        src = scene.camera
        cam = bpy.data.objects.new('NR_Camera', src.data.copy())
        scene.collection.objects.link(cam)
    scene.camera = cam

    tmp = os.path.join(out, '_frame.exr')
    views = job['views']
    times = []
    t_all = time.time()
    for i, v in enumerate(views):
        if 'frame' in v:
            scene.frame_set(int(v['frame']))
        materials.apply(v.get('materials', {}))
        lamps.apply(v.get('lights', {}), white=groups is not None)
        cam.matrix_world = Matrix(v['matrix'])
        if 'lens' in v:
            cam.data.lens = v['lens']
        r.filepath = tmp
        t = time.time()
        if groups is not None:
            bpy.ops.render.render(write_still=False)
            np.save(os.path.join(out, v['name'] + '.npy'), read_groups(out, groups).astype(np.float16))
            times.append(time.time() - t)
            print('NR_PROGRESS %d %d %.3f' % (i + 1, len(views), times[-1]), flush=True)
            continue
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t
        if job.get('save_png'):
            s = r.image_settings
            s.file_format, s.color_mode, s.color_depth = 'PNG', 'RGB', '8'
            bpy.data.images['Render Result'].save_render(os.path.join(out, v['name'] + '.png'), scene=scene)
            s.file_format = 'OPEN_EXR_MULTILAYER' if want_pos else 'OPEN_EXR'
            s.color_mode, s.color_depth, s.exr_codec = 'RGB', '32', 'NONE'
        a, names = read_exr(tmp)
        rgb = channels(a, names, ['Combined.R', 'Combined.G', 'Combined.B']) if want_pos else channels(a, names, ['R', 'G', 'B'])
        np.save(os.path.join(out, v['name'] + '.npy'), rgb.astype(np.float16))
        if want_pos:
            np.save(os.path.join(out, v['name'] + '.pos.npy'), channels(a, names, ['Position.X', 'Position.Y', 'Position.Z']).astype(np.float32))
        times.append(dt)
        print('NR_PROGRESS %d %d %.3f' % (i + 1, len(views), dt), flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(os.path.join(out, 'timing.json'), 'w') as f:
        json.dump(dict(mode=mode, device=device, samples=cyc.samples, width=r.resolution_x, height=r.resolution_y,
                       seconds=times, total=time.time() - t_all, names=[v['name'] for v in views]), f)
    print('NR_DONE', flush=True)


main()
