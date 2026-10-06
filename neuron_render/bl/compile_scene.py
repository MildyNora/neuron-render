"""Runs inside Blender: compiles a .blend into the flat scene description Neuron Render consumes.

    blender -b scene.blend --python compile_scene.py -- --out DIR

Writes DIR/scene.npz (geometry + colour pipeline LUT) and DIR/scene.json (camera, lights, materials).
Nothing is saved back to the .blend.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

MESH_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
# Shaper for the baked view-transform LUT: u = log(v / A + 1) / log(VMAX / A + 1)
LUT_VMAX = 32.0
LUT_A = 2.0 ** -10


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--lut-size', type=int, default=129)
    ap.add_argument('--frames', default=None, help='a:b (or "scene" for the file\'s own range): also compile every frame of this range')
    return ap.parse_args(argv)


def sync_viewport_to_render(scene):
    """The depsgraph we can reach from a script evaluates viewport settings; make them match F12."""
    for col in bpy.data.collections:
        col.hide_viewport = col.hide_render
    for ob in scene.objects:
        ob.hide_viewport = ob.hide_render
        for m in ob.modifiers:
            m.show_viewport = m.show_render
            if m.type in {'SUBSURF', 'MULTIRES'}:
                m.levels = m.render_levels


def socket(node, *names):
    for n in names:
        s = node.inputs.get(n)
        if s is not None:
            return s
    return None


def socket_value(node, default, *names):
    s = socket(node, *names)
    if s is None or s.is_linked:
        return default
    v = s.default_value
    try:
        return [float(x) for x in v]
    except TypeError:
        return float(v)


def compile_material(mat):
    """Reduce a material to the Principled parameters the neural field is conditioned on."""
    out = dict(name=mat.name if mat else 'default', base_color=[0.8, 0.8, 0.8], color_attribute=None, metallic=0.0,
               roughness=0.5, ior=1.5, transmission=0.0, coat=0.0, coat_roughness=0.03, emission=0.0,
               specular=0.5, subsurface=0.0, alpha=1.0, resolved=False)
    if mat is None:
        return out
    if not mat.use_nodes or mat.node_tree is None:
        out.update(base_color=list(mat.diffuse_color)[:3], metallic=mat.metallic, roughness=mat.roughness, resolved=True)
        return out
    outputs = [n for n in mat.node_tree.nodes if n.bl_idname == 'ShaderNodeOutputMaterial' and n.target in {'ALL', 'CYCLES'}]
    outputs.sort(key=lambda n: not n.is_active_output)
    bsdf = None
    for o in outputs:
        s = o.inputs.get('Surface')
        if s is not None and s.is_linked and s.links[0].from_node.bl_idname == 'ShaderNodeBsdfPrincipled':
            bsdf = s.links[0].from_node
            break
    if bsdf is None:
        return out  # arbitrary node graph: appearance is left entirely to the neural field
    out['resolved'] = True
    bc = socket(bsdf, 'Base Color')
    if bc.is_linked:
        src = bc.links[0].from_node
        if src.bl_idname == 'ShaderNodeVertexColor':
            out['color_attribute'] = src.layer_name or '<render>'
        elif src.bl_idname == 'ShaderNodeAttribute' and src.attribute_type == 'GEOMETRY':
            out['color_attribute'] = src.attribute_name
        elif src.bl_idname == 'ShaderNodeRGB':
            out['base_color'] = list(src.outputs[0].default_value)[:3]
        else:
            out['resolved'] = False
    else:
        out['base_color'] = list(bc.default_value)[:3]
    out['metallic'] = socket_value(bsdf, 0.0, 'Metallic')
    out['roughness'] = socket_value(bsdf, 0.5, 'Roughness')
    out['ior'] = socket_value(bsdf, 1.5, 'IOR')
    out['alpha'] = socket_value(bsdf, 1.0, 'Alpha')
    out['transmission'] = socket_value(bsdf, 0.0, 'Transmission Weight', 'Transmission')
    out['coat'] = socket_value(bsdf, 0.0, 'Coat Weight', 'Clearcoat')
    out['coat_roughness'] = socket_value(bsdf, 0.03, 'Coat Roughness', 'Clearcoat Roughness')
    out['emission'] = socket_value(bsdf, 0.0, 'Emission Strength')
    out['specular'] = socket_value(bsdf, 0.5, 'Specular IOR Level', 'Specular')
    out['subsurface'] = socket_value(bsdf, 0.0, 'Subsurface Weight', 'Subsurface')
    return out


def corner_colors(me, name, vi, li, fallback):
    """Per-triangle-corner linear RGB from a colour attribute, or a constant."""
    nt = len(vi) // 3
    attr = None
    if name == '<render>':
        ca = me.color_attributes
        if len(ca):
            idx = ca.render_color_index if 0 <= ca.render_color_index < len(ca) else 0
            attr = ca[idx]
    elif name:
        attr = me.color_attributes.get(name) or me.attributes.get(name)
    if attr is None or attr.data_type not in {'FLOAT_COLOR', 'BYTE_COLOR'}:
        return np.tile(np.asarray(fallback, np.float32), (nt, 3, 1))
    buf = np.empty(len(attr.data) * 4, np.float32)
    attr.data.foreach_get('color', buf)
    buf = buf.reshape(-1, 4)[:, :3]
    idx = li if attr.domain == 'CORNER' else vi
    return buf[idx].reshape(nt, 3, 3)


def compile_geometry(dg):
    verts, normals, colors, mat_ids, obj_ids = [], [], [], [], []
    materials, mat_index, objects = [], {}, []

    def material_id(mat):
        key = mat.name if mat else None
        if key not in mat_index:
            mat_index[key] = len(materials)
            materials.append(compile_material(mat))
        return mat_index[key]

    for inst in dg.object_instances:
        ob = inst.object
        if ob.type not in MESH_TYPES:
            continue
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        if me is None or len(me.polygons) == 0:
            ob.to_mesh_clear()
            continue
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        vi = np.empty(nt * 3, np.int32)
        li = np.empty(nt * 3, np.int32)
        mi = np.empty(nt, np.int32)
        me.loop_triangles.foreach_get('vertices', vi)
        me.loop_triangles.foreach_get('loops', li)
        me.loop_triangles.foreach_get('material_index', mi)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get('co', co)
        cn = np.empty(len(me.loops) * 3, np.float32)
        me.corner_normals.foreach_get('vector', cn)
        M = np.array(inst.matrix_world, dtype=np.float64)
        wco = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        wn = cn.reshape(-1, 3).astype(np.float64) @ np.linalg.inv(M[:3, :3])
        wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-20)

        slots = [s.material for s in ob.material_slots] or [None]
        slot_ids = [material_id(m) for m in slots]
        tri_mat = np.asarray(slot_ids, np.int32)[np.clip(mi, 0, len(slot_ids) - 1)]
        col = np.empty((nt, 3, 3), np.float32)
        for s, gid in enumerate(slot_ids):
            sel = np.clip(mi, 0, len(slot_ids) - 1) == s
            if not sel.any():
                continue
            m = materials[gid]
            col[sel] = corner_colors(me, m['color_attribute'], vi, li, m['base_color'])[sel]

        verts.append(wco[vi].reshape(nt, 3, 3).astype(np.float32))
        normals.append(wn[li].reshape(nt, 3, 3).astype(np.float32))
        colors.append(col)
        mat_ids.append(tri_mat)
        obj_ids.append(np.full(nt, len(objects), np.int32))
        objects.append(dict(name=ob.name, triangles=int(nt)))
        ob.to_mesh_clear()

    if not verts:
        raise RuntimeError('scene has no renderable geometry')
    return dict(tri_verts=np.concatenate(verts), tri_normals=np.concatenate(normals), tri_color=np.concatenate(colors),
                tri_material=np.concatenate(mat_ids), tri_object=np.concatenate(obj_ids)), materials, objects


def compile_lights(dg):
    lights = []
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != 'LIGHT':
            continue
        L = ob.data
        e = dict(name=ob.name, type=L.type, color=list(L.color), energy=float(L.energy),
                 matrix=[list(r) for r in inst.matrix_world])
        if L.type == 'AREA':
            two = L.shape in {'RECTANGLE', 'ELLIPSE'}
            e.update(shape=L.shape, size=float(L.size), size_y=float(L.size_y if two else L.size), spread=float(L.spread))
        elif L.type in {'POINT', 'SPOT'}:
            e.update(radius=float(L.shadow_soft_size))
            if L.type == 'SPOT':
                e.update(spot_size=float(L.spot_size), spot_blend=float(L.spot_blend))
        elif L.type == 'SUN':
            e.update(angle=float(L.angle))
        lights.append(e)
    return lights


def frame_geometry(dg):
    """World-space triangle corners and corner normals of every renderable mesh, in compile order."""
    out = []
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type not in MESH_TYPES:
            continue
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        if me is None or len(me.polygons) == 0:
            ob.to_mesh_clear()
            continue
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        vi = np.empty(nt * 3, np.int32)
        li = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get('vertices', vi)
        me.loop_triangles.foreach_get('loops', li)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get('co', co)
        cn = np.empty(len(me.loops) * 3, np.float32)
        me.corner_normals.foreach_get('vector', cn)
        M = np.array(inst.matrix_world, dtype=np.float64)
        wco = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        wn = cn.reshape(-1, 3).astype(np.float64) @ np.linalg.inv(M[:3, :3])
        wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-20)
        out.append((wco[vi].reshape(nt, 3, 3), wn[li].reshape(nt, 3, 3)))
        ob.to_mesh_clear()
    return out


def compile_animation(scene, frames, path, n_objects):
    """Every frame of the range: where each object is (an affine transform of its reference pose, or all of
    its vertices when it deforms), where the lights and the camera are. Topology must not change."""
    ref_frame = scene.frame_current
    scene.frame_set(ref_frame)
    ref = frame_geometry(bpy.context.evaluated_depsgraph_get())
    verts = [[] for _ in ref]
    norms = [[] for _ in ref]
    cams, lens, lights = [], [], []
    for f in frames:
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        geo = frame_geometry(dg)
        if len(geo) != len(ref) or any(g[0].shape != r[0].shape for g, r in zip(geo, ref)):
            raise RuntimeError('frame %d: the scene\'s triangles change over time (topology must stay fixed)' % f)
        for k, (v, n) in enumerate(geo):
            verts[k].append(v)
            norms[k].append(n)
        cams.append(np.array(scene.camera.evaluated_get(dg).matrix_world, dtype=np.float64))
        lens.append(scene.camera.data.lens)
        lights.append(compile_lights(dg))
    scene.frame_set(ref_frame)
    out = dict(frames=np.asarray(frames, np.int32), cam_matrix=np.stack(cams), cam_lens=np.asarray(lens, np.float64),
               light_matrix=np.array([[L['matrix'] for L in fl] for fl in lights], np.float64).reshape(len(frames), -1, 4, 4),
               light_energy=np.array([[L['energy'] for L in fl] for fl in lights], np.float64).reshape(len(frames), -1),
               light_color=np.array([[L['color'] for L in fl] for fl in lights], np.float64).reshape(len(frames), -1, 3))
    kinds = []
    for k, (rv, _) in enumerate(ref):
        V = np.stack(verts[k])                                   # F, nt, 3, 3
        X = rv.reshape(-1, 3)
        extent = float(np.linalg.norm(X.max(0) - X.min(0))) + 1e-9
        if np.abs(V - rv[None]).max() < 1e-6 * extent:
            kinds.append('static')
            continue
        # affine fit of each frame to the reference pose: x_f = x_ref @ A[:3] + A[3]
        Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
        A = np.stack([np.linalg.lstsq(Xh, V[i].reshape(-1, 3), rcond=None)[0] for i in range(len(frames))])
        err = max(float(np.abs(Xh @ A[i] - V[i].reshape(-1, 3)).max()) for i in range(len(frames)))
        if err < 1e-5 * extent:
            kinds.append('rigid')
            out['o%d_affine' % k] = A
        else:
            kinds.append('deform')
            out['o%d_verts' % k] = V.astype(np.float32)
            out['o%d_normals' % k] = np.stack(norms[k]).astype(np.float32)
    assert len(kinds) == n_objects
    np.savez(path, **out)
    return kinds


def light_groups(scene, lights, materials, objects):
    """How the scene's light can be taken apart: one group per lamp, one for the world, one for glowing
    meshes. Light transport is linear in each of them, so their colour and strength stay free after the fit."""
    groups = [dict(name=L['name'], kind='light', energy=L['energy'], color=L['color']) for L in lights]
    w = scene.world
    if w is not None:
        strength, linked = 1.0, False
        if w.use_nodes and w.node_tree:
            for n in w.node_tree.nodes:
                if n.bl_idname == 'ShaderNodeBackground':
                    sock = n.inputs['Strength']
                    strength, linked = float(sock.default_value), sock.is_linked
                    break
        groups.append(dict(name='World', kind='world', strength=strength, strength_linked=linked))
    if any(m['emission'] > 0 for m in materials):
        groups.append(dict(name='Emission', kind='emission'))
    return groups


def bake_view_lut(scene, n, tmp_png):
    """Push a lattice of scene-linear colours through Blender's own save pipeline (exposure, look,
    view transform, display, curves) so the neural renderer can reproduce it exactly with a LUT."""
    import OpenImageIO as oiio
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    v = LUT_A * (np.power(LUT_VMAX / LUT_A + 1.0, u) - 1.0)
    r, g, b = np.meshgrid(v, v, v, indexing='ij')
    lat = np.stack([r, g, b, np.ones_like(r)], -1).astype(np.float32).reshape(n, n * n, 4)
    img = bpy.data.images.new('nr_lut', width=n * n, height=n, alpha=False, float_buffer=True)
    img.pixels.foreach_set(lat[::-1].ravel())
    s = scene.render.image_settings
    keep = (s.file_format, s.color_mode, s.color_depth, s.compression)
    s.file_format, s.color_mode, s.color_depth, s.compression = 'PNG', 'RGB', '16', 0
    img.save_render(tmp_png, scene=scene)
    s.file_format, s.color_mode = keep[0], keep[1]
    try:
        s.color_depth, s.compression = keep[2], keep[3]
    except TypeError:
        pass
    bpy.data.images.remove(img)
    inp = oiio.ImageInput.open(tmp_png)
    spec = inp.spec()
    a = np.asarray(inp.read_image(0, 0, 0, spec.nchannels, 'uint16')).reshape(spec.height, spec.width, spec.nchannels)
    inp.close()
    os.remove(tmp_png)
    return (a[..., :3].astype(np.float32) / 65535.0).reshape(n, n, n, 3)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    scene = bpy.context.scene
    if args.frame is not None:
        scene.frame_set(args.frame)
    sync_viewport_to_render(scene)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    geo, materials, objects = compile_geometry(dg)
    lights = compile_lights(dg)

    cam_ob = scene.camera
    if cam_ob is None:
        raise RuntimeError('scene has no active camera')
    c = cam_ob.data
    r = scene.render
    cyc = scene.cycles
    vs = scene.view_settings
    warnings = []
    if c.type != 'PERSP':
        warnings.append('camera type %s is not supported (perspective only)' % c.type)
    if c.dof.use_dof:
        warnings.append('depth of field is ignored')
    if r.use_motion_blur:
        warnings.append('motion blur is ignored')
    if scene.use_nodes and scene.node_tree and any(n.bl_idname not in {'CompositorNodeRLayers', 'CompositorNodeComposite', 'CompositorNodeViewer'} for n in scene.node_tree.nodes):
        warnings.append('compositor nodes are ignored')
    if r.engine != 'CYCLES':
        warnings.append('render engine is %s; the teacher always uses Cycles' % r.engine)

    scale = r.resolution_percentage / 100.0
    meta = dict(
        blend=bpy.data.filepath, blender=bpy.app.version_string, frame=scene.frame_current,
        camera=dict(matrix=[list(row) for row in cam_ob.evaluated_get(dg).matrix_world], type=c.type, lens=c.lens,
                    sensor_width=c.sensor_width, sensor_height=c.sensor_height, sensor_fit=c.sensor_fit,
                    shift_x=c.shift_x, shift_y=c.shift_y, clip_start=c.clip_start, clip_end=c.clip_end),
        render=dict(width=int(r.resolution_x * scale), height=int(r.resolution_y * scale),
                    pixel_aspect=r.pixel_aspect_y / r.pixel_aspect_x, film_transparent=r.film_transparent,
                    filter_type=cyc.pixel_filter_type, filter_width=cyc.filter_width, dither=r.dither_intensity,
                    engine=r.engine, device=cyc.device, samples=cyc.samples, denoise=cyc.use_denoising,
                    adaptive=cyc.use_adaptive_sampling, adaptive_threshold=cyc.adaptive_threshold,
                    max_bounces=cyc.max_bounces),
        view=dict(display=scene.display_settings.display_device, transform=vs.view_transform, look=vs.look,
                  exposure=vs.exposure, gamma=vs.gamma),
        lut=dict(size=args.lut_size, vmax=LUT_VMAX, a=LUT_A),
        materials=materials, lights=lights, objects=objects, warnings=warnings,
        groups=light_groups(scene, lights, materials, objects), frame_range=[scene.frame_start, scene.frame_end], fps=scene.render.fps)

    if args.frames:
        a, b = (scene.frame_start, scene.frame_end) if args.frames == 'scene' else (int(x) for x in args.frames.split(':'))
        kinds = compile_animation(scene, list(range(a, b + 1)), os.path.join(args.out, 'anim.npz'), len(objects))
        for o, kind in zip(objects, kinds):
            o['motion'] = kind
        meta['animation'] = dict(frames=[a, b], reference=scene.frame_current)
        print('NR_ANIMATION frames=%d..%d moving=%s' % (a, b, ','.join('%s:%s' % (o['name'], o['motion']) for o in objects if o['motion'] != 'static')), flush=True)
    lut = bake_view_lut(scene, args.lut_size, os.path.join(args.out, '_lut_tmp.png'))
    np.savez(os.path.join(args.out, 'scene.npz'), lut=lut, **geo)
    with open(os.path.join(args.out, 'scene.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('NR_COMPILED triangles=%d materials=%d lights=%d objects=%d' % (len(geo['tri_verts']), len(materials), len(lights), len(objects)), flush=True)
    for w in warnings:
        print('NR_WARNING ' + w, flush=True)
    print('NR_DONE', flush=True)


main()
