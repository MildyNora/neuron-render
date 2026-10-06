"""Export a fitted bundle to a portable pack that other renderers (the Unity runtime in unity/) can load.

    neuron-render export scenes/stage.blend --parametric -o stage.nrpack

A pack is a directory: pack.json describes the fit (sizes, kernel settings, lights, materials, frames) and
names the binary arrays stored beside it (raw little-endian, dtype and shape in the JSON). Half-precision
tables are stored as float16, geometry as float32. Nothing in it depends on PyTorch or Metal.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import plan as P
from .model import NeuralField
from .scene import Scene


def _bvh_depth(nodes):
    """Depth of every BVH node (root 0), for a level-by-level refit on the GPU."""
    depth = np.zeros(len(nodes), np.int32)
    stack = [0]
    while stack:
        i = stack.pop()
        a, b = int(nodes[i, 3]), int(nodes[i, 7])
        if b <= 0:   # inner: a = left, b = -right
            for c in (a, -b):
                depth[c] = depth[i] + 1
                stack.append(c)
    return depth


def export(bundle, out, log=print):
    scene = Scene(bundle)
    field = NeuralField.load(scene, os.path.join(bundle, 'model.pt'), device='cpu')
    plan = P.load_plan(bundle)
    fit = json.load(open(os.path.join(bundle, 'fit.json'))) if os.path.exists(os.path.join(bundle, 'fit.json')) else {}
    os.makedirs(out, exist_ok=True)
    cfg = field.cfg
    arrays = {}

    def put(name, a, dtype):
        a = np.ascontiguousarray(np.asarray(a).astype(dtype))
        a.tofile(os.path.join(out, name + '.bin'))
        arrays[name] = dict(file=name + '.bin', dtype=np.dtype(dtype).name, shape=list(a.shape))

    t = lambda x: x.detach().cpu().numpy()
    # the network
    put('table', t(field.table), np.float16)
    put('ltable', t(field.ltable), np.float16)
    put('emb', t(field.emb), np.float16)
    if field.ldec is not None:
        put('ldec', t(field._ldec()), np.float32)
    for i, layer in enumerate(field.mlp):
        put('mlp_w%d' % i, t(layer.weight), np.float16)
        put('mlp_b%d' % i, t(layer.bias), np.float16)
    put('lv', t(field.lv), np.int32)
    put('llv', t(field.llv), np.int32)
    put('lut', t(field.view.lut).reshape(-1), np.float16)
    if field.icfg_single[17].item() > 0:
        put('env', scene.env, np.float16)
    # the scene, in its reference pose
    ref = scene.pack_frame(None)
    put('tri_ref', ref['tri'], np.float32)                       # [T + 1, 24]: corners, corner normals, plane, material, flat
    put('tref', scene.reference_table(), np.float32)             # [T + 1, 9]: corners in the reference pose, what the grids are indexed by
    put('tri_material', scene.tri_material, np.int32)
    put('tri_object', scene.tri_object, np.int32)
    put('tri_smooth', (np.abs(scene.tri_normals - scene.tri_normals[:, :1]).max(axis=(1, 2)) >= 1e-4), np.uint8)
    put('tri_textured', scene.textured(), np.uint8)
    put('colors', scene.color_table(tint=field.parametric), np.float32)
    put('materials', scene.material_table(tint=field.parametric), np.float32)
    put('bvh', ref['bvh'], np.float32)
    put('bvh_order', ref['bvh_order'], np.int32)
    put('bvh_depth', _bvh_depth(ref['bvh']), np.int32)
    put('lights', np.stack([scene.pack_lights(fi) for fi in range(scene.n_frames)]) if scene.anim is not None else ref['lights'][None], np.float32)
    objects = []
    for k, o in enumerate(scene.meta['objects']):
        kind = o.get('motion', 'static') if scene.anim is not None else 'static'
        entry = dict(name=o['name'], motion=kind)
        if kind == 'rigid':
            put('o%d_affine' % k, scene.anim['o%d_affine' % k], np.float32)       # [F, 4, 3]: x_f = x_ref @ A[:3] + A[3]
            entry['affine'] = 'o%d_affine' % k
        elif kind == 'deform':
            sel = np.nonzero(scene.tri_object == k)[0]
            put('o%d_verts' % k, scene.anim['o%d_verts' % k], np.float32)         # [F, n, 3, 3]
            put('o%d_normals' % k, scene.anim['o%d_normals' % k], np.float32)
            entry.update(verts='o%d_verts' % k, normals='o%d_normals' % k, first=int(sel[0]), count=int(len(sel)))
        objects.append(entry)
    cam = scene.camera
    cams = []
    for fi in range(scene.n_frames):
        c = scene.camera_at(fi)
        cams.append(dict(matrix=np.asarray(c.matrix).tolist(), lens=c.lens))
    meta = dict(
        format='neuron-render pack', version=1, parametric=field.parametric, blend=os.path.basename(scene.meta.get('blend', '')),
        width=cam.width, height=cam.height, sensor_width=cam.sensor_width, sensor_height=cam.sensor_height, sensor_fit=cam.sensor_fit,
        shift_x=cam.shift_x, shift_y=cam.shift_y, clip_start=cam.clip_start, cameras=cams,
        n_tris=scene.n_tris, sky_id=scene.sky_id, dims=dict(DIN=field.DIN, D=field.D, L=field.L, F=field.F, FT=field.FT, LL=field.LL, FL=field.FL,
                                                               G=field.G, NL=field.NL, KT=field.KT, DG=field.DG),
        icfg=[int(x) for x in field.icfg_single.tolist()], cfg_f=[float(x) for x in field.cfg_f.tolist()],
        eps=cfg['eps'], radiance_scale=cfg['radiance_scale'], lut=dict(size=scene.meta['lut']['size'], vmax=scene.meta['lut']['vmax'], a=scene.meta['lut']['a']),
        dither=field.view.dither, sky=cfg['sky'], lobe=int(cfg['lobe']), follow=bool(cfg['follow']),
        filter=dict(type=scene.meta['render'].get('filter_type', 'BLACKMAN_HARRIS'), width=scene.meta['render'].get('filter_width', 1.5)),
        frames=[int(f) for f in scene.frames], fps=scene.meta.get('fps', 24),
        groups=scene.groups, light_order=list(scene.light_order),
        materials=[dict(name=m['name'], roughness=m['roughness'], metallic=m['metallic'], transmission=m['transmission'],
                        base_color=list(m['base_color'][:3]), editable=plan.get('editable', {}).get(m['name'], []) if field.parametric else [])
                   for m in scene.materials],
        varied=[i for i, m in enumerate(scene.materials) if m['name'] in cfg.get('varied', ())],
        objects=objects, arrays=arrays,
        accuracy=fit.get('accuracy', {}).get('presets'), one_time_seconds=fit.get('one_time_seconds'))
    with open(os.path.join(out, 'pack.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    total = sum(os.path.getsize(os.path.join(out, a['file'])) for a in arrays.values())
    log('exported %s -> %s (%.0f MB, %d arrays)' % (os.path.basename(bundle), out, total / 1e6, len(arrays)))
    return out
