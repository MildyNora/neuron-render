"""The world's own light, baked once: what a ray that leaves the scene sees.

Cycles renders the world shader on its own through six 90-degree cameras (bl/teacher.py, "environment");
the faces are folded into one equirectangular map of scene-linear radiance, <bundle>/env.npy. The traced
reflection and refraction rays of the network's inputs read it wherever they escape the geometry, so a
mirror shows the real sky in every direction, including the ones no training view ever looked at.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import blender

# (forward, up) of the six faces. Each is rendered a little wider than 90 degrees so lookups never fall off an edge.
FACES = [((1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)),
         ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0))]
MARGIN = 1.04


def _frame(k):
    f, u = (np.asarray(a, np.float64) for a in FACES[k])
    return np.cross(f, u), u, f


def face_matrix(k):
    """Camera-to-world matrix (Blender: columns right, up, -forward) of face k, at the origin."""
    right, up, fwd = _frame(k)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = right, up, -fwd
    return M


def directions(width):
    """Unit directions of an equirectangular map's texel centres, [width / 2, width, 3]: column = azimuth
    (atan2(y, x), -pi at the left edge), row = polar angle from +Z (the zenith is the top row)."""
    h = width // 2
    az = ((np.arange(width) + 0.5) / width - 0.5) * 2.0 * np.pi
    po = (np.arange(h) + 0.5) / h * np.pi
    s = np.sin(po)[:, None]
    return np.stack([s * np.cos(az)[None], s * np.sin(az)[None], np.broadcast_to(np.cos(po)[:, None], (h, width))], -1)


def lookup(env, dirs):
    """Bilinear lookup of radiance along unit directions [..., 3] (the same arithmetic as the nr_env kernel)."""
    h, w = env.shape[:2]
    u = (np.arctan2(dirs[..., 1], dirs[..., 0]) / (2.0 * np.pi) + 0.5) * w - 0.5
    v = np.arccos(np.clip(dirs[..., 2], -1.0, 1.0)) / np.pi * h - 0.5
    u0, v0 = np.floor(u), np.floor(v)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    x0 = (u0.astype(np.int64) + w) % w
    x1 = (x0 + 1) % w
    y0 = np.clip(v0.astype(np.int64), 0, h - 1)
    y1 = np.clip(v0.astype(np.int64) + 1, 0, h - 1)
    return (env[y0, x0] * (1 - fu) + env[y0, x1] * fu) * (1 - fv) + (env[y1, x0] * (1 - fu) + env[y1, x1] * fu) * fv


def assemble(faces, mirror, width=2048):
    """Six face renders [6, N, N, 3] -> equirectangular radiance [width / 2, width, 3].

    mirror: the faces were rendered looking into a mirror, so a camera ray d recorded the world along
    d - 2 (d . forward) forward, the direction behind the camera."""
    n = faces.shape[1]
    w = directions(width)
    out = np.zeros(w.shape, np.float32)
    axis = np.argmax(np.abs(w), -1)
    positive = np.take_along_axis(w, axis[..., None], -1)[..., 0] > 0
    for k in range(6):
        right, up, fwd = _frame(k)
        a = int(np.argmax(np.abs(fwd)))
        sel = (axis == a) & (positive == ((fwd[a] > 0) != mirror))
        d = w[sel]
        if mirror:
            d = d - 2.0 * (d @ fwd)[:, None] * fwd[None]
        z = d @ fwd
        x = ((d @ right) / z / MARGIN + 1.0) * 0.5 * n - 0.5
        y = (1.0 - (d @ up) / z / MARGIN) * 0.5 * n - 0.5
        x0, y0 = np.clip(np.floor(x).astype(np.int64), 0, n - 2), np.clip(np.floor(y).astype(np.int64), 0, n - 2)
        fx, fy = np.clip(x - x0, 0.0, 1.0)[:, None], np.clip(y - y0, 0.0, 1.0)[:, None]
        img = faces[k].astype(np.float64)
        out[sel] = (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy
    return out


def bake(blend, bundle, frame=None, size=1024, width=2048, log=print):
    """Render the world alone and write <bundle>/env.npy. Returns the map, or None for a scene without a world."""
    out = os.path.join(bundle, 'env')
    views = [dict(name='face%d' % k, matrix=face_matrix(k).tolist(), lens=18.0 / MARGIN, **({} if frame is None else dict(frame=int(frame))))
             for k in range(6)]
    blender.run_job(blend, dict(out=out, mode='teacher', width=size, height=size, samples=8, adaptive_threshold=0, denoise=False,
                                device='CPU', environment=True, views=views))
    faces = np.stack([np.load(os.path.join(out, 'face%d.npy' % k)).astype(np.float32) for k in range(6)])
    mirror = json.load(open(os.path.join(out, 'environment.json')))['mirror']
    env = np.clip(np.nan_to_num(assemble(faces, mirror, width)), 0.0, 6.0e4)   # stored as float16
    for k in range(6):
        os.remove(os.path.join(out, 'face%d.npy' % k))
    np.save(os.path.join(bundle, 'env.npy'), env.astype(np.float16))
    log('  environment: %dx%d from the world shader%s, radiance %.3g to %.3g' % (
        width, width // 2, ' (as reflections see it)' if mirror else '', float(env.min()), float(env.max())))
    return env
