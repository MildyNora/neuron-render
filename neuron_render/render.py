"""Frame rendering: native visibility -> neural shading -> filtered, view-transformed pixels."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import NeuralField
from .native import Rasterizer
from .scene import Scene


@dataclass(frozen=True)
class Quality:
    """The speed/quality dial.

    scale   internal resolution relative to the output (upsampled when < 1)
    aa      visibility samples per pixel axis (aa*aa per pixel); edges are filtered exactly as Cycles does
    rate    shading block size in pixels: one neural evaluation per visible triangle per rate x rate block
            (textures and curved glossy surfaces are always shaded per pixel)
    full    evaluate the network at every visibility sample instead of once per triangle per block
    lobe    rays that sample a rough metal's blurred reflection, at most the number the fit was made with
            (fits that cover changes only; the others always use their four-ray estimate)
    """
    scale: float = 1.0
    aa: int = 2
    rate: int = 1
    full: bool = False
    lobe: int = 16


PRESETS = {
    'draft': Quality(scale=0.5, aa=2, rate=4, lobe=4),
    'fast': Quality(aa=2, rate=4, lobe=4),
    'balanced': Quality(aa=2, rate=2, lobe=8),
    'high': Quality(aa=4, rate=1),
    'ultra': Quality(aa=2, rate=1, full=True),
}


class Renderer:
    def __init__(self, scene, field, threads=None):
        self.scene, self.field = scene, field
        self.device = field.device
        self.lib = field.lib
        self.raster = Rasterizer(scene.tri_verts, threads)
        self.view = field.view
        self._subw = {}
        self.cap = scene.per_pixel() if scene.per_pixel().any() else None   # where block shading would show
        self.parametric = field.parametric
        self._lobes = {}
        self.reset()

    @staticmethod
    def open(bundle, device='mps', threads=None):
        scene = Scene(bundle)
        return Renderer(scene, NeuralField.load(scene, os.path.join(bundle, 'model.pt'), device), threads)

    def reset(self):
        """Forget cached per-frame state (after the field's tables were used for something else, e.g. fitting)."""
        self._frames, self._mat_key = {}, None

    @property
    def sky_neural(self):
        return self.field.cfg['sky']['mode'] == 'neural'

    def _icfg(self, lobe):
        """The field's kernel settings with the reflection lobe thinned to `lobe` rays (see Quality.lobe)."""
        fitted = int(self.field.cfg['lobe'])
        n = max(1, min(int(lobe), fitted)) if fitted > 0 else 0
        if n not in self._lobes:
            t = self.field.icfg_single.clone()
            t[19] = n
            self._lobes[n] = t
        return self._lobes[n]

    # -- parametric state ------------------------------------------------------------------------
    def _state(self, fi, materials):
        """Point the field at one frame and one material state; returns that frame's triangles."""
        fld, up = self.field, lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        if fi not in self._frames:
            if len(self._frames) >= 256:
                self._frames.pop(next(iter(self._frames)))
            f = self.scene.pack_frame(fi)
            self._frames[fi] = dict(verts=np.ascontiguousarray(f['verts'], np.float32), T=up(f['tri']), Lg=up(f['lights']),
                                    bvh=up(f['bvh']), order=up(f['bvh_order']))
        st = self._frames[fi]
        key = json.dumps(materials or {}, sort_keys=True)
        if key != self._mat_key:
            table = self.scene.material_table(materials, tint=True)
            self._M, self._g, self._mat_key = up(table), fld.global_state(table), key
            cap = self.scene.per_pixel(materials, flat_mirrors=True)   # a repaint can turn a surface into a mirror
            self.cap = cap if cap.any() else None
        fld.T, fld.Lg, fld.bvh, fld.bvh_order, fld.M = st['T'], st['Lg'], st['bvh'], st['order'], self._M
        fld.use_views(1, [[0, 0]], [np.concatenate([[self.scene.time_of(fi)], self._g])])
        return st['verts']

    # -- rendering -------------------------------------------------------------------------------
    def render(self, camera=None, quality='balanced', output='u8', profile=False, frame=None, materials=None, lights=None):
        """Render one frame. output: 'u8' (display-ready uint8), 'display' (float 0..1) or 'linear' (scene-linear).

        Parametric fits only: frame (animation frame number), materials ({name: {base_color, roughness}}) and
        lights ({lamp name: {energy, color}, 'World': {strength}}) choose the scene state; all default to the file's.
        Returns (image [H, W, 3], info). Timings in info cover everything up to pixels in host memory.
        """
        q = PRESETS[quality] if isinstance(quality, str) else quality
        t0 = time.perf_counter()
        verts, weights, sky = None, None, None
        fld = self.field
        # Rendering points the field at one frame; whatever it was pointed at (the stacks of a fit in progress)
        # is put back afterwards.
        held = (fld.T, fld.Lg, fld.bvh, fld.bvh_order, fld.M, fld.vs, fld.vt, fld.icfg_single)
        try:
            fld.icfg_single = self._icfg(q.lobe)
            return self._render(camera, q, output, profile, frame, materials, lights, t0)
        finally:
            fld.T, fld.Lg, fld.bvh, fld.bvh_order, fld.M, fld.vs, fld.vt, fld.icfg_single = held

    def _render(self, camera, q, output, profile, frame, materials, lights, t0):
        verts, weights, sky = None, None, None
        if self.parametric:
            fi = self.scene.frame_index(frame)
            verts = self._state(fi, materials)
            weights = self.scene.group_weights(lights)
            target = camera or self.scene.camera_at(fi)
            if not self.sky_neural:
                sky = (np.asarray(self.field.cfg['sky']['groups']) * np.asarray(weights)).sum(0).tolist()
        else:
            if frame is not None or materials or lights:
                raise ValueError('this is a static fit: frame, materials and lights need a parametric one (fit --vary ...)')
            target = camera or self.scene.camera
            if not self.sky_neural:
                sky = self.field.cfg['sky']['color']
        cam = target if q.scale == 1.0 else target.scaled(q.scale)
        dev = self.device
        sync = torch.mps.synchronize if (profile and dev.type == 'mps') else (lambda: None)
        ts = time.perf_counter()

        if q.aa not in self._subw:
            self._subw[q.aa] = self.scene.pixel_filter(q.aa)
        r = self.raster.resolve(cam, self._subw[q.aa], q.aa, q.rate, q.full, self.field.sky_id if self.sky_neural else -1, self.cap, verts)
        K, E = r['n_groups'], r['n_entries']
        t1 = time.perf_counter()

        up = lambda a: torch.from_numpy(a).to(dev) if len(a) else torch.zeros((1,) + a.shape[1:], dtype=torch.from_numpy(a).dtype, device=dev)
        g_tri, g_dir = up(r['g_tri']), up(r['g_dir'])
        px_ref, e_group, e_w, px_miss = up(r['px_ref']), up(r['e_group']), up(r['e_w']), up(r['px_miss'])
        origin = torch.tensor(cam.basis()[0], dtype=torch.float32, device=dev)
        sync()
        t2 = time.perf_counter()

        if K > 0:
            rad = self.field.infer(g_tri, g_dir, origin)
        else:
            rad = torch.zeros(1, 3 * self.field.G, dtype=torch.float16, device=dev)
        sync()
        t3 = time.perf_counter()

        P = cam.width * cam.height
        rcfg = lambda **kw: self.view.rcfg(sky, weights=weights, **kw)
        if output == 'u8' and cam is target:
            img = torch.empty(P * 3, dtype=torch.uint8, device=dev)
            self.lib.nr_resolve_u8(img, rad, px_ref, e_group, e_w, px_miss, self.view.lut, rcfg(dither=self.view.dither), threads=P)
            img = img.reshape(cam.height, cam.width, 3)
        else:
            img = torch.empty(P * 3, dtype=torch.float32, device=dev)
            self.lib.nr_resolve_f(img, rad, px_ref, e_group, e_w, px_miss, self.view.lut, rcfg(linear=output == 'linear'), threads=P)
            if output == 'u8':   # rendered below output resolution: upsample and quantise on the GPU
                small, img = img, torch.empty(target.width * target.height * 3, dtype=torch.uint8, device=dev)
                dims = torch.tensor([cam.width, cam.height, target.width, target.height], dtype=torch.int32, device=dev)
                self.lib.nr_upsample_u8(img, small, dims, rcfg(dither=self.view.dither), threads=target.width * target.height)
                img = img.reshape(target.height, target.width, 3)
            else:
                img = img.reshape(cam.height, cam.width, 3)
                if cam is not target:
                    img = F.interpolate(img.permute(2, 0, 1)[None], size=(target.height, target.width), mode='bicubic',
                                        align_corners=False)[0].permute(1, 2, 0)
                if output == 'display':
                    img = img.clamp(0, 1)
        img = img.cpu().numpy()
        t4 = time.perf_counter()
        info = dict(total_ms=(t4 - t0) * 1e3, state_ms=(ts - t0) * 1e3, visibility_ms=(t1 - ts) * 1e3, upload_ms=(t2 - t1) * 1e3,
                    shade_ms=(t3 - t2) * 1e3, resolve_ms=(t4 - t3) * 1e3, samples=K, width=target.width, height=target.height)
        return img, info

    def warmup(self, quality='balanced', n=3, **state):
        for _ in range(n):
            self.render(quality=quality, **state)

    def display(self, linear):
        """Blender's view transform (the baked LUT) applied to a scene-linear float image, on the CPU."""
        m = self.scene.meta['lut']
        N, vmax, a = m['size'], m['vmax'], m['a']
        lut = self.scene.lut
        u = np.log(np.clip(linear.astype(np.float64), 0, vmax) / a + 1.0) / np.log(vmax / a + 1.0) * (N - 1)
        i0 = np.clip(np.floor(u).astype(np.int64), 0, N - 2)
        f = u - i0
        out = np.zeros(linear.shape, np.float64)
        for k in range(8):
            d = (k & 1, (k >> 1) & 1, (k >> 2) & 1)
            w = np.ones(linear.shape[:-1])
            for c in range(3):
                w = w * (f[..., c] if d[c] else 1.0 - f[..., c])
            out += w[..., None] * lut[i0[..., 0] + d[0], i0[..., 1] + d[1], i0[..., 2] + d[2]]
        return out.astype(np.float32)
