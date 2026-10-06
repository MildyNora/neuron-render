"""The neural field: analytic surface features + multiresolution feature grid + per-triangle code -> MLP -> radiance.

A *static* fit predicts one radiance per shading sample for one frozen scene. A *parametric* fit predicts one
radiance per light group (so every lamp's colour and strength stay free), reads its grids at (reference position,
time) (so the scene can move), and takes the materials' parameters as inputs (so they can be edited).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels

DEFAULTS = dict(levels=12, features=2, log2_table=20, n_min=16, n_max=2048, embed=16, width=128, depth=3, octaves=6, eps=0.01,
                lightmap_levels=12, lightmap_log2=19,
                # what escaping reflection rays see: the baked environment (else the sky as fitted), and how many
                # rays sample a rough metal's reflection lobe (0: the four-ray estimate)
                # follow: traced rays go one bounce further off metal and through glass
                env=False, lobe=0, follow=False,
                # parametric fits only
                parametric=False, groups=None, lightmap_features=4, time_octaves=6, time_cells=0, time_levels=6, time_crossed=False,
                varied=())


def grid_levels(levels, log2_table, n_min, n_max, time_cells=0, time_levels=0, crossed=False):
    """[(resolution, offset, size, dense, time resolution)] per level; coarse levels are dense, fine ones hashed.

    The first `levels` rows are three-dimensional: what they store is shared by every moment of an animation,
    so everything that does not change is learned once, from all the data. With time_cells > 0 they are
    followed by `time_levels` four-dimensional rows (coarser in space, up to time_cells slices) that carry
    only what changes: moving shadows, moving reflections.

    crossed: give the finest time resolution to the coarsest spatial level and the other way round. What moves
    fast (a soft shadow following a ball) is smooth in space, and what is detailed in space changes slowly;
    a level that is fine in both is seen by too few teacher pixels per cell and only stores noise."""
    T = 1 << log2_table
    rows, off = [], 0

    def add(res, rt):
        nonlocal off
        cells = (res + 1) ** 3 * (rt + 1)
        dense = cells <= T
        size = cells if dense else T
        rows.append([res, off, size, int(dense), rt])
        off += size

    growth = (n_max / n_min) ** (1.0 / max(levels - 1, 1))
    for l in range(levels):
        add(int(math.floor(n_min * growth ** l + 1e-6)), 0)
    if time_cells > 0:
        top = min(n_max, 512)
        growth = (top / n_min) ** (1.0 / max(time_levels - 1, 1))
        for l in range(time_levels):
            share = (time_levels - l) if crossed else (l + 1)
            add(int(math.floor(n_min * growth ** l + 1e-6)), max(2, int(round(time_cells * share / time_levels))))
    return rows, off


class _GridLookup(torch.autograd.Function):
    """Trilinear (or, with time, quadrilinear) lookup in a feature grid, forward and backward both as Metal kernels."""

    @staticmethod
    def forward(ctx, pos, table, lib, lv, L, F):
        n = pos.shape[0]
        out = torch.empty(n, L * F, device=pos.device)
        lib.nr_hash_fwd(out, pos, table, lv, L, F, threads=n)
        ctx.save_for_backward(pos)
        ctx.args, ctx.shape = (lib, lv, L, F), table.shape
        return out

    @staticmethod
    def backward(ctx, g):
        pos, = ctx.saved_tensors
        lib, lv, L, F = ctx.args
        grad = torch.zeros(ctx.shape, device=g.device)
        lib.nr_hash_bwd(grad, pos, g.contiguous(), lv, L, F, threads=pos.shape[0])
        return None, grad, None, None, None, None


class _ViewFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, view):
        n = y.shape[0]
        disp = torch.empty(n, 3, device=y.device)
        jac = torch.empty(n, 3, 3, device=y.device)
        view.lib.nr_view(disp, jac, y.contiguous(), view.lut, view.rcfg(), threads=n)
        ctx.save_for_backward(jac)
        return disp

    @staticmethod
    def backward(ctx, g):
        jac, = ctx.saved_tensors  # jac[n, c, k] = d display_k / d y_c
        return torch.einsum('nk,nck->nc', g, jac), None


class ViewTransform:
    """Blender's view transform (exposure, look, tone curve, display) as the LUT baked at compile time."""

    def __init__(self, scene, cfg, device, groups=1, lib=None):
        self.lib = lib or kernels.library()
        self.device = torch.device(device)
        self.lut = torch.from_numpy(scene.lut.reshape(-1)).to(self.device).half()
        m = scene.meta
        self._base = [cfg['eps'], 1.0 / cfg['radiance_scale'], 0.0, 0.0, 0.0, m['lut']['size'], m['lut']['vmax'], m['lut']['a']]
        self.dither = float(m['render'].get('dither', 0.0))
        self.groups = groups
        self._plain = None

    def rcfg(self, sky=None, dither=0.0, linear=False, weights=None):
        """Parameters of the resolve kernels. weights: [groups][3], each light group's current colour x strength
        relative to the white light it was fitted with (default: all ones)."""
        plain = sky is None and dither == 0.0 and not linear and weights is None
        if plain and self._plain is not None:
            return self._plain
        v = list(self._base)
        if sky is not None:
            v[2:5] = sky
        w = [1.0] * (3 * self.groups) if weights is None else [float(x) for row in weights for x in row]
        t = torch.tensor(v + [dither, 1.0 if linear else 0.0, float(self.groups)] + w, dtype=torch.float32, device=self.device)
        if plain:
            self._plain = t
        return t

    def __call__(self, y):
        """Display colour of log-radiance y [n, 3]; differentiable."""
        return _ViewFn.apply(y, self)


class NeuralField(nn.Module):
    def __init__(self, scene, cfg, device='mps'):
        super().__init__()
        cfg = {**DEFAULTS, **cfg}
        self.cfg = cfg
        self.device = dev = torch.device(device)
        self.scene = scene
        self.parametric = bool(cfg['parametric'])
        self.lib = kernels.library(plain=not (self.parametric or cfg['env'] or cfg['lobe'] or cfg['follow']))
        self.G = len(cfg['groups']) if self.parametric else 1
        self.DG = 4 * len(cfg['varied']) if self.parametric else 0
        self.KT = cfg['time_octaves'] if (self.parametric and cfg['time_cells'] > 0) else 0
        up = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)

        # Scene tables. A static fit has one of each; a parametric one swaps in stacks (see use_state).
        frame = scene.pack_frame(None)
        self.T, self.Lg = up(frame['tri']), up(frame['lights'])
        self.bvh, self.bvh_order = up(frame['bvh']), up(frame['bvh_order'])
        self.C = up(scene.color_table(tint=self.parametric))
        self.M = up(scene.material_table(tint=self.parametric))
        self.Tref = up(scene.reference_table())
        self.use_views(1)
        self.sky_id = scene.sky_id
        self.NL = len(scene.light_order)
        tc = cfg['time_cells'] if self.parametric else 0
        extra = cfg['time_levels'] if tc > 0 else 0
        self.L, self.F, self.FT = cfg['levels'] + extra, cfg['features'], cfg['embed']
        self.D = kernels.n_dense(self.NL, cfg['octaves'], self.G, self.parametric, self.KT, self.DG)
        self.DIN = self.D + self.L * self.F + self.FT
        rows, entries = grid_levels(cfg['levels'], cfg['log2_table'], cfg['n_min'], cfg['n_max'], tc, extra, cfg['time_crossed'])
        self.lv = torch.tensor(rows, dtype=torch.int32, device=dev)
        lrows, lentries = grid_levels(cfg['lightmap_levels'], cfg['lightmap_log2'], cfg['n_min'], cfg['n_max'], tc, extra, cfg['time_crossed'])
        self.llv = torch.tensor(lrows, dtype=torch.int32, device=dev)
        self.LL = cfg['lightmap_levels'] + extra
        self.FL = cfg['lightmap_features'] if self.parametric else 3
        # The baked environment, as log-radiance; it belongs to the world's light group.
        world = [i for i, g in enumerate(cfg['groups'] or []) if g['kind'] == 'world'] if self.parametric else [0]
        use_env = bool(cfg['env']) and scene.env is not None and bool(world)
        if cfg['env'] and not use_env and (scene.env is None):
            raise RuntimeError('this fit needs the baked environment (%s/env.npy), which is missing' % scene.bundle)
        radiance = np.minimum(scene.env * cfg['radiance_scale'], scene.meta['lut']['vmax']) if use_env else np.zeros((1, 1, 3), np.float32)
        self.env = up(np.log(radiance + cfg['eps']).astype(np.float32).reshape(-1))
        icfg = [self.sky_id, self.NL, self.L, self.F, self.FT, 0, cfg['octaves'], self.LL, int(self.parametric), scene.n_tris + 1,
                frame['bvh'].shape[0], len(scene.materials) + 1, self.G, self.FL, self.KT, self.DG, frame['lights'].shape[0],
                scene.env.shape[1] if use_env else 0, world[0] if world else 0, int(cfg['lobe']), int(bool(cfg['follow']))]
        self.icfg_multi = torch.tensor(icfg, dtype=torch.int32, device=dev)
        icfg[5] = 1
        self.icfg_single = torch.tensor(icfg, dtype=torch.int32, device=dev)
        self._zero = torch.zeros(1, dtype=torch.int32, device=dev)
        self._zero_f = torch.zeros(4, dtype=torch.float32, device=dev)
        self.set_sky(cfg['sky'])

        self.ltable = nn.Parameter(torch.zeros(lentries, self.FL, device=dev) if not self.parametric
                                   else torch.empty(lentries, self.FL, device=dev).uniform_(-1e-4, 1e-4))
        self.ldec = nn.Linear(self.LL * self.FL, 3 * self.G).to(dev) if self.parametric else None
        self.table = nn.Parameter(torch.empty(entries, self.F, device=dev).uniform_(-1e-4, 1e-4))
        self.emb = nn.Parameter(torch.empty(scene.n_tris + 1, self.FT, device=dev).uniform_(-1e-4, 1e-4))
        dims = [self.DIN] + [cfg['width']] * cfg['depth'] + [3 * self.G]
        self.mlp = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])).to(dev)
        self._half = None
        self.view = ViewTransform(scene, cfg, dev, self.G, self.lib)

    @staticmethod
    def create(scene, domain, device='mps', sky=None, **overrides):
        cfg = dict(DEFAULTS)
        cfg.update(overrides)
        center, scale = scene.field_frame(domain)
        cfg.update(center=[float(c) for c in center], scale=float(scale), feature_set=kernels.FEATURE_SET,
                   radiance_scale=float(2.0 ** scene.meta['view'].get('exposure', 0.0)),
                   sky=sky or dict(mode='constant', color=[0.0, 0.0, 0.0]))
        return NeuralField(scene, cfg, device)

    # -- state -----------------------------------------------------------------------------------
    def use_views(self, n, vs=None, vt=None):
        """Per-view rows the kernels index by view: (frame slot, material state) and (time, global state)."""
        dev = self.device
        self.vs = torch.zeros(n, 2, dtype=torch.int32, device=dev) if vs is None else torch.as_tensor(np.asarray(vs), dtype=torch.int32).to(dev)
        self.vt = (torch.zeros(n, 1 + self.DG, dtype=torch.float32, device=dev) if vt is None
                   else torch.as_tensor(np.asarray(vt), dtype=torch.float32).to(dev))

    def use_state(self, tri, lights, bvh, order, materials, vs, vt):
        """Swap in scene tables: stacks over frames / material states for fitting, or one frame for rendering."""
        up = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        self.T, self.Lg, self.bvh, self.bvh_order, self.M = up(tri), up(lights), up(bvh), up(order), up(materials)
        self.use_views(len(vs), vs, vt)

    def global_state(self, materials):
        """The editable material parameters as one vector: what every shading point is told about the rest of
        the scene. materials: a material table [M + 1, MAT_STRIDE]."""
        idx = [i for i, m in enumerate(self.scene.materials) if m['name'] in self.cfg['varied']]
        if not idx:
            return np.zeros(0, np.float32)
        return np.concatenate([np.concatenate([np.sqrt(np.clip(materials[i, 8:11], 0, None)), materials[i, :1]]) for i in idx]).astype(np.float32)

    def set_sky(self, sky):
        """A constant background is a colour; anything else is learned (it lives in the lightmap at infinity)."""
        c = self.cfg
        c['sky'] = sky
        y = lambda rgb: [math.log(max(v, 0.0) * c['radiance_scale'] + c['eps']) for v in rgb]
        const = sky['mode'] == 'constant'
        base = y(sky['color']) if const else [0.0, 0.0, 0.0]
        per_group = []
        if self.parametric:   # each light group has its own share of a constant sky
            for g in (sky.get('groups') if const else None) or [[0.0, 0.0, 0.0]] * self.G:
                per_group += y(g)
        self.cfg_f = torch.tensor(list(c['center']) + [1.0 / c['scale']] + base + [0.0 if const else 1.0] + per_group,
                                  dtype=torch.float32, device=self.device)

    def _ldec(self):
        if self.ldec is None:
            return self._zero_f
        return torch.cat([self.ldec.weight, self.ldec.bias[:, None]], 1).contiguous()

    # -- fitting ---------------------------------------------------------------------------------
    def lightmap(self, tri, oidx, dirs, origins):
        """The view-independent first guess, per light group: log-radiance from (reference position, time)."""
        n = tri.shape[0]
        pos = torch.empty(n, 4, device=self.device)
        alb = torch.empty(n, 3, device=self.device)
        self.lib.nr_pos(pos, alb, tri, dirs, origins, oidx, self.vs, self.vt, self.T, self.Tref, self.C, self.M,
                        self.cfg_f, self.icfg_multi, threads=n)
        h = _GridLookup.apply(pos, self.ltable, self.lib, self.llv, self.LL, self.FL)
        if not self.parametric:
            return h.reshape(n, self.LL, 3).sum(1)
        return self.ldec(h) + alb.repeat(1, self.G)

    def forward(self, tri, oidx, dirs, origins, half=False):
        """Log-radiance y = log(radiance * scale + eps), [n, 3 * groups], for a batch of (triangle, ray) samples."""
        n = tri.shape[0]
        feat = torch.empty(n, self.D, device=self.device)
        pos = torch.empty(n, 4, device=self.device)
        with torch.no_grad():
            ldec = self._ldec()
        self.lib.nr_geo(feat, pos, tri, dirs, origins, oidx, self.vs, self.vt, self.T, self.Tref, self.C, self.M, self.Lg,
                        self.bvh, self.bvh_order, self.ltable, self.llv, ldec, self.env, self.cfg_f, self.icfg_multi, threads=n)
        x = torch.cat([feat, _GridLookup.apply(pos, self.table, self.lib, self.lv, self.L, self.F), self.emb[tri]], -1)
        if half:  # the MLP is most of a training step; half precision makes it cheaper on Apple GPUs
            x = x.half()
        for i, layer in enumerate(self.mlp):
            x = F.linear(x, layer.weight.half(), layer.bias.half()) if half else layer(x)
            if i + 1 < len(self.mlp):
                x = F.relu(x)
        return x.float()

    # -- rendering -------------------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self):
        """Freeze half-precision copies of the network for rendering."""
        self._half = [(l.weight.detach().half().contiguous(), l.bias.detach().half().contiguous()) for l in self.mlp]
        self._table = self.table.detach().contiguous()
        self._ltable = self.ltable.detach().contiguous()
        self._emb = self.emb.detach().contiguous()
        self._ld = self._ldec().detach().clone()

    @torch.no_grad()
    def infer(self, tri, dirs, origin, chunk=1 << 20):
        """Half-precision log-radiance [n, 3 * groups] for shading samples seen from a single origin."""
        if self._half is None:
            self.prepare()
        n = tri.shape[0]
        out = []
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            t, d = (tri, dirs) if (a == 0 and b == n) else (tri[a:b].contiguous(), dirs[a:b].contiguous())
            x = torch.empty(b - a, self.DIN, dtype=torch.float16, device=self.device)
            self.lib.nr_encode(x, t, d, origin, self._zero, self.vs, self.vt, self.T, self.Tref, self.C, self.M, self.Lg,
                               self.bvh, self.bvh_order, self._ltable, self.llv, self._ld, self.env, self._table, self._emb, self.lv,
                               self.cfg_f, self.icfg_single, threads=b - a)
            for i, (w, bias) in enumerate(self._half):
                x = F.linear(x, w, bias)
                if i + 1 < len(self._half):
                    x = F.relu_(x)
            out.append(x)
        return out[0] if len(out) == 1 else torch.cat(out)

    # -- persistence -----------------------------------------------------------------------------
    def save(self, path):
        torch.save(dict(cfg=self.cfg, state={k: v.detach().cpu() for k, v in self.state_dict().items()}), path)

    @staticmethod
    def load(scene, path, device='mps'):
        blob = torch.load(path, map_location='cpu', weights_only=True)
        if blob['cfg'].get('feature_set') != kernels.FEATURE_SET:
            raise RuntimeError('%s was fitted by a different version of Neuron Render; run `neuron-render fit --force`' % path)
        field = NeuralField(scene, blob['cfg'], device)
        field.load_state_dict(blob['state'])
        field.prepare()
        return field
