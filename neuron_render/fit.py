"""Fitting: Cycles is the teacher, the neural field is the student.

compile (.blend -> scene)  ->  plan views  ->  teacher renders  ->  dataset  ->  train  ->  score on held-out views

A static fit learns one frozen scene over a band of camera poses. A parametric fit (vary=...) also covers
animation time, every light group's colour and strength, and the materials' base colour and roughness.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from . import blender, environment, plan as P
from .metrics import psnr, ssim
from .model import NeuralField
from .native import Rasterizer
from .render import PRESETS, Renderer
from .scene import OrbitDomain, Scene


# ---------------------------------------------------------------------------------------------
# data

class Dataset:
    """Every teacher pixel, expressed exactly as the renderer will reconstruct it: a filter-weighted
    mixture of shading samples (one per triangle visible inside the pixel). Training through that
    mixture is what lets sub-pixel geometry -- bevels, silhouettes -- receive the right supervision.

    Pixels are grouped by object and pre-shuffled on the GPU; a batch is a few contiguous slices.
    For a parametric fit each view also has a frame and a material state, a pixel has one target per light
    group, and the scene tables for every frame and state used are stacked on the GPU (field.use_state)."""

    def __init__(self, scene, bundle, plan, field, log=print, view=None, aa=4, sky='auto', max_pixels=40e6):
        device, eps, scale = field.device, field.cfg['eps'], field.cfg['radiance_scale']
        par, C = field.parametric, 3 * field.G
        ra = Rasterizer(scene.tri_verts)
        dom = OrbitDomain.from_json(plan['domain'])
        views = plan['train']
        n_obj = len(scene.meta['objects'])
        frame_of = [scene.frame_index(v['frame']) if par else None for v in views]
        verts = {}

        def geometry(fi):
            if fi not in verts:
                verts[fi] = np.ascontiguousarray(scene.geometry(fi)[0], np.float32)
            return verts[fi]

        def load(v):
            a = np.load(os.path.join(bundle, 'teacher', v['name'] + '.npy')).astype(np.float32)
            return a if a.ndim == 3 else np.ascontiguousarray(a.transpose(1, 2, 0, 3)).reshape(a.shape[1], a.shape[2], -1)

        def camera(i, rgb):
            M = P.view_matrix(scene, dom, views[i]) if par else dom.pose(*views[i]['pose'])
            return scene.camera_at(frame_of[i]).with_matrix(M).resized(rgb.shape[1], rgb.shape[0])

        # A background that never varies is a constant, not something to spend network capacity on
        # (the tolerance absorbs the denoiser bleeding a little light across silhouettes).
        sky_ref, sky_n, sky_off = None, 0, 0
        for i in range(0, len(views), max(len(views) // 12, 1)):
            rgb = load(views[i])
            miss = ra.ids(camera(i, rgb), 1, geometry(frame_of[i])) < 0
            if miss.any():
                s = rgb[miss]
                if sky_ref is None:
                    sky_ref = np.median(s, axis=0)
                sky_n += len(s)
                sky_off += int((np.abs(s - sky_ref).max(1) > 2e-3 * (1.0 + np.abs(sky_ref).max())).sum())
        nominal = np.asarray(scene.group_weights(), np.float32) if par else np.ones((1, 3), np.float32)
        if sky == 'constant' or (sky == 'auto' and (sky_n == 0 or sky_off < 0.005 * sky_n)):
            per_group = (sky_ref if sky_ref is not None else np.zeros(C)).reshape(-1, 3)
            self.sky = dict(mode='constant', color=[float(c) for c in (per_group * nominal).sum(0)], groups=per_group.tolist())
        else:
            self.sky = dict(mode='neural')
        sky_id = scene.sky_id if self.sky['mode'] == 'neural' else -1
        sky_term = np.asarray(self.sky.get('groups', np.zeros((C // 3, 3))), np.float32).reshape(-1) * scale + eps

        subw = scene.pixel_filter(aa)
        obj_of = np.concatenate([scene.tri_object, [n_obj]]).astype(np.int32)
        per = [dict(rgb=[], miss=[], cnt=[], tri=[], oidx=[], dirs=[], w=[]) for _ in range(n_obj + 1)]
        origins = np.zeros((len(views), 3), np.float32)
        rng_keep = np.random.default_rng(1)
        for i, v in enumerate(views):
            rgb = load(v)
            cam = camera(i, rgb)
            origins[i] = cam.basis()[0]
            r = ra.resolve(cam, subw, aa, 1, False, sky_id, tri=geometry(frame_of[i]))
            cnt, start = r['px_ref'] & 31, r['px_ref'] >> 5
            covered = cnt > 0
            # Views are plentiful; past max_pixels a random subset of each view's pixels says the same thing.
            share = max_pixels / (len(views) * cnt.size)
            keep = covered & (rng_keep.random(cnt.size) < share) if share < 1.0 else covered
            ent_keep = np.repeat(keep[covered], cnt[covered])   # entries are stored pixel after pixel
            cnt, start = cnt[keep], start[keep]
            pix_obj = obj_of[r['g_tri'][start]]                  # a pixel belongs to its first triangle's object
            ent_obj = np.repeat(pix_obj, cnt)
            rgbk, missk = rgb.reshape(-1, C)[keep], r['px_miss'][keep]
            tri_e, dir_e, w_e = r['g_tri'][ent_keep], r['g_dir'][ent_keep], r['e_w'][ent_keep]
            for o in np.unique(pix_obj):
                mp, me = pix_obj == o, ent_obj == o
                d = per[o]
                d['rgb'].append(rgbk[mp]); d['miss'].append(missk[mp]); d['cnt'].append(cnt[mp])
                d['tri'].append(tri_e[me]); d['dirs'].append(dir_e[me]); d['w'].append(w_e[me])
                d['oidx'].append(np.full(int(me.sum()), i, np.int32))

        names = [o['name'] for o in scene.meta['objects']] + ['<sky>']
        rng = np.random.default_rng(0)
        up = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.strata = []
        for o, d in enumerate(per):
            if not d['cnt']:
                continue
            cnt = np.concatenate(d['cnt']).astype(np.int64)
            n_pix, n_ent = len(cnt), int(cnt.sum())
            perm = rng.permutation(n_pix)
            c = cnt[perm]
            off = np.concatenate([[0], np.cumsum(c)])
            src = np.repeat((np.cumsum(cnt) - cnt)[perm], c) + (np.arange(n_ent) - np.repeat(off[:-1], c))
            y = torch.log(up(np.concatenate(d['rgb'])[perm]).clamp_min(0.0) * scale + eps)
            disp = torch.cat([view(y[a:a + (1 << 22)]) for a in range(0, n_pix, 1 << 22)]) if (view is not None and not par) else None
            self.strata.append(dict(
                name=names[o], n=n_pix, n_entries=n_ent, off=off, y=y, disp=disp, miss=up(np.concatenate(d['miss'])[perm]),
                tri=up(np.concatenate(d['tri'])[src]), oidx=up(np.concatenate(d['oidx'])[src]), dirs=up(np.concatenate(d['dirs'])[src]),
                w=up(np.concatenate(d['w'])[src]), pix=up(np.repeat(np.arange(n_pix, dtype=np.int32), c))))
            log('  %-16s %10d pixels, %10d shading samples' % (names[o], n_pix, n_ent))
        self.origins = torch.from_numpy(origins).to(device)
        self.sky_term = torch.from_numpy(sky_term).to(device)
        self.nominal = torch.from_numpy(nominal).to(device) if par else None
        self.n = sum(s['n'] for s in self.strata)
        self.n_views = len(views)

        if not par:
            field.use_views(len(views))
            return
        # Stack the scene tables of every frame and material state the views use.
        slots = sorted(set(frame_of))
        packed = [scene.pack_frame(fi) for fi in slots]
        base = scene.material_table(tint=True)
        states, vs, vt = [base], [], []
        for i, v in enumerate(views):
            ms = 0
            if v.get('materials'):
                states.append(scene.material_table(v['materials'], tint=True))
                ms = len(states) - 1
            vs.append([slots.index(frame_of[i]), ms])
            vt.append(np.concatenate([[scene.time_of(frame_of[i])], field.global_state(states[ms])]))
        field.use_state(np.stack([f['tri'] for f in packed]), np.stack([f['lights'] for f in packed]),
                        np.stack([f['bvh'] for f in packed]), np.stack([f['bvh_order'] for f in packed]),
                        np.stack(states), np.asarray(vs, np.int32), np.asarray(vt, np.float32))
        log('  %d frames and %d material states stacked on the GPU' % (len(slots), len(states)))


class Batcher:
    """Slices batches out of the strata, steering samples towards whichever object is currently worst."""

    def __init__(self, data, batch):
        self.data, self.batch = data, batch
        self.ptr = [0] * len(data.strata)
        self.base = np.array([math.sqrt(s['n']) for s in data.strata])
        self.share = self.base / self.base.sum()

    def next(self):
        """(tri, oidx, dirs, w, seg) per shading sample; (y, disp, miss) per pixel; pixels per stratum."""
        ent, pix, sizes, base = [], [], [], 0
        for i, s in enumerate(self.data.strata):
            b = min(max(int(round(self.batch * self.share[i])), 1), s['n'])
            if self.ptr[i] + b > s['n']:
                self.ptr[i] = 0
            a = self.ptr[i]
            self.ptr[i] += b
            e0, e1 = int(s['off'][a]), int(s['off'][a + b])
            ent.append((s['tri'][e0:e1], s['oidx'][e0:e1], s['dirs'][e0:e1], s['w'][e0:e1], s['pix'][e0:e1] + (base - a)))
            pix.append((s['y'][a:a + b], s['disp'][a:a + b] if s['disp'] is not None else s['y'][a:a + b], s['miss'][a:a + b]))
            sizes.append(b)
            base += b
        return [torch.cat([p[k] for p in ent]) for k in range(5)], [torch.cat([p[k] for p in pix]) for k in range(3)], sizes

    def update(self, losses):
        w = self.base * (np.asarray(losses) + 1e-3)
        w = np.maximum(w / w.sum(), 0.05)
        self.share = w / w.sum()


# ---------------------------------------------------------------------------------------------
# training

GRAD_SCALE = 4096.0


def train_lightmap(field, data, steps=None, batch=1 << 18, log=print):
    """Stage one: bake what does not depend on the viewpoint. Position (and time) -> log-radiance per light
    group, fitted to the same teacher pixels. The result is frozen and becomes (a) the network's first guess
    at every shading point and (b) what its traced reflection and refraction rays see where they land."""
    steps = steps or (1000 if field.parametric else 600)   # a lightmap that also changes over time takes longer
    params = [field.ltable] + (list(field.ldec.parameters()) if field.ldec is not None else [])
    opt = torch.optim.Adam([dict(params=[field.ltable], lr=2e-2, eps=1e-15)]
                           + ([dict(params=field.ldec.parameters(), lr=5e-3, eps=1e-8)] if field.ldec is not None else []), betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(s / steps, 1.0))))
    batcher = Batcher(data, batch)
    t0 = time.perf_counter()
    for step in range(steps):
        (tri, oidx, dirs, w, seg), (y, _, miss), sizes = batcher.next()
        loss = (field.lightmap(tri, oidx, dirs, data.origins) - y[seg]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            log('  lightmap %4d/%d  loss %.4f' % (step, steps, loss.item()))
    for p in params:
        p.requires_grad_(False)
    return dict(seconds=time.perf_counter() - t0, steps=steps, loss=loss.item())


def train(field, data, steps=4000, batch=1 << 17, lr_grid=1e-2, lr_mlp=2e-3, log=print, on_eval=None, eval_every=500,
          display_weight=0.0, log_weight=1.0, half=False):
    """L1 on log-radiance (per light group for a parametric fit, plus the composite at the scene's own light
    settings). display_weight > 0 adds squared error measured on the display, through Blender's own view
    transform; on the scenes tried so far that traded structure in the shadows for nothing, so it is off."""
    opt = torch.optim.Adam([dict(params=[field.table, field.emb], lr=lr_grid, eps=1e-15),
                            dict(params=field.mlp.parameters(), lr=lr_mlp, eps=1e-8)], betas=(0.9, 0.99))
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(s / steps, 1.0)))))
    batcher = Batcher(data, batch)
    eps = field.cfg['eps']
    composite = None
    if data.nominal is not None:   # what the viewer sees: all light groups at their file settings
        composite = lambda t: torch.log(((torch.exp(t) - eps).clamp_min(0.0).reshape(t.shape[0], -1, 3) * data.nominal).sum(1) + eps)
    t0 = time.perf_counter()
    history = []
    for step in range(steps):
        (tri, oidx, dirs, w, seg), (y, disp, miss), sizes = batcher.next()
        # Shading samples -> pixels, mixed in linear radiance exactly as the renderer's resolve does.
        mix = torch.zeros(y.shape[0], y.shape[1], device=y.device).index_add_(0, seg, w[:, None] * torch.exp(field(tri, oidx, dirs, data.origins, half)))
        pred = torch.log(mix + miss[:, None] * data.sky_term + 1e-12)
        err = log_weight * (pred - y).abs().mean(1)
        if composite is not None:
            err = err + (composite(pred) - composite(y)).abs().mean(1)
        if display_weight > 0 and composite is None:
            err = err + display_weight * (field.view(pred) - disp).square().mean(1)
        loss = err.mean()
        opt.zero_grad(set_to_none=True)
        # Adam is indifferent to the gradient's scale; half precision is not (tiny values underflow).
        (loss * GRAD_SCALE if half else loss).backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            seg_loss, a = [], 0
            for b in sizes:
                seg_loss.append(float(err[a:a + b].detach().mean()))
                a += b
            batcher.update(seg_loss)
            if step % 250 == 0 or step == steps - 1:
                log('  step %5d/%d  loss %.4f  [%s]  %.0fs' % (step, steps, loss.item(), ' '.join(
                    '%s %.4f' % (s['name'][:10], l) for s, l in zip(data.strata, seg_loss)), time.perf_counter() - t0))
        if on_eval and ((step + 1) % eval_every == 0 or step == steps - 1):
            field.prepare()
            history.append(dict(step=step + 1, seconds=time.perf_counter() - t0, **on_eval()))
            log('  eval @%d: %s' % (step + 1, {k: round(v, 3) for k, v in history[-1].items() if k not in {'step', 'seconds'}}))
    if dev_is_mps(field.device):
        torch.mps.synchronize()
    field.prepare()
    return dict(seconds=time.perf_counter() - t0, steps=steps, batch=batch, history=history)


def dev_is_mps(device):
    return torch.device(device).type == 'mps'


# ---------------------------------------------------------------------------------------------
# scoring

def load_reference(renderer, bundle, sub, name):
    path = os.path.join(bundle, sub, name + '.npy')
    return renderer.display(np.load(path).astype(np.float32)) if os.path.exists(path) else None


def test_state(renderer, plan, v):
    """Keyword arguments that make renderer.render reproduce a planned test view."""
    if not renderer.parametric:
        return dict(camera=renderer.scene.camera.with_matrix(OrbitDomain.from_json(plan['domain']).pose(*v['pose'])))
    scene = renderer.scene
    fi = scene.frame_index(v['frame'])
    cam = scene.camera_at(fi).with_matrix(P.view_matrix(scene, OrbitDomain.from_json(plan['domain']), v))
    return dict(camera=cam, frame=v['frame'], materials=v.get('materials') or None, lights=v.get('lights') or None)


def evaluate(renderer, bundle, plan, presets=tuple(PRESETS), log=print):
    """PSNR / SSIM against converged Cycles on the held-out test states, in display space (what you see)."""
    out = dict(views={}, presets={})
    for v in plan['test']:
        truth = load_reference(renderer, bundle, 'truth', v['name'])
        if truth is None:
            continue
        state = test_state(renderer, plan, v)
        row = {}
        for p in presets:
            img, _ = renderer.render(quality=p, output='display', **state)
            row[p] = dict(psnr=psnr(img, truth), ssim=ssim(img, truth))
        ordinary = load_reference(renderer, bundle, 'ordinary', v['name'])
        if ordinary is not None:
            row['blender'] = dict(psnr=psnr(ordinary, truth), ssim=ssim(ordinary, truth))
        out['views'][v['name']] = row
    for p in list(presets) + ['blender']:
        rows = [v[p] for v in out['views'].values() if p in v]
        if rows:
            out['presets'][p] = dict(psnr=float(np.mean([r['psnr'] for r in rows])), ssim=float(np.mean([r['ssim'] for r in rows])))
            log('  %-9s PSNR %.2f dB  SSIM %.4f  (mean of %d held-out states)' % (p, out['presets'][p]['psnr'], out['presets'][p]['ssim'], len(rows)))
    return out


# ---------------------------------------------------------------------------------------------
# self-check

def verify_geometry(scene, blend, bundle, log=print):
    """Ask Cycles where its camera rays land (position pass, one sample through each pixel centre) and
    compare with our own visibility. If the compiled scene is faithful this agrees to float precision."""
    cam = scene.camera
    job = dict(out=os.path.join(bundle, 'verify'), mode='teacher', width=cam.width, height=cam.height, samples=1,
               adaptive_threshold=0, denoise=False, passes=True, point_sample=True,
               views=[dict(name='hero', matrix=np.asarray(cam.matrix).tolist())])
    blender.run_job(blend, job)
    pos = np.load(os.path.join(bundle, 'verify', 'hero.pos.npy'))
    ids = Rasterizer(scene.tri_verts).ids(cam, 1)
    hit = ids >= 0
    theirs = np.abs(pos).sum(-1) > 0
    n, d = scene.planes()
    o = cam.basis()[0]
    dirs = cam.pixel_dirs().astype(np.float64)[hit]
    t = (d[ids[hit]] - n[ids[hit]] @ o) / (n[ids[hit]] * dirs).sum(1)
    err = np.linalg.norm(o + dirs * t[:, None] - pos[hit], axis=1)
    depth = np.maximum(t, 1e-9)
    out = dict(coverage_agreement=float((hit == theirs).mean()), position_agreement=float((err < 1e-3 * depth).mean()),
               median_error=float(np.median(err)))
    log('  visibility matches Cycles on %.3f%% of pixels; hit points agree on %.3f%% (median error %.1e)' % (
        100 * out['coverage_agreement'], 100 * out['position_agreement'], out['median_error']))
    if out['coverage_agreement'] < 0.995 or out['position_agreement'] < 0.99:
        log('  WARNING: the compiled scene does not reproduce what Cycles sees; expect a poor fit')
    return out


# ---------------------------------------------------------------------------------------------
# orchestration

def _have(bundle, sub, names):
    return all(os.path.exists(os.path.join(bundle, sub, n + '.npy')) for n in names)


def _teacher_seconds(bundle, sub):
    path = os.path.join(bundle, sub, 'timing.json')
    return json.load(open(path))['total'] if os.path.exists(path) else None


def fit(blend, bundle=None, views=None, test_views=6, teacher_scale=None, teacher_samples=64, truth_samples=1024,
        steps=None, batch=1 << 17, score=True, verify=True, device='mps', log=print, azimuth_span=360.0,
        elevation_span=6.0, radius_span=0.08, vary=(), frames=None, **model):
    """vary: any of 'time', 'lights', 'materials' makes this a parametric fit (see the module docstring)."""
    blend = os.path.abspath(blend)
    par = bool(vary)
    bundle = bundle or os.path.splitext(blend)[0] + ('.parametric.neuron' if par else '.neuron')
    views = views or (480 if par else 200)
    steps = steps or (4000 if par else 3000)
    os.makedirs(bundle, exist_ok=True)
    report = dict(blend=blend, vary=list(vary))

    stamp = os.path.join(bundle, 'compile.json')
    if not (os.path.exists(os.path.join(bundle, 'scene.npz')) and os.path.exists(stamp)):
        t = time.perf_counter()
        log('compiling %s' % os.path.basename(blend))
        args = ['--out', bundle]
        if 'time' in vary:
            args += ['--frames', frames or 'scene']
        for note in blender.run_script(blend, 'compile_scene.py', args):
            log('  ' + note)
        scene = Scene(bundle)
        check = verify_geometry(scene, blend, bundle, log) if verify else None
        with open(stamp, 'w') as f:
            json.dump(dict(seconds=time.perf_counter() - t, geometry=check), f)
    scene = Scene(bundle)
    report.update(zip(('compile_seconds', 'geometry'), (lambda d: (d['seconds'], d['geometry']))(json.load(open(stamp)))))
    for w in scene.meta['warnings']:
        log('  note: ' + w)

    if not os.path.exists(os.path.join(bundle, 'views.json')):
        if par:
            P.save_plan(bundle, P.make_parametric_plan(scene, views, vary=vary, elevation_span=elevation_span, radius_span=radius_span))
        else:
            P.save_plan(bundle, P.make_plan(scene, views, test_views, azimuth_span=azimuth_span, elevation_span=elevation_span,
                                            radius_span=radius_span))
    plan = P.load_plan(bundle)
    W, H = scene.camera.width, scene.camera.height
    if teacher_scale is None:
        # One shading sample per triangle per pixel cannot resolve detail *inside* a triangle, so a texture is
        # learned pre-filtered to the teacher's pixel size: teach at the output resolution when there is one.
        teacher_scale = 1.0 if scene.textured().any() else 0.5
    progress = lambda tag: (lambda i, n, dt: log('  %s %d/%d (%.2fs)' % (tag, i, n, dt)) if i % 20 == 0 or i == n else None)
    job = (lambda split, sub, mode, w, h, **kw: P.parametric_job(scene, bundle, plan, split, sub, mode, w, h, **kw)) if par else \
          (lambda split, sub, mode, w, h, **kw: P.job(bundle, plan, split, sub, mode, w, h, **kw))

    if not _have(bundle, 'teacher', [v['name'] for v in plan['train']]):
        log('teacher: %d Cycles views at %dx%d, %d spp' % (len(plan['train']), W * teacher_scale, H * teacher_scale, teacher_samples))
        extra = dict(lightgroups=scene.groups) if par else {}
        blender.run_job(blend, job('train', 'teacher', 'teacher', W * teacher_scale, H * teacher_scale, samples=teacher_samples,
                                   adaptive_threshold=0.02, denoise=True, **extra), progress('teacher'))
    report['teacher_seconds'] = _teacher_seconds(bundle, 'teacher')
    if score and not _have(bundle, 'truth', [v['name'] for v in plan['test']]):
        log('truth: %d held-out states at %d spp (for scoring only)' % (len(plan['test']), truth_samples))
        blender.run_job(blend, job('test', 'truth', 'teacher', W, H, samples=truth_samples, adaptive_threshold=0, denoise=False), progress('truth'))
    if score and not _have(bundle, 'ordinary', [v['name'] for v in plan['test']]):
        log('yardstick: the same held-out states rendered by Blender with the file\'s own settings')
        blender.run_job(blend, job('test', 'ordinary', 'scene', W, H, save_png=True), progress('ordinary'))

    domain = OrbitDomain.from_json(plan['domain'])
    if par:
        # Once things move or change, the network can no longer memorise place by place what its traced hints get
        # wrong, so the hints have to be right: the world shader itself, baked, for rays that leave the scene; a
        # proper lobe of rays for the blurred reflections of rough metal; and one more bounce where a traced ray
        # lands on metal or glass.
        if scene.env is None:
            log('environment')
            environment.bake(blend, bundle, frame=scene.frames[scene.ref_index], log=log)
            scene = Scene(bundle)
        # Time slices: at most one per two frames. With one per frame, a frame that was held out for scoring would
        # read a slice no teacher view ever touched.
        model = dict(dict(parametric=True, groups=scene.groups, varied=list(plan['editable']), log2_table=21, lightmap_log2=20,
                          time_cells=max(2, min((scene.n_frames - 1) // 2, 40)) if 'time' in vary and scene.n_frames > 1 else 0,
                          time_crossed=True, env=True, lobe=16, follow=True), **model)
    field = NeuralField.create(scene, domain, device, **model)
    log('dataset')
    t = time.perf_counter()
    data = Dataset(scene, bundle, plan, field, log, field.view)
    report['dataset_seconds'] = time.perf_counter() - t
    field.set_sky(data.sky)
    renderer = Renderer(scene, field)
    log('training on %d pixels from %d views (%d network inputs, %.1fM parameters)' % (
        data.n, data.n_views, field.DIN, sum(p.numel() for p in field.parameters()) / 1e6))

    hero = load_reference(renderer, bundle, 'truth', 'hero') if score else None
    hero_state = test_state(renderer, plan, plan['test'][0])
    on_eval = (lambda: dict(hero_psnr=psnr(renderer.render(quality='high', output='display', **hero_state)[0], hero))) if hero is not None else None
    report['lightmap'] = train_lightmap(field, data, log=log)
    report['train'] = train(field, data, steps, batch, log=log, on_eval=on_eval)
    field.save(os.path.join(bundle, 'model.pt'))
    del data
    renderer.reset()

    if score:
        log('held-out accuracy vs converged Cycles')
        report['accuracy'] = evaluate(renderer, bundle, plan, log=log)
    report['one_time_seconds'] = (report['compile_seconds'] + (report['teacher_seconds'] or 0) + report['dataset_seconds']
                                  + report['lightmap']['seconds'] + report['train']['seconds'])
    report['model'] = {k: v for k, v in field.cfg.items()}
    with open(os.path.join(bundle, 'fit.json'), 'w') as f:
        json.dump(report, f, indent=1)
    return bundle, report
