"""Which camera poses the teacher renders: training views and held-out test views."""
from __future__ import annotations

import json
import os

import numpy as np

from .scene import OrbitDomain


def make_plan(scene, n_train=200, n_test=6, seed=0, min_sep=3.0, azimuth_span=360.0, elevation_span=6.0, radius_span=0.08):
    """Test views are the .blend's own camera ('hero') plus poses spread around the domain; training
    views are sampled from the domain but kept min_sep degrees away from every test view, so the
    reported accuracy is always for viewpoints the network has never seen."""
    dom = scene.default_domain(elevation_span, radius_span, azimuth_span)
    rng = np.random.default_rng(seed + 1)
    test = [dom.hero]
    lo, hi = dom.azimuth
    for k in range(1, n_test):
        az = lo + (k - 1 + rng.uniform(0.25, 0.75)) / max(n_test - 1, 1) * (hi - lo)
        test.append((az, rng.uniform(*dom.elevation), rng.uniform(*dom.radius)))
    if hi - lo < 359.0:  # a partial orbit starts at one end of its range; keep the extra test views off the hero
        test = [test[0]] + [t for t in test[1:] if dom.separation(t, dom.hero) >= 2 * min_sep]
    train = dom.sample(n_train, seed, exclude=test, min_sep=min_sep)
    return dict(domain=dom.to_json(), min_sep=min_sep,
                test=[dict(name='hero' if i == 0 else 'test%d' % i, pose=list(p)) for i, p in enumerate(test)],
                train=[dict(name='t%04d' % i, pose=list(p)) for i, p in enumerate(train)])


def save_plan(bundle, plan):
    with open(os.path.join(bundle, 'views.json'), 'w') as f:
        json.dump(plan, f, indent=1)


def load_plan(bundle):
    with open(os.path.join(bundle, 'views.json')) as f:
        return json.load(f)


def views(plan, split):
    """[(name, 4x4 matrix)] for 'train' or 'test'."""
    dom = OrbitDomain.from_json(plan['domain'])
    return [(v['name'], dom.pose(*v['pose'])) for v in plan[split]]


def job(bundle, plan, split, sub, mode, width, height, **settings):
    return dict(out=os.path.join(bundle, sub), mode=mode, width=int(width), height=int(height),
                views=[dict(name=n, matrix=m.tolist()) for n, m in views(plan, split)], **settings)


# ---------------------------------------------------------------------------------------------
# parametric plans: every view also has a frame and a material state

def random_edit(rng, editable, change=0.5):
    """A random material state: each material is left alone or given a new base colour and roughness."""
    import colorsys
    out = {}
    for name, params in editable.items():
        if rng.random() >= change:
            continue
        e = {}
        if 'base_color' in params:
            e['base_color'] = [float(c) for c in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 1.0), rng.uniform(0.12, 0.95))]
        if 'roughness' in params:
            e['roughness'] = float(rng.uniform(0.03, 1.0))
        out[name] = e
    return out


def random_lights(rng, groups):
    """A random light state: every lamp's strength and colour, and the world's strength, moved about."""
    import colorsys
    out = {}
    for g in groups:
        if g['kind'] == 'light':
            out[g['name']] = dict(energy=float(g['energy'] * rng.uniform(0.2, 2.2)),
                                  color=[float(c) for c in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.7), 1.0)])
        elif g['kind'] == 'world' and not g.get('strength_linked'):
            out[g['name']] = dict(strength=float(g['strength'] * rng.uniform(0.3, 2.0)))
    return out


def make_parametric_plan(scene, n_train=480, n_held=5, n_edits=4, seed=0, min_sep=3.0, vary=('time', 'lights', 'materials'),
                         elevation_span=6.0, radius_span=0.08, azimuth_margin=30.0, original_share=0.4):
    """Training views are spread over camera poses, animation frames and material states. Test states are:
    the scene camera at the reference frame ('hero'); the animation's own camera at frames that are withheld
    from training entirely; and withheld frames with materials and lights edited. Lights never need to vary
    during training (their effect is linear, see the light groups), only in the tests."""
    rng = np.random.default_rng(seed + 11)
    timed = 'time' in vary and scene.n_frames > 1
    dom = scene.path_domain(elevation_span, radius_span, azimuth_margin) if timed else scene.default_domain(elevation_span, radius_span)
    editable = scene.editable() if 'materials' in vary else {}
    frames = scene.frames if timed else [scene.frames[scene.ref_index]]
    ref = scene.frames[scene.ref_index]
    held = []
    if timed and len(frames) > 2 * n_held:
        held = sorted({frames[int(round((k + 0.5) / n_held * (len(frames) - 1)))] for k in range(n_held)} - {ref})
    test = [dict(name='hero', frame=ref, camera='scene', materials={}, lights={})]
    for f in held:
        test.append(dict(name='frame%03d' % f, frame=f, camera='scene', materials={}, lights={}))
    pool = held or [ref]
    for k in range(n_edits if (editable or 'lights' in vary) else 0):
        test.append(dict(name='edit%d' % k, frame=pool[k % len(pool)], camera='scene',
                         materials=random_edit(rng, editable, 0.6) if editable else {},
                         lights=random_lights(rng, scene.groups) if 'lights' in vary else {}))
    for t in test:
        t['pose'] = list(scene.camera_pose(dom, scene.frame_index(t['frame'])))
    poses = dom.sample(n_train, seed, exclude=[tuple(t['pose']) for t in test], min_sep=min_sep)
    usable = [f for f in frames if f not in held]
    order = []
    while len(order) < len(poses):
        order += [usable[i] for i in rng.permutation(len(usable))]
    train = []
    for i, p in enumerate(poses):
        mats = {} if (not editable or rng.random() < original_share) else random_edit(rng, editable)
        train.append(dict(name='t%04d' % i, pose=list(p), frame=int(order[i]), materials=mats))
    return dict(domain=dom.to_json(), min_sep=min_sep, vary=list(vary), editable=editable, held_frames=held, test=test, train=train)


def view_matrix(scene, dom, v):
    """Camera matrix of a planned view: the scene's own (animated) camera, or a pose in the domain."""
    if v.get('camera') == 'scene':
        return np.asarray(scene.camera_at(scene.frame_index(v['frame'])).matrix)
    return dom.pose(*v['pose'])


def parametric_job(scene, bundle, plan, split, sub, mode, width, height, names=None, **settings):
    """A teacher job whose views carry a frame, material overrides and light overrides."""
    dom = OrbitDomain.from_json(plan['domain'])
    views = []
    for v in plan[split]:
        if names is not None and v['name'] not in names:
            continue
        cam = scene.camera_at(scene.frame_index(v['frame']))
        views.append(dict(name=v['name'], matrix=view_matrix(scene, dom, v).tolist(), lens=cam.lens, frame=int(v['frame']),
                          materials=v.get('materials', {}), lights=v.get('lights', {})))
    return dict(out=os.path.join(bundle, sub), mode=mode, width=int(width), height=int(height), views=views,
                editable=plan['editable'], **settings)
