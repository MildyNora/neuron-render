"""neuron-render: fit a .blend once, then render it with a neural network."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace


def _bundle(args):
    if args.bundle:
        return args.bundle
    stem = os.path.splitext(os.path.abspath(args.blend))[0]
    parametric = getattr(args, 'parametric', False) or bool(getattr(args, 'vary', None))
    if not parametric and any(getattr(args, k, None) for k in ('frame', 'set', 'light')):
        parametric = True
    return stem + ('.parametric.neuron' if parametric else '.neuron')


def _fit(args, **kw):
    from .fit import fit
    return fit(args.blend, _bundle(args), log=lambda *a: print(*a, flush=True), **kw)


def _open(args):
    from .render import Renderer
    bundle = _bundle(args)
    if not os.path.exists(os.path.join(bundle, 'model.pt')):
        if bundle.endswith('.parametric.neuron'):
            sys.exit('no parametric fit for %s yet: run `neuron-render fit %s --vary time,lights,materials`' % (os.path.basename(args.blend), args.blend))
        print('no fitted model for %s yet; fitting once (the one-time cost)' % os.path.basename(args.blend))
        _fit(args)
    t = time.perf_counter()
    r = Renderer.open(bundle)
    return r, bundle, time.perf_counter() - t


def _quality(args):
    from .render import PRESETS
    q = PRESETS[args.quality]
    over = {k: getattr(args, k) for k in ('aa', 'rate', 'scale', 'lobe') if getattr(args, k) is not None}
    if args.full:
        over['full'] = True
    return replace(q, **over) if over else q


def _assignments(items):
    """['Paint.base_color=0.1,0.3,0.8', 'Paint.roughness=0.4'] -> {'Paint': {'base_color': [...], 'roughness': 0.4}}"""
    out = {}
    for item in items or []:
        key, _, value = item.partition('=')
        name, _, param = key.rpartition('.')
        if not name or not value:
            sys.exit('expected NAME.parameter=value, got %r' % item)
        nums = [float(x) for x in value.split(',')]
        out.setdefault(name, {})[param] = nums if len(nums) > 1 else nums[0]
    return out


def _state(renderer, args):
    """The scene state asked for on the command line (parametric fits only), checked against what was fitted."""
    state = dict(frame=getattr(args, 'frame', None), materials=_assignments(getattr(args, 'set', None)) or None,
                 lights=_assignments(getattr(args, 'light', None)) or None)
    if not renderer.parametric:
        if any(v is not None for v in state.values()):
            sys.exit('this fit is of one frozen scene: --frame, --set and --light need one made with `fit --vary`')
        return {}
    scene = renderer.scene
    if state['frame'] is not None and state['frame'] not in scene.frames:
        sys.exit('frame %d is not covered by this fit (frames %d..%d)' % (state['frame'], scene.frames[0], scene.frames[-1]))
    from .plan import load_plan
    editable = load_plan(scene.bundle).get('editable', {})
    for name, params in (state['materials'] or {}).items():
        if name not in editable:
            sys.exit('material %r is not editable in this fit (editable: %s)' % (name, ', '.join(editable) or 'none'))
        for k, v in params.items():
            if k not in editable[name]:
                sys.exit('%s.%s was not fitted (editable for %s: %s)' % (name, k, name, ', '.join(editable[name])))
            if (k == 'base_color') != isinstance(v, list) or (isinstance(v, list) and len(v) != 3):
                sys.exit('%s.%s takes %s' % (name, k, 'three numbers, r,g,b' if k == 'base_color' else 'one number'))
    groups = {g['name']: g for g in scene.groups}
    for name, params in (state['lights'] or {}).items():
        if name not in groups:
            sys.exit('no light group %r (have: %s)' % (name, ', '.join(groups)))
        allowed = ('energy', 'color') if groups[name]['kind'] == 'light' else ('strength',)
        for k, v in params.items():
            if k not in allowed:
                sys.exit('%s.%s is not a light setting (for %s: %s)' % (name, k, name, ', '.join(allowed)))
            if (k == 'color') != isinstance(v, list) or (isinstance(v, list) and len(v) != 3):
                sys.exit('%s.%s takes %s' % (name, k, 'three numbers, r,g,b' if k == 'color' else 'one number'))
    return state


def _camera(renderer, bundle, args, state):
    cam = renderer.scene.camera_at(renderer.scene.frame_index(state.get('frame'))) if renderer.parametric else renderer.scene.camera
    if args.azimuth is None and args.elevation is None and args.radius is None:
        return cam
    from .plan import load_plan
    from .scene import OrbitDomain
    dom = OrbitDomain.from_json(load_plan(bundle)['domain'])
    az, el, r = dom.hero
    pose = (args.azimuth if args.azimuth is not None else az, args.elevation if args.elevation is not None else el,
            args.radius if args.radius is not None else r)
    if not dom.contains(*pose):
        print('warning: this pose is outside the fitted view domain (elevation %.1f..%.1f, radius %.2f..%.2f); expect artefacts'
              % (*dom.elevation, *dom.radius), file=sys.stderr)
    return cam.with_matrix(dom.pose(*pose))


def _video(path, width, height, fps):
    import subprocess
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (width, height),
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', path], stdin=subprocess.PIPE)


def cmd_fit(args):
    vary = tuple(v for v in (args.vary or '').split(',') if v)
    for v in vary:
        if v not in {'time', 'lights', 'materials'}:
            sys.exit('--vary takes time, lights, materials (got %r)' % v)
    if args.force:
        path = os.path.join(_bundle(args), 'model.pt')
        if os.path.exists(path):
            os.remove(path)
    bundle, rep = _fit(args, views=args.views, steps=args.steps, teacher_samples=args.teacher_samples,
                       teacher_scale=args.teacher_scale, score=not args.no_score, azimuth_span=args.azimuth_span,
                       elevation_span=args.elevation_span, radius_span=args.radius_span, vary=vary, frames=args.frames)
    print('fitted %s in %.0f s (teacher %.0f s, training %.0f s) -> %s' % (
        os.path.basename(args.blend), rep['one_time_seconds'], rep['teacher_seconds'] or 0, rep['train']['seconds'], bundle))


def cmd_render(args):
    from PIL import Image
    r, bundle, load_s = _open(args)
    q, state = _quality(args), _state(r, args)
    cam = _camera(r, bundle, args, state)
    _, first = r.render(cam, q, **state)
    runs = sorted(r.render(cam, q, **state)[1]['total_ms'] for _ in range(max(args.repeat, 1)))
    img, info = r.render(cam, q, **state)
    t = time.perf_counter()
    Image.fromarray(img).save(args.output)
    save_ms = (time.perf_counter() - t) * 1e3
    print('%s  %dx%d  %s' % (args.output, info['width'], info['height'], q))
    print('frame %.1f ms (median of %d; %d shading samples) | first frame %.0f ms | model load %.2f s | png save %.0f ms'
          % (runs[len(runs) // 2], len(runs), info['samples'], first['total_ms'], load_s, save_ms))


def cmd_orbit(args):
    from PIL import Image
    from .plan import load_plan
    from .scene import OrbitDomain
    r, bundle, _ = _open(args)
    q, state = _quality(args), _state(r, args)
    dom = OrbitDomain.from_json(load_plan(bundle)['domain'])
    az, el, rad = dom.hero
    cam = r.scene.camera
    video = args.output.lower().endswith('.mp4')
    if video:
        pipe = _video(args.output, cam.width, cam.height, args.fps)
    else:
        os.makedirs(args.output, exist_ok=True)
    r.warmup(q, **state)
    ms = []
    for i in range(args.frames):
        img, info = r.render(cam.with_matrix(dom.pose(dom.sweep(i / args.frames), el, rad)), q, **state)
        ms.append(info['total_ms'])
        if video:
            pipe.stdin.write(img.tobytes())
        else:
            Image.fromarray(img).save(os.path.join(args.output, 'frame_%04d.png' % i))
    if video:
        pipe.stdin.close()
        pipe.wait()
    ms.sort()
    print('%d frames -> %s | %.1f ms per frame (median), %.0f fps' % (args.frames, args.output, ms[len(ms) // 2], 1e3 / ms[len(ms) // 2]))


def cmd_animate(args):
    """The file's own animation, frame by frame, through its own camera."""
    from PIL import Image
    r, bundle, _ = _open(args)
    if not r.parametric or r.scene.n_frames < 2:
        sys.exit('animate needs a fit that covers time: neuron-render fit %s --vary time,...' % args.blend)
    q, state = _quality(args), _state(r, args)
    frames = r.scene.frames
    if args.frames:
        a, b = (int(x) for x in args.frames.split(':'))
        frames = [f for f in frames if a <= f <= b]
    cam = r.scene.camera
    video = args.output.lower().endswith('.mp4')
    if video:
        pipe = _video(args.output, cam.width, cam.height, args.fps or r.scene.meta.get('fps', 24))
    else:
        os.makedirs(args.output, exist_ok=True)
    r.warmup(q, **dict(state, frame=frames[0]))
    t0 = time.perf_counter()
    ms = []
    for f in frames:
        img, info = r.render(None, q, **dict(state, frame=f))
        ms.append(info['total_ms'])
        if video:
            pipe.stdin.write(img.tobytes())
        else:
            Image.fromarray(img).save(os.path.join(args.output, 'frame_%04d.png' % f))
    total = time.perf_counter() - t0
    if video:
        pipe.stdin.close()
        pipe.wait()
    ms.sort()
    print('%d frames -> %s | %.1f ms per frame (median) | %.2f s for the animation' % (len(frames), args.output, ms[len(ms) // 2], total))


def cmd_bench(args):
    from . import bench
    from .render import PRESETS
    r, bundle, load_s = _open(args)
    out = dict(blend=os.path.abspath(args.blend), width=r.scene.camera.width, height=r.scene.camera.height, model_load_seconds=load_s)
    print('Neuron Render, steady state')
    out['neuron'] = bench.neural_frames(r, list(PRESETS), args.frames)
    print('Blender, the file as it is')
    frame = r.scene.meta.get('frame')
    out['blender'] = bench.blender_frame(args.blend, frame, runs=args.runs, keep=os.path.join(bundle, 'blender_frame.png'))
    if args.gpu:
        print('Blender, Cycles switched to the GPU')
        out['blender_gpu'] = bench.blender_frame(args.blend, frame, runs=args.runs, gpu=True)
    fit_json = os.path.join(bundle, 'fit.json')
    if os.path.exists(fit_json):
        rep = json.load(open(fit_json))
        out['one_time_seconds'] = rep.get('one_time_seconds')
        out['accuracy'] = rep.get('accuracy', {}).get('presets')
    with open(os.path.join(bundle, 'bench.json'), 'w') as f:
        json.dump(out, f, indent=1)
    # Blender is scored by its best run and the network by its median frame, so that whatever else the
    # machine is doing can only make the comparison harder for the network.
    base = out['blender']['best']
    print('\n%-22s %12s %10s' % ('', 'per frame', 'speed-up'))
    print('%-22s %10.2f s %10s   (best of %d; median %.2f s)' % ('Blender (as the file)', base, '1x', args.runs, out['blender']['median']))
    if 'blender_gpu' in out:
        g = out['blender_gpu']
        print('%-22s %10.2f s %9.1fx   (best of %d; median %.2f s)' % ('Blender (Cycles GPU)', g['best'], base / g['best'], args.runs, g['median']))
    for p, v in out['neuron'].items():
        acc = (out.get('accuracy') or {}).get(p)
        print('%-22s %9.1f ms %9.0fx%s' % ('Neuron ' + p, v['median_ms'], base * 1e3 / v['median_ms'],
                                           '   %.1f dB' % acc['psnr'] if acc else ''))


def cmd_info(args):
    bundle = _bundle(args)
    path = os.path.join(bundle, 'fit.json')
    if not os.path.exists(path):
        print('not fitted yet: run `neuron-render fit %s`' % args.blend)
        return
    rep = json.load(open(path))
    print('bundle      %s' % bundle)
    print('one-time    %.0f s (teacher %.0f s, training %.0f s)' % (rep['one_time_seconds'], rep['teacher_seconds'] or 0, rep['train']['seconds']))
    if rep.get('vary'):
        m = rep['model']
        print('covers      %s' % ', '.join(rep['vary']))
        print('lights      %s' % ', '.join(g['name'] for g in m.get('groups') or []))
        print('materials   %s' % ', '.join(m.get('varied') or []))
    for p, v in (rep.get('accuracy', {}).get('presets') or {}).items():
        print('%-11s PSNR %.2f dB  SSIM %.4f  (held-out, vs converged Cycles)' % (p, v['psnr'], v['ssim']))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='neuron-render', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, quality=True, state=True):
        p.add_argument('blend')
        p.add_argument('--bundle', help='where the fitted scene lives (default: next to the .blend, <name>.neuron)')
        if state:
            p.add_argument('--parametric', action='store_true', help='use the parametric fit (<name>.parametric.neuron)')
        if quality:
            p.add_argument('-q', '--quality', default='balanced', choices=['draft', 'fast', 'balanced', 'high', 'ultra'])
            p.add_argument('--aa', type=int, help='visibility samples per pixel axis (1-4)')
            p.add_argument('--rate', type=int, help='shading block size in pixels (1 = every pixel)')
            p.add_argument('--scale', type=float, help='internal resolution scale')
            p.add_argument('--full', action='store_true', help='evaluate the network at every visibility sample')
            p.add_argument('--lobe', type=int, help='rays per blurred metal reflection, 1-16 (fits that cover changes)')
        if quality and state:
            p.add_argument('--set', action='append', metavar='MATERIAL.param=value', help='edit a material: base_color=r,g,b or roughness=x (parametric fits)')
            p.add_argument('--light', action='append', metavar='LIGHT.param=value', help='edit a light: energy=x, color=r,g,b; World.strength=x (parametric fits)')

    p = sub.add_parser('fit', help='compile the scene, render teacher views with Cycles, train the network')
    common(p, quality=False, state=False)
    p.add_argument('--vary', default='', help='comma list of time, lights, materials: fit a parametric model that covers them')
    p.add_argument('--frames', help='a:b  frame range for --vary time (default: the scene\'s own range)')
    p.add_argument('--views', type=int, default=None, help='teacher views (default 200, or 480 for a parametric fit)')
    p.add_argument('--steps', type=int, default=None, help='training steps (default 3000, or 4000 for a parametric fit)')
    p.add_argument('--teacher-samples', type=int, default=64)
    p.add_argument('--teacher-scale', type=float, default=None, help='teacher resolution relative to the output (default: 0.5, or 1 if the scene has textures)')
    p.add_argument('--azimuth-span', type=float, default=360.0, help='degrees of orbit around the scene camera to fit (360 = all round)')
    p.add_argument('--elevation-span', type=float, default=6.0, help='degrees of elevation either side of the scene camera')
    p.add_argument('--radius-span', type=float, default=0.08, help='fraction of camera distance either side')
    p.add_argument('--no-score', action='store_true', help='skip the held-out accuracy check (saves the ground-truth renders)')
    p.add_argument('--force', action='store_true', help='retrain even if a model exists')
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser('render', help='render a frame with the network')
    common(p)
    p.add_argument('-o', '--output', default='neuron.png')
    p.add_argument('--frame', type=int, help='animation frame (parametric fits that cover time)')
    p.add_argument('--azimuth', type=float, help='orbit angle in degrees (default: the scene camera)')
    p.add_argument('--elevation', type=float)
    p.add_argument('--radius', type=float)
    p.add_argument('--repeat', type=int, default=20, help='frames to time')
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser('orbit', help='render a turntable (a folder of PNGs, or an .mp4 if ffmpeg is installed)')
    common(p)
    p.add_argument('-o', '--output', default='orbit.mp4')
    p.add_argument('--frame', type=int, help='animation frame to freeze (parametric fits)')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--fps', type=int, default=30)
    p.set_defaults(fn=cmd_orbit)

    p = sub.add_parser('animate', help='render the file\'s animation through its own camera (parametric fits that cover time)')
    common(p)
    p.add_argument('-o', '--output', default='animation.mp4')
    p.add_argument('--frames', help='a:b  frame range (default: everything that was fitted)')
    p.add_argument('--fps', type=int, default=None)
    p.set_defaults(fn=cmd_animate, parametric=True)

    p = sub.add_parser('bench', help='time Blender against the network on the scene camera')
    common(p, quality=False)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--gpu', action='store_true', help='also time Blender with Cycles on the GPU')
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser('info', help='what was fitted and how accurate it is')
    common(p, quality=False)
    p.set_defaults(fn=cmd_info)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
