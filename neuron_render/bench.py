"""Timing: Blender rendering the frame the ordinary way vs Neuron Render."""
from __future__ import annotations

import os
import re
import statistics
import subprocess
import tempfile
import time

from .blender import find_blender

_GPU = ("import bpy\n"
        "p = bpy.context.preferences.addons['cycles'].preferences\n"
        "for k in ('METAL', 'OPTIX', 'CUDA', 'HIP', 'ONEAPI'):\n"
        "    try:\n"
        "        p.compute_device_type = k\n"
        "    except TypeError:\n"
        "        continue\n"
        "    p.get_devices()\n"
        "    if any(d.type == k for d in p.devices):\n"
        "        for d in p.devices: d.use = d.type == k\n"
        "        bpy.context.scene.cycles.device = 'GPU'\n"
        "        break\n")


def _seconds(stamp):
    s = 0.0
    for part in stamp.split(':'):
        s = s * 60.0 + float(part)
    return s


def blender_frame(blend, frame=None, runs=3, gpu=False, keep=None, blender=None, log=print):
    """Render one frame with `blender -b file -f N`, exactly as the file is set up (gpu=True only swaps
    the Cycles device). Returns Blender's own reported render time and the process wall time per run."""
    blender = blender or find_blender()
    render, wall = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(runs):
            cmd = [blender, '-b', blend] + (['--python-expr', _GPU] if gpu else []) + ['-o', os.path.join(tmp, 'frame_####'), '-F', 'PNG']
            cmd += ['-f', str(frame)] if frame is not None else ['-f', '1']
            t = time.perf_counter()
            out = subprocess.run(cmd, capture_output=True, text=True, errors='replace').stdout
            wall.append(time.perf_counter() - t)
            m = re.findall(r'Time: ([\d:.]+) \(Saving: ([\d:.]+)\)', out)
            if not m:
                raise RuntimeError('could not find the render time in Blender output:\n' + out[-2000:])
            render.append(_seconds(m[-1][0]) - _seconds(m[-1][1]))
            log('  blender %s run %d: render %.2f s (process %.2f s)' % ('gpu' if gpu else 'as-is', i + 1, render[-1], wall[-1]))
            if keep and i == runs - 1:
                pngs = sorted(f for f in os.listdir(tmp) if f.endswith('.png'))
                if pngs:
                    os.replace(os.path.join(tmp, pngs[-1]), keep)
    return dict(render_seconds=render, wall_seconds=wall, median=statistics.median(render), best=min(render),
                device='GPU' if gpu else 'file')


def neural_frames(renderer, presets, frames=30, log=print):
    """Steady-state time per frame (visibility + shading + resolve, pixels in host memory) per preset."""
    out = {}
    for p in presets:
        renderer.warmup(p, 5)
        runs = [renderer.render(quality=p)[1] for _ in range(frames)]
        ms = sorted(r['total_ms'] for r in runs)
        out[p] = dict(median_ms=ms[len(ms) // 2], best_ms=ms[0], p90_ms=ms[int(len(ms) * 0.9)], samples=runs[0]['samples'])
        log('  neuron %-9s %.1f ms  (%d shading samples)' % (p, out[p]['median_ms'], out[p]['samples']))
    return out


def blender_animation(blend, out_dir, gpu=False, blender=None, log=print):
    """Render the file's whole animation with `blender -b file -a`, as the file is set up (gpu=True only swaps
    the Cycles device). Frames land in out_dir; returns Blender's own reported render time per frame."""
    import json
    stamp = os.path.join(out_dir, 'timing.json')
    if os.path.exists(stamp):
        return json.load(open(stamp))
    os.makedirs(out_dir, exist_ok=True)
    cmd = [blender or find_blender(), '-b', blend] + (['--python-expr', _GPU] if gpu else []) + ['-o', os.path.join(out_dir, 'f_####'), '-F', 'PNG', '-a']
    t = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, errors='replace').stdout
    wall = time.perf_counter() - t
    times = [_seconds(a) - _seconds(b) for a, b in re.findall(r'Time: ([\d:.]+) \(Saving: ([\d:.]+)\)', out)]
    if not times:
        raise RuntimeError('could not find render times in Blender output:\n' + out[-2000:])
    res = dict(seconds=times, total=sum(times), wall=wall, device='GPU' if gpu else 'file',
               frames=sorted(f for f in os.listdir(out_dir) if f.endswith('.png')))
    log('  blender %s: %d frames, %.1f s of rendering (%.1f s wall)' % ('gpu' if gpu else 'as-is', len(times), res['total'], wall))
    with open(stamp, 'w') as f:
        json.dump(res, f)
    return res
