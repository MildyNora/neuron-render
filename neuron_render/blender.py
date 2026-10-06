"""Locating Blender and running the scripts in neuron_render/bl inside it."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile

_BL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bl')


def find_blender():
    cands = [os.environ.get('NEURON_RENDER_BLENDER'), os.environ.get('BLENDER'), shutil.which('blender'),
             '/Applications/Blender.app/Contents/MacOS/Blender']
    cands += sorted(glob.glob('/Applications/Blender*.app/Contents/MacOS/Blender'), reverse=True)
    if shutil.which('mdfind'):
        try:
            q = "kMDItemCFBundleIdentifier == 'org.blenderfoundation.blender'"
            for line in subprocess.run(['mdfind', q], capture_output=True, text=True, timeout=10).stdout.splitlines():
                cands.append(os.path.join(line, 'Contents/MacOS/Blender'))
        except (subprocess.TimeoutExpired, OSError):
            pass
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError('Blender not found: put it on PATH or set NEURON_RENDER_BLENDER')


def run_script(blend, script, args, on_progress=None, blender=None):
    """Run bl/<script> headless against blend. Lines starting with NR_ are the script's protocol."""
    cmd = [blender or find_blender(), '-b', blend, '--python-exit-code', '1', '--python', os.path.join(_BL, script), '--'] + list(args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    tail, done, notes = [], False, []
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-40]
        if line.startswith('NR_PROGRESS') and on_progress:
            _, i, n, dt = line.split()
            on_progress(int(i), int(n), float(dt))
        elif line.startswith('NR_DONE'):
            done = True
        elif line.startswith('NR_'):
            notes.append(line)
    if proc.wait() != 0 or not done:
        raise RuntimeError('Blender script %s failed:\n%s' % (script, '\n'.join(tail)))
    return notes


def run_job(blend, job, on_progress=None, blender=None):
    """Run a teacher.py job (see that file for the modes)."""
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(job, f)
    try:
        return run_script(blend, 'teacher.py', ['--job', f.name], on_progress, blender)
    finally:
        os.remove(f.name)
