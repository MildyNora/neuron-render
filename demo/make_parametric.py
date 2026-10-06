"""Builds the video for a parametric fit: one network covering animation time, lights and materials.

    python demo/make_parametric.py scenes/stage.blend -o demo/stage_parametric.mp4

  1. the file's animation, Blender's frames beside the network's, with the render time each has spent so far
  2. the lights changed while it plays (strength and colour of every lamp, strength of the world)
  3. the materials changed while it plays (base colour and roughness)
  4. states the network never saw -- withheld frames with edited lights and materials -- beside a converged
     Cycles render of exactly that state
  5. the numbers

Blender's side is measured by rendering the animation for real (`blender -b file -a`); the network's frames
are rendered while the video is built, and timed.
"""
import argparse
import colorsys
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from make_video import BG, COOL, DIM, FAINT, FPS, H, INK, W, WARM, draw_text, ease, encode, fit_box, font, mix  # noqa: E402
from neuron_render import bench, plan as P  # noqa: E402
from neuron_render.fit import test_state  # noqa: E402
from neuron_render.metrics import psnr  # noqa: E402
from neuron_render.render import Renderer  # noqa: E402


def clock(sec):
    return '%d:%04.1f' % (sec // 60, sec % 60) if sec >= 60 else '%.2f s' % sec


class Show:
    def __init__(self, blend, quality):
        self.blend = os.path.abspath(blend)
        self.bundle = os.path.splitext(self.blend)[0] + '.parametric.neuron'
        self.r = Renderer.open(self.bundle)
        self.q = quality
        self.scene = self.r.scene
        self.plan = P.load_plan(self.bundle)
        self.fit = json.load(open(os.path.join(self.bundle, 'fit.json')))
        self.frames = self.scene.frames
        self.fps = self.scene.meta.get('fps', 24)
        print('rendering the animation in Blender (cached after the first time)')
        self.cpu = bench.blender_animation(self.blend, os.path.join(self.bundle, 'blender_anim'))
        self.gpu = bench.blender_animation(self.blend, os.path.join(self.bundle, 'blender_anim_gpu'), gpu=True)
        self.size = fit_box((self.scene.camera.width, self.scene.camera.height), 880, 900)
        self.big = fit_box((self.scene.camera.width, self.scene.camera.height), W - 200, H - 220, upscale=1.6)
        self.ref = [Image.open(os.path.join(self.bundle, 'blender_anim', f)).convert('RGB').resize(self.size, Image.LANCZOS) for f in self.cpu['frames']]
        # The network's own pass over the animation, timed: every frame's tables are built from scratch, as on a
        # first viewing. The machine is shared, so the pass is made three times and the median one is reported.
        self.r.warmup(quality, frame=self.frames[0])
        passes = []
        for _ in range(3):
            self.r.reset()
            shots = [self.r.render(quality=quality, frame=f) for f in self.frames]
            passes.append(([img for img, _ in shots], [info['total_ms'] for _, info in shots]))
        passes.sort(key=lambda p: sum(p[1]))
        print('  neuron: three passes over the animation took %s s' % ', '.join('%.2f' % (sum(p[1]) / 1e3) for p in passes))
        self.neuron = [Image.fromarray(img).resize(self.size, Image.LANCZOS) for img in passes[1][0]]
        self.ms = passes[1][1]
        self.t_blender, self.t_neuron = float(np.sum(self.cpu['seconds'])), float(np.sum(self.ms)) / 1e3
        self.cum_b, self.cum_n = np.cumsum(self.cpu['seconds']), np.cumsum(self.ms) / 1e3
        self.f_label, self.f_time, self.f_note, self.f_big, self.f_mid, self.f_small = (
            font(21, 'Medium'), font(58, 'Semibold'), font(18, 'Regular'), font(96, 'Bold'), font(40, 'Semibold'), font(16, 'Medium'))
        self.groups = self.scene.groups
        self.editable = self.r.field.cfg['varied']

    # -- part 1: the animation ---------------------------------------------------------------------
    def playback(self, t, counting):
        k = min(int(t * self.fps), len(self.frames) - 1)
        pw, ph = self.size
        lx, rx, py = 480 - pw // 2, 1440 - pw // 2, 36 + (900 - ph) // 2
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        im.paste(self.ref[k], (lx, py))
        im.paste(self.neuron[k], (rx, py))
        frac = (k + 1) / len(self.frames)
        for x, col in ((lx, WARM), (rx, COOL)):
            d.rectangle([x, py + ph + 8, x + pw - 1, py + ph + 10], fill=(24, 24, 28))
            d.rectangle([x, py + ph + 8, x + int((pw - 1) * frac), py + ph + 10], fill=col)
        tb, tn = (self.cum_b[k], self.cum_n[k]) if counting else (self.t_blender, self.t_neuron)
        draw_text(d, 480, 962, 'BLENDER CYCLES', self.f_label, DIM, tracking=3.2)
        draw_text(d, 1440, 962, 'NEURON RENDER', self.f_label, DIM, tracking=3.2)
        draw_text(d, 480, 992, clock(tb), self.f_time, INK, tabular=True)
        draw_text(d, 1440, 992, clock(tn), self.f_time, INK, tabular=True)
        return im

    def verdict(self, im, a):
        if a <= 0:
            return im
        d = ImageDraw.Draw(im)
        draw_text(d, W / 2, 966, '%d×' % round(self.t_blender / self.t_neuron), self.f_big, mix(BG, COOL, a))
        note = mix(BG, FAINT, a)
        draw_text(d, 60, 1046, '%d frames  ·  Cycles on the GPU: %s' % (len(self.frames), clock(self.gpu['total'])), self.f_note, note, anchor='l')
        draw_text(d, W - 60, 1046, 'after a one-time %.0f min fit' % (self.fit['one_time_seconds'] / 60.0), self.f_note, note, anchor='r')
        return im

    # -- parts 2 and 3: one live picture with the controls under it ----------------------------------
    def live(self, t, lights=None, materials=None, controls=()):
        """The animation keeps playing while the given state is applied. controls: [(label, value 0..1, rgb)]."""
        k = int(t * self.fps) % len(self.frames)
        img, _ = self.r.render(quality=self.q, frame=self.frames[k], lights=lights, materials=materials)
        im = Image.new('RGB', (W, H), BG)
        pic = Image.fromarray(img).resize(self.big, Image.LANCZOS)
        x0, y0 = (W - self.big[0]) // 2, 40
        im.paste(pic, (x0, y0))
        d = ImageDraw.Draw(im)
        n = len(controls)
        cw, gap = 250, 60
        x = (W - (n * cw + (n - 1) * gap)) // 2
        y = y0 + self.big[1] + 46
        for label, value, rgb in controls:
            draw_text(d, x, y - 8, label, self.f_small, DIM, tracking=2.6, anchor='l')
            d.rectangle([x, y + 22, x + cw, y + 26], fill=(26, 26, 30))
            col = tuple(int(255 * max(0.0, min(1.0, c)) ** (1 / 2.2)) for c in rgb)
            d.rectangle([x, y + 22, x + int(cw * max(0.0, min(1.0, value))), y + 26], fill=col)
            d.ellipse([x + cw + 14, y + 12, x + cw + 38, y + 36], fill=col)
            x += cw + gap
        return im

    def relight(self, t, T):
        """Each lamp in turn: strength down and up, colour round the wheel; then the world."""
        u = t / T
        lights, controls = {}, []
        lamps = [g for g in self.groups if g['kind'] == 'light']
        seg = 1.0 / (len(lamps) + 1)
        for i, g in enumerate(lamps):
            v = min(max((u - i * seg) / seg, 0.0), 1.0)
            active = 0.0 < v < 1.0
            k = 1.0 + (0.9 * math.sin(2 * math.pi * v) if active else 0.0)                      # 0.1x .. 1.9x
            hue_shift = v if active else 0.0
            h, s_, _ = colorsys.rgb_to_hsv(*g['color'])
            col = list(colorsys.hsv_to_rgb((h + hue_shift) % 1.0, max(s_, 0.55 * math.sin(math.pi * v)) if active else s_, 1.0))
            lights[g['name']] = dict(energy=g['energy'] * k, color=col)
            controls.append((g['name'].upper(), k / 2.0, col))
        world = next((g for g in self.groups if g['kind'] == 'world'), None)
        if world is not None:
            v = min(max((u - len(lamps) * seg) / seg, 0.0), 1.0)
            k = 1.0 + (0.8 * math.sin(2 * math.pi * v) if 0.0 < v < 1.0 else 0.0)
            lights[world['name']] = dict(strength=world['strength'] * k)
            controls.append((world['name'].upper(), k / 2.0, (0.8, 0.85, 1.0)))
        return self.live(t, lights=lights, controls=controls)

    def rematerial(self, t, T):
        """Base colour round the wheel and roughness up and down, on up to three materials."""
        u = t / T
        mats = [m for m in self.scene.materials if m['name'] in self.editable and m['resolved'] and not m['color_attribute']]
        mats.sort(key=lambda m: -(m['metallic'] + m['transmission'] + (1.0 - m['roughness'])))   # the showy ones first
        mats = mats[:3]
        materials, controls = {}, []
        for i, m in enumerate(mats):
            h, s_, v_ = colorsys.rgb_to_hsv(*m['base_color'][:3])
            hue = (h + u * (1.0 + 0.5 * i)) % 1.0
            col = list(colorsys.hsv_to_rgb(hue, max(s_, 0.6), max(v_, 0.5)))
            rough = min(max(m['roughness'] + 0.45 * math.sin(2 * math.pi * (u * 1.5 + i / 3.0)), 0.03), 1.0)
            materials[m['name']] = dict(base_color=col, roughness=rough)
            controls.append((m['name'].upper(), rough, col))
        return self.live(t, materials=materials, controls=controls)

    # -- part 4: unseen states against Cycles -------------------------------------------------------
    def checks(self):
        out = []
        for v in self.plan['test']:
            if not v['name'].startswith('edit'):
                continue
            truth = self.r.display(np.load(os.path.join(self.bundle, 'truth', v['name'] + '.npy')).astype(np.float32))
            img, _ = self.r.render(quality='high', output='display', **test_state(self.r, self.plan, v))
            to8 = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).resize(self.size, Image.LANCZOS)
            out.append((to8(truth), to8(img), psnr(img, truth)))
        return out

    def check(self, item):
        ref, mine, db = item
        pw, ph = self.size
        lx, rx, py = 480 - pw // 2, 1440 - pw // 2, 36 + (900 - ph) // 2
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        im.paste(ref, (lx, py))
        im.paste(mine, (rx, py))
        draw_text(d, 480, 962, 'BLENDER CYCLES  ·  1024 SAMPLES', self.f_label, DIM, tracking=3.2)
        draw_text(d, 1440, 962, 'NEURON RENDER  ·  NEVER SEEN', self.f_label, DIM, tracking=3.2)
        draw_text(d, W / 2, 992, '%.1f dB' % db, self.f_time, INK)
        return im

    # -- part 5 -----------------------------------------------------------------------------------
    def summary(self):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        acc = self.fit['accuracy']['views']
        mean = lambda names, key: float(np.mean([acc[n][key]['psnr'] for n in names if n in acc and key in acc[n]]))
        frames = [n for n in acc if n.startswith('frame')]
        edits = [n for n in acc if n.startswith('edit')]
        rows = [('%d FRAMES' % len(self.frames), clock(self.t_blender), clock(self.t_neuron), '%d×' % round(self.t_blender / self.t_neuron))]
        if frames:
            rows.append(('WITHHELD FRAMES', '%.1f dB' % mean(frames, 'blender'), '%.1f dB' % mean(frames, self.q), ''))
        if edits:
            rows.append(('WITHHELD FRAMES, EDITED LIGHTS + MATERIALS', '%.1f dB' % mean(edits, 'blender'), '%.1f dB' % mean(edits, self.q), ''))
        y = 300
        draw_text(d, 1010, y - 90, 'BLENDER', self.f_label, WARM, tracking=3.2)
        draw_text(d, 1330, y - 90, 'NEURON', self.f_label, COOL, tracking=3.2)
        for label, a, b, c in rows:
            draw_text(d, 180, y + 16, label, self.f_label, DIM, tracking=3.0, anchor='l')
            draw_text(d, 1010, y, a, self.f_mid, INK, tabular=True)
            draw_text(d, 1330, y, b, self.f_mid, INK, tabular=True)
            if c:
                draw_text(d, 1660, y - 6, c, font(56, 'Bold'), COOL)
            y += 130
        draw_text(d, 180, y + 40, 'accuracy: PSNR against a converged Cycles render of the same state', self.f_note, FAINT, anchor='l')
        draw_text(d, 180, y + 70, 'one fit, %.0f min: time + %d lights + %d materials' % (self.fit['one_time_seconds'] / 60.0, len(self.groups), len(self.editable)),
                  self.f_note, FAINT, anchor='l')
        return im


def timeline(s):
    dur = len(s.frames) / s.fps
    lead, relight, remat, each, fade, last = 0.8, 12.0, 10.0, 3.2, 0.5, 7.0
    checks = s.checks()
    card = s.summary()
    prev = None
    segments = [('play1', dur), ('play2', dur + 2.5), ('relight', relight), ('remat', remat)] + [('check%d' % i, each) for i in range(len(checks))] + [('card', last)]
    for i in range(int(lead * FPS)):
        yield s.playback(0.0, True)
    for name, length in segments:
        first = None
        for i in range(int(length * FPS)):
            t = i / FPS
            if name == 'play1':
                im = s.playback(t, True)
            elif name == 'play2':
                im = s.verdict(s.playback(t % dur, False), ease(t / 0.6))
            elif name == 'relight':
                im = s.relight(t, length)
            elif name == 'remat':
                im = s.rematerial(t, length)
            elif name == 'card':
                im = card
            else:
                im = s.check(checks[int(name[5:])])
            if name not in ('play1', 'play2') and t < fade and prev is not None:
                im = Image.blend(prev, im, ease(t / fade))
            last_im = im
            yield im
        prev = last_im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('blend')
    ap.add_argument('-o', '--output')
    ap.add_argument('-q', '--quality', default='balanced')
    ap.add_argument('--stills', help='also write a few frames here as PNG, to check the layout')
    args = ap.parse_args()
    s = Show(args.blend, args.quality)
    name = os.path.splitext(os.path.basename(args.blend))[0]
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), '%s_parametric.mp4' % name)
    print('%s: %d frames | blender %s (gpu %s) | neuron %s | %dx' % (name, len(s.frames), clock(s.t_blender), clock(s.gpu['total']), clock(s.t_neuron),
                                                                    round(s.t_blender / s.t_neuron)))
    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        for tag, im in (('p_play', s.verdict(s.playback(2.0, False), 1.0)), ('p_relight', s.relight(3.0, 12.0)), ('p_remat', s.rematerial(4.0, 10.0)),
                        ('p_check', s.check(s.checks()[0])), ('p_card', s.summary())):
            im.save(os.path.join(args.stills, tag + '.png'))
    encode(timeline(s), out)


if __name__ == '__main__':
    main()
