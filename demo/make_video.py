"""Builds the comparison video for one scene: Blender rendering a frame vs Neuron Render rendering it.

    python demo/make_video.py scenes/mori.blend -o demo/mori_vs_blender.mp4

Everything on screen is measured, nothing is staged:
  * the Blender timer runs for Blender's own reported render time of `blender -b file -f 1` (its best
    run), and the picture shows what Cycles actually has after 1, 2, 4 ... samples, when it has them;
  * the Neuron timer runs for the engine's median steady-state frame time, and the picture is that frame.
"""
import argparse
import json
import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from neuron_render import blender, cli, plan as P  # noqa: E402
from neuron_render.render import Renderer  # noqa: E402

W, H, FPS = 1920, 1080, 60
BG, INK, DIM, FAINT = (8, 8, 10), (244, 244, 246), (150, 150, 158), (118, 118, 128)
COOL, WARM = (120, 170, 255), (255, 150, 110)
DIAL = ('draft', 'fast', 'balanced', 'high')        # the speed/quality dial, shown on the closing card


def font(size, weight='Semibold'):
    try:
        f = ImageFont.truetype('/System/Library/Fonts/SFNS.ttf', size)
        try:
            f.set_variation_by_name(weight)
        except (OSError, ValueError):
            pass
        return f
    except OSError:
        return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size)


def text_width(draw, s, f, tracking=0.0, tabular=False):
    cell = max(draw.textlength(d, font=f) for d in '0123456789') if tabular else 0
    return sum((cell if (tabular and c.isdigit()) else draw.textlength(c, font=f)) + tracking for c in s) - tracking


def draw_text(draw, cx, y, s, f, fill, tracking=0.0, tabular=False, anchor='c'):
    """Text with letter-spacing and (optionally) fixed-width digits, so a running timer does not jitter."""
    cell = max(draw.textlength(d, font=f) for d in '0123456789') if tabular else 0
    w = text_width(draw, s, f, tracking, tabular)
    x = cx - w / 2 if anchor == 'c' else (cx if anchor == 'l' else cx - w)
    for c in s:
        adv = cell if (tabular and c.isdigit()) else draw.textlength(c, font=f)
        draw.text((x + (adv - draw.textlength(c, font=f)) / 2, y), c, font=f, fill=fill)
        x += adv + tracking


def mix(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fit_box(size, box_w, box_h, upscale=1.0):
    """Largest (w, h) with the aspect of size that fits the box."""
    s = min(box_w / size[0], box_h / size[1], upscale)
    return max(int(size[0] * s), 1), max(int(size[1] * s), 1)


def seconds_label(sec):
    return '%.2f s' % sec if sec >= 1 else '%.0f ms' % (sec * 1e3)


def encode(frames, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS),
                            '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            path], stdin=subprocess.PIPE)
    n = 0
    for im in frames:
        enc.stdin.write(im.tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    print('%s  %d frames, %.1f s' % (path, n, n / FPS))


# ---------------------------------------------------------------------------------------------
# measurements

def progressive_steps(samples):
    """Sample counts at which to show Cycles' work in progress, ending at the file's sample count."""
    steps = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if k < samples]
    return steps + [samples]


def gather(blend, quality='balanced', log=print):
    """Everything one scene contributes to a video: timings, accuracy and the pictures."""
    blend = os.path.abspath(blend)
    bundle = os.path.splitext(blend)[0] + '.neuron'
    if not os.path.exists(os.path.join(bundle, 'bench.json')):
        cli.main(['bench', blend, '--gpu'])
    b = json.load(open(os.path.join(bundle, 'bench.json')))
    r = Renderer.open(bundle)
    r.warmup(quality, 5)
    neuron = Image.fromarray(r.render(quality=quality)[0])
    final = Image.open(os.path.join(bundle, 'blender_frame.png')).convert('RGB')

    # What Cycles has after k samples, and when: the file's own settings, cut short, denoiser off.
    plan = P.load_plan(bundle)
    hero = dict(plan)
    hero['test'] = plan['test'][:1]
    cam = r.scene.camera
    steps = progressive_steps(int(r.scene.meta['render']['samples']))
    stamps = []
    for k in steps + ['full']:
        sub = os.path.join('progressive', 's%s' % k)
        if not os.path.exists(os.path.join(bundle, sub, 'timing.json')):
            extra = dict(save_png=True) if k == 'full' else dict(save_png=True, samples=k, denoise=False)
            blender.run_job(blend, P.job(bundle, hero, 'test', sub, 'scene', cam.width, cam.height, **extra))
            log('  cycles after %s samples: %.2f s' % (k, json.load(open(os.path.join(bundle, sub, 'timing.json')))['seconds'][0]))
        stamps.append(json.load(open(os.path.join(bundle, sub, 'timing.json')))['seconds'][0])
    full = stamps.pop()
    T = b['blender']['best']
    frames = [(T * min(s / full, 0.985), Image.open(os.path.join(bundle, 'progressive', 's%d' % k, 'hero.png')).convert('RGB'))
              for k, s in zip(steps, stamps)]
    fit = json.load(open(os.path.join(bundle, 'fit.json')))
    acc = fit.get('accuracy', {})
    dial = [(p, Image.fromarray(r.render(quality=p)[0]), b['neuron'][p]['median_ms'] / 1e3) for p in DIAL]
    return dict(name=os.path.splitext(os.path.basename(blend))[0], quality=quality, T=T, Tn=b['neuron'][quality]['median_ms'] / 1e3,
                gpu=b.get('blender_gpu', {}).get('best'), frames=frames, final=final, neuron=neuron, one_time=fit['one_time_seconds'],
                dial=dial, accuracy=acc.get('presets', {}), hero_accuracy=acc.get('views', {}).get('hero', {}),
                samples=int(r.scene.meta['render']['samples']), triangles=r.scene.n_tris)


# ---------------------------------------------------------------------------------------------
# drawing

class Scenes:
    def __init__(self, m):
        self.m = m
        # Two panels, one per half of the frame, as large as the picture's shape allows.
        self.pw, self.ph = fit_box(m['final'].size, 860, 900)
        self.lx, self.rx = 480 - self.pw // 2, 1440 - self.pw // 2
        self.py = 36 + (900 - self.ph) // 2
        size = (self.pw, self.ph)
        self.frames = [(t, im.resize(size, Image.LANCZOS)) for t, im in m['frames']]
        self.final, self.neuron = m['final'].resize(size, Image.LANCZOS), m['neuron'].resize(size, Image.LANCZOS)
        self.f_label, self.f_time, self.f_note, self.f_big = font(21, 'Medium'), font(58, 'Semibold'), font(18, 'Regular'), font(96, 'Bold')
        self.f_mid = font(44, 'Semibold')
        self.speed = m['T'] / m['Tn']

    def race(self, t):
        """t = seconds since both renderers were told to start (negative before the start)."""
        m, pw, ph, py = self.m, self.pw, self.ph, self.py
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        for x in (self.lx, self.rx):
            d.rectangle([x - 1, py - 1, x + pw, py + ph], outline=(26, 26, 30))
            d.rectangle([x, py, x + pw - 1, py + ph - 1], fill=(3, 3, 4))
        if t >= m['T']:
            im.paste(self.final, (self.lx, py))
        else:
            shown = [f for s, f in self.frames if t >= s]
            if shown:
                im.paste(shown[-1], (self.lx, py))
        if t >= m['Tn']:
            im.paste(self.neuron, (self.rx, py))
        for x, total, col in ((self.lx, m['T'], WARM), (self.rx, m['Tn'], COOL)):
            d.rectangle([x, py + ph + 8, x + pw - 1, py + ph + 10], fill=(24, 24, 28))
            frac = min(max(t / total, 0.0), 1.0)
            if frac > 0:
                d.rectangle([x, py + ph + 8, x + int((pw - 1) * frac), py + ph + 10], fill=col)
        tb, tn = min(max(t, 0.0), m['T']), min(max(t, 0.0), m['Tn'])
        draw_text(d, 480, 962, 'BLENDER CYCLES', self.f_label, DIM, tracking=3.2)
        draw_text(d, 1440, 962, 'NEURON RENDER', self.f_label, DIM, tracking=3.2)
        draw_text(d, 480, 992, '%.2f s' % tb, self.f_time, INK if t >= m['T'] else mix(INK, WARM, 0.35), tabular=True)
        draw_text(d, 1440, 992, '%.3f s' % tn, self.f_time, INK, tabular=True)
        return im

    def verdict(self, im, a):
        """Fade the speed-up in between the two timers."""
        if a <= 0:
            return im
        d = ImageDraw.Draw(im)
        m = self.m
        draw_text(d, W / 2, 966, '%d×' % round(self.speed), self.f_big, mix(BG, COOL, a))
        note = mix(BG, FAINT, a)
        if m['gpu']:
            draw_text(d, 60, 1046, 'Cycles on the GPU: %.2f s' % m['gpu'], self.f_note, note, anchor='l')
        draw_text(d, W - 60, 1046, 'after a one-time %.0f min fit' % (m['one_time'] / 60.0), self.f_note, note, anchor='r')
        return im

    def wipe(self, u):
        """Same frame with a divider sweeping across: Blender left of it, Neuron right."""
        m = self.m
        im = Image.new('RGB', (W, H), BG)
        size = fit_box(m['final'].size, W - 520, H - 80, upscale=1.5)
        a = m['final'] if size == m['final'].size else m['final'].resize(size, Image.LANCZOS)
        b = m['neuron'] if size == m['neuron'].size else m['neuron'].resize(size, Image.LANCZOS)
        x0, y0 = (W - size[0]) // 2, (H - size[1]) // 2
        cut = int(round(size[0] * (0.5 + 0.42 * math.sin(u * 2 * math.pi))))
        im.paste(b, (x0, y0))
        im.paste(a.crop((0, 0, cut, size[1])), (x0, y0))
        d = ImageDraw.Draw(im)
        d.rectangle([x0 + cut - 1, y0, x0 + cut, y0 + size[1] - 1], fill=(250, 250, 252))
        draw_text(d, x0 - 36, H / 2 - 12, 'BLENDER', self.f_label, DIM, tracking=3.2, anchor='r')
        draw_text(d, x0 + size[0] + 36, H / 2 - 12, 'NEURON', self.f_label, DIM, tracking=3.2, anchor='l')
        return im

    def dial(self):
        """Closing card: the same detail from Blender and from each setting of the dial, with its frame time."""
        m = self.m
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        tiles = [('blender', m['final'], m['T'])] + list(m['dial'])
        side, gap = 340, 30
        x = (W - (len(tiles) * side + (len(tiles) - 1) * gap)) // 2
        iw, ih = m['final'].size
        crop = min(side, iw, ih)
        cx, cy = int(iw * 0.50), min(max(int(ih * 0.53), crop // 2), ih - crop // 2)
        box = (cx - crop // 2, cy - crop // 2, cx - crop // 2 + crop, cy - crop // 2 + crop)
        y = 300
        for i, (name, img, sec) in enumerate(tiles):
            tile = img.crop(box)
            im.paste(tile if crop == side else tile.resize((side, side), Image.LANCZOS), (x, y))
            d.rectangle([x, y + side + 10, x + side - 1, y + side + 12], fill=(24, 24, 28))
            frac = max(math.log10(sec * 1e3) / math.log10(m['T'] * 1e3), 0.02)   # log scale: 1 ms .. Blender
            d.rectangle([x, y + side + 10, x + int((side - 1) * frac), y + side + 12], fill=WARM if i == 0 else COOL)
            draw_text(d, x + side / 2, y + side + 34, name.upper(), self.f_label, DIM, tracking=3.2)
            draw_text(d, x + side / 2, y + side + 64, seconds_label(sec), self.f_mid, INK, tabular=True)
            x += side + gap
        return im


def timeline(sc):
    """Yields every video frame."""
    m = sc.m
    lead, hold, verdict, fade, wipe, dial = 1.2, 0.9, 3.2, 0.5, 8.0, 5.0
    race_end = lead + m['T'] + hold
    wipe_at = race_end + verdict
    dial_at = wipe_at + wipe
    card = sc.dial()
    last = None
    for i in range(int((dial_at + dial) * FPS)):
        t = i / FPS
        if t < wipe_at:
            im = sc.verdict(sc.race(t - lead), ease((t - race_end) / 0.6))
            last = im
        elif t < dial_at:
            im = sc.wipe((t - wipe_at) / wipe)
            if t - wipe_at < fade:
                im = Image.blend(last, im, ease((t - wipe_at) / fade))
            else:
                last = im
        else:
            im = card if t - dial_at >= fade else Image.blend(last, card, ease((t - dial_at) / fade))
        yield im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('blend')
    ap.add_argument('-o', '--output')
    ap.add_argument('-q', '--quality', default='balanced')
    ap.add_argument('--stills', help='also write a few frames here as PNG, to check the layout')
    args = ap.parse_args()
    m = gather(args.blend, args.quality)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), '%s_vs_blender.mp4' % m['name'])
    sc = Scenes(m)
    print('%s: blender %.2f s | neuron %.1f ms | %.0fx | one-time fit %.0f s' % (m['name'], m['T'], m['Tn'] * 1e3, sc.speed, m['one_time']))
    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        for name, im in (('race_mid', sc.race(m['T'] * 0.6)), ('wipe', sc.wipe(0.1)), ('verdict', sc.verdict(sc.race(m['T'] + 1), 1.0)), ('dial', sc.dial())):
            im.save(os.path.join(args.stills, '%s_%s.png' % (m['name'], name)))
    encode(timeline(sc), out)


if __name__ == '__main__':
    main()
