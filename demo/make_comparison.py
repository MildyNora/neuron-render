"""Races several scenes at once, then sums them up on one card.

    python demo/make_comparison.py scenes/mori.blend scenes/stage.blend scenes/cornell.blend scenes/glassware.blend

Every cell is one scene: Blender on the left (what Cycles has after each sample, in real time),
Neuron Render on the right. All races start together. Numbers are the same measurements the
per-scene videos use (see make_video.py).
"""
import argparse
import math
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_video import BG, COOL, DIM, FAINT, FPS, H, INK, W, WARM, draw_text, ease, encode, fit_box, font, gather, mix, seconds_label  # noqa: E402


class Cell:
    """One scene inside the grid: two pictures, two timers, a speed-up once Blender is done."""

    def __init__(self, m, rect):
        self.m, self.rect = m, rect
        x0, y0, cw, ch = rect
        pad, gap = 28, 28
        self.bw, self.bh = (cw - 2 * pad - gap) // 2, ch - 132
        self.size = fit_box(m['final'].size, self.bw, self.bh)
        self.x = [x0 + pad + (self.bw - self.size[0]) // 2, x0 + pad + self.bw + gap + (self.bw - self.size[0]) // 2]
        self.cx = [x0 + pad + self.bw // 2, x0 + pad + self.bw + gap + self.bw // 2]
        self.y = y0 + 22 + (self.bh - self.size[1])          # pictures sit on their progress bars
        self.text_y = y0 + 22 + self.bh + 14
        self.frames = [(t, im.resize(self.size, Image.LANCZOS)) for t, im in m['frames']]
        self.final, self.neuron = m['final'].resize(self.size, Image.LANCZOS), m['neuron'].resize(self.size, Image.LANCZOS)
        self.f_label, self.f_time, self.f_big = font(14, 'Medium'), font(34, 'Semibold'), font(46, 'Bold')

    def draw(self, im, d, t):
        m, (pw, ph), y = self.m, self.size, self.y
        for x in self.x:
            d.rectangle([x - 1, y - 1, x + pw, y + ph], outline=(26, 26, 30))
            d.rectangle([x, y, x + pw - 1, y + ph - 1], fill=(3, 3, 4))
        if t >= m['T']:
            im.paste(self.final, (self.x[0], y))
        else:
            shown = [f for s, f in self.frames if t >= s]
            if shown:
                im.paste(shown[-1], (self.x[0], y))
        if t >= m['Tn']:
            im.paste(self.neuron, (self.x[1], y))
        for x, total, col in ((self.x[0], m['T'], WARM), (self.x[1], m['Tn'], COOL)):
            d.rectangle([x, y + ph + 6, x + pw - 1, y + ph + 7], fill=(24, 24, 28))
            frac = min(max(t / total, 0.0), 1.0)
            if frac > 0:
                d.rectangle([x, y + ph + 6, x + int((pw - 1) * frac), y + ph + 7], fill=col)
        ty = self.text_y
        draw_text(d, self.cx[0], ty, 'BLENDER', self.f_label, DIM, tracking=2.4)
        draw_text(d, self.cx[1], ty, 'NEURON', self.f_label, DIM, tracking=2.4)
        draw_text(d, self.cx[0], ty + 22, '%.2f s' % min(max(t, 0.0), m['T']), self.f_time, INK if t >= m['T'] else mix(INK, WARM, 0.35), tabular=True)
        draw_text(d, self.cx[1], ty + 22, '%.3f s' % min(max(t, 0.0), m['Tn']), self.f_time, INK, tabular=True)
        a = ease((t - m['T'] - 0.3) / 0.6)
        if a > 0:
            draw_text(d, 0.5 * (self.cx[0] + self.cx[1]), ty + 12, '%d×' % round(m['T'] / m['Tn']), self.f_big, mix(BG, COOL, a))


def grid(n):
    cols = 1 if n == 1 else 2
    rows = math.ceil(n / cols)
    cw, ch = W // cols, H // rows
    return [(c * cw, r * ch, cw, ch) for r in range(rows) for c in range(cols)][:n]


def summary(ms):
    """One row per scene: the frame, both times on a shared log axis, the speed-up, and how close each
    renderer is to a converged Cycles render (PSNR on held-out views)."""
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    f_name, f_num, f_big, f_small = font(20, 'Medium'), font(30, 'Semibold'), font(64, 'Bold'), font(19, 'Regular')
    top, rowh = 60, (H - 120) // len(ms)
    lo, hi = math.log10(1e-3), math.log10(max(m['T'] for m in ms) * 1.15)
    bx0, bx1 = 420, 1330
    px = lambda sec: bx0 + (bx1 - bx0) * (math.log10(max(sec, 1e-3)) - lo) / (hi - lo)
    for i, m in enumerate(ms):
        y = top + i * rowh
        thumb = m['neuron'].resize(fit_box(m['neuron'].size, 250, rowh - 50), Image.LANCZOS)
        im.paste(thumb, (60 + (250 - thumb.size[0]) // 2, y + (rowh - 30 - thumb.size[1]) // 2))
        mid = y + (rowh - 30) // 2
        draw_text(d, bx0, mid - 62, m['name'].upper(), f_name, DIM, tracking=3.0, anchor='l')
        for k, (sec, col) in enumerate(((m['T'], WARM), (m['Tn'], COOL))):
            by = mid - 22 + k * 40
            d.rectangle([bx0, by, bx1, by + 3], fill=(22, 22, 26))
            d.rectangle([bx0, by, int(px(sec)), by + 3], fill=col)
            draw_text(d, px(sec) + 14, by - 15, seconds_label(sec), f_num, INK, tabular=True, anchor='l')
        draw_text(d, 1640, mid - 52, '%d×' % round(m['T'] / m['Tn']), f_big, COOL)
        acc, q = m['accuracy'], m['quality']
        if q in acc and 'blender' in acc:
            draw_text(d, 1640, mid + 28, 'accuracy %.1f dB  ·  Blender %.1f dB' % (acc[q]['psnr'], acc['blender']['psnr']), f_small, FAINT)
    # axis: decades
    ay = H - 44
    for e in range(-3, int(math.floor(hi)) + 1):
        x = px(10.0 ** e)
        d.rectangle([x, ay - 6, x, ay], fill=(60, 60, 68))
        draw_text(d, x, ay + 4, seconds_label(10.0 ** e) if e < 0 else '%d s' % 10 ** e, f_small, FAINT)
    return im


def timeline(cells, card):
    lead, hold, fade, stay = 1.2, 2.4, 0.6, 9.0
    race_end = lead + max(c.m['T'] for c in cells) + hold
    last = None
    for i in range(int((race_end + stay) * FPS)):
        t = i / FPS
        if t < race_end:
            im = Image.new('RGB', (W, H), BG)
            d = ImageDraw.Draw(im)
            for c in cells:
                c.draw(im, d, t - lead)
            last = im
        else:
            im = card if t - race_end >= fade else Image.blend(last, card, ease((t - race_end) / fade))
        yield im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('blends', nargs='+')
    ap.add_argument('-o', '--output', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenes_compare.mp4'))
    ap.add_argument('-q', '--quality', default='balanced')
    ap.add_argument('--stills', help='also write a few frames here as PNG, to check the layout')
    args = ap.parse_args()
    ms = [gather(b, args.quality) for b in args.blends]
    cells = [Cell(m, rect) for m, rect in zip(ms, grid(len(ms)))]
    card = summary(ms)
    for m in ms:
        print('%-10s blender %6.2f s | neuron %5.1f ms | %4.0fx' % (m['name'], m['T'], m['Tn'] * 1e3, m['T'] / m['Tn']))
    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        tmax = max(m['T'] for m in ms)
        for name, t in (('grid_early', 0.4 * min(m['T'] for m in ms)), ('grid_done', tmax + 1.5)):
            im = Image.new('RGB', (W, H), BG)
            d = ImageDraw.Draw(im)
            for c in cells:
                c.draw(im, d, t)
            im.save(os.path.join(args.stills, name + '.png'))
        card.save(os.path.join(args.stills, 'summary.png'))
    encode(timeline(cells, card), args.output)


if __name__ == '__main__':
    main()
