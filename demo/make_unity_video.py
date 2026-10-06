"""Captions and encodes the frames recorded by the Unity runtime (NeuronDemoCapture) into demo/unity_stage.mp4.

    python demo/make_unity_video.py <capture dir> [-o demo/unity_stage.mp4]
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_video import BG, COOL, DIM, FPS, H, INK, W, draw_text, encode, fit_box, font  # noqa: E402

CAPTIONS = {'animation': 'THE ANIMATION, THROUGH ITS OWN CAMERA', 'camera': 'A CAMERA MOVE OF OUR OWN',
            'lights': 'LIGHTS CHANGED LIVE', 'materials': 'MATERIALS CHANGED LIVE'}


def frames(cap, out):
    rows = [line.split() for line in open(os.path.join(cap, 'frames.txt')) if line.strip()]
    f_label, f_small = font(21, 'Medium'), font(18, 'Regular')
    size = None
    for i, (idx, part, frame, _ms) in enumerate(rows):
        im = Image.open(os.path.join(cap, 'f_%04d.png' % int(idx))).convert('RGB')
        size = size or fit_box(im.size, W - 160, H - 200, upscale=1.6)
        pic = im.resize(size, Image.LANCZOS)
        canvas = Image.new('RGB', (W, H), BG)
        canvas.paste(pic, ((W - size[0]) // 2, 60))
        d = ImageDraw.Draw(canvas)
        draw_text(d, W / 2, 60 + size[1] + 46, 'UNITY 6  ·  NEURON RENDER RUNTIME  ·  ' + CAPTIONS[part], f_label, DIM, tracking=3.0)
        draw_text(d, W / 2, 60 + size[1] + 80, 'one fit of stage.blend, exported as a pack; the camera, every light and material are component properties',
                  f_small, DIM)
        for _ in range(FPS // 24 + 1):   # the capture is one picture per animation frame, 24 a second
            yield canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('-o', '--output', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unity_stage.mp4'))
    a = ap.parse_args()
    encode(frames(a.capture, a.output), a.output)


if __name__ == '__main__':
    main()
