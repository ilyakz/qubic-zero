#!/usr/bin/env python
"""Анимация для начала README: настоящая проекция доски 4×4×4×4 и линия через все четыре измерения.

    python tools/make_hero.py
"""
import itertools
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WIDTH, HEIGHT, SCALE, FRAMES = 720, 400, 2, 90
BACKGROUND, FRAME, CELL = (13, 13, 13), (58, 58, 56), (150, 149, 140)
CROSS, NOUGHT, WHITE = (57, 135, 229), (217, 89, 38), (255, 255, 255)
HALF = 1.5
LINE = [(i, i, i, i) for i in range(4)]  # главная диагональ гиперкуба
NOUGHTS = [(0, 1, 2, 3), (1, 3, 0, 2), (2, 0, 3, 1)]


def project(point, turn):
    """Двойное вращение в двух независимых плоскостях (оно замыкается в кольцо), затем две перспективы."""
    x, y, z, w = ((v - HALF) / HALF for v in point)
    c, s = math.cos(turn), math.sin(turn)
    x, w = x * c - w * s, x * s + w * c
    y, z = y * c - z * s, y * s + z * c
    near = 3 / (3 - w)
    x, y, z = x * near, y * near, z * near
    cos_a, sin_a = math.cos(0.62), math.sin(0.62)
    x, z = x * cos_a - z * sin_a, x * sin_a + z * cos_a
    cos_b, sin_b = math.cos(0.42), math.sin(0.42)
    y, z = y * cos_b - z * sin_b, y * sin_b + z * cos_b
    k = 9 / (9 + z)
    size = (HEIGHT / 2 - 8) * SCALE / 2.8
    return WIDTH * SCALE / 2 + x * size * k, HEIGHT * SCALE / 2 + y * size * k, z, k * near


def frame(turn):
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    corners = list(itertools.product((0, 3), repeat=4))
    for a, b in itertools.combinations(corners, 2):
        if sum(p != q for p, q in zip(a, b)) == 1:
            pa, pb = project(a, turn), project(b, turn)
            draw.line((pa[0], pa[1], pb[0], pb[1]), fill=FRAME, width=2)
    cells = sorted(itertools.product(range(4), repeat=4), key=lambda p: -project(p, turn)[2])
    for point in cells:
        if point in LINE or point in NOUGHTS:
            continue
        x, y, _, k = project(point, turn)
        r = max(2.0, 2.4 * k) * SCALE
        shade = min(1.0, 0.25 + 0.3 * k)
        color = tuple(int(b + (c - b) * shade) for c, b in zip(CELL, BACKGROUND))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    glow = Image.new("RGB", image.size, (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    points = [project(p, turn)[:2] for p in LINE]
    glow_draw.line(points, fill=CROSS, width=11 * SCALE, joint="curve")
    glow = glow.filter(ImageFilter.GaussianBlur(9 * SCALE))
    image = Image.fromarray(np.clip(np.asarray(image, dtype=int) + np.asarray(glow, dtype=int), 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(image)
    draw.line(points, fill=CROSS, width=4 * SCALE, joint="curve")
    draw.line(points, fill=WHITE, width=1 * SCALE, joint="curve")
    for point, color in [(p, NOUGHT) for p in NOUGHTS] + [(p, CROSS) for p in LINE]:
        x, y, _, k = project(point, turn)
        r = max(5.0, 6.0 * k) * SCALE
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        if color == CROSS:
            a = r * 0.45
            draw.line((x - a, y - a, x + a, y + a), fill=WHITE, width=max(2, int(r * 0.26)))
            draw.line((x + a, y - a, x - a, y + a), fill=WHITE, width=max(2, int(r * 0.26)))
        else:
            a = r * 0.5
            draw.ellipse((x - a, y - a, x + a, y + a), outline=WHITE, width=max(2, int(r * 0.26)))
    return image.resize((WIDTH, HEIGHT), Image.LANCZOS)


def main():
    frames = [frame(2 * math.pi * i / FRAMES) for i in range(FRAMES)]
    palette = frames[0].convert("P", palette=Image.ADAPTIVE, colors=128)
    paletted = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    path = os.path.join(ROOT, "docs", "img", "hero.gif")
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=60, loop=0, optimize=True, disposal=1)
    print(path, round(os.path.getsize(path) / 1e6, 2), "МБ")


if __name__ == "__main__":
    main()
