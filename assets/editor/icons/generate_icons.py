#!/usr/bin/env python3
"""Generate white-silhouette editor icons (RGBA, alpha-shaped) for the
entity hierarchy panel. Icons are drawn supersampled with PIL ImageDraw and
downsampled for anti-aliasing, then tinted per-type at runtime in ImGui via
ImageWithBg's tint_col -- so every icon here is pure white with alpha only.
"""
import math
import os
from PIL import Image, ImageDraw

SS = 8                 # supersample factor
SIZE = 64               # final icon size (square)
CANVAS = SIZE * SS
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

WHITE = (255, 255, 255)


def new_canvas():
    return Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))


def save(img, name):
    small = img.resize((SIZE, SIZE), Image.LANCZOS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    small.save(path)
    print("wrote", path)


def px(v):
    """Map a 0..24 design-grid unit to supersampled canvas pixels."""
    return v * CANVAS / 24.0


def w(a=255):
    return (255, 255, 255, a)


def rot(cx, cy, x, y, deg):
    a = math.radians(deg)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a))


def poly(draw, pts, alpha=255):
    draw.polygon([(px(x), px(y)) for x, y in pts], fill=w(alpha))


def line(draw, x0, y0, x1, y1, width, alpha=255):
    lw = max(1, round(width * SS))
    draw.line([(px(x0), px(y0)), (px(x1), px(y1))], fill=w(alpha), width=lw)
    # Round caps: PIL's line() gives flat ends, so stamp a disc at each
    # endpoint to match the rounded-stroke look used throughout this set.
    r = lw / 2.0
    for (x, y) in [(x0, y0), (x1, y1)]:
        draw.ellipse([px(x) - r, px(y) - r, px(x) + r, px(y) + r], fill=w(alpha))


def circle(draw, cx, cy, r, alpha=255):
    draw.ellipse([px(cx - r), px(cy - r), px(cx + r), px(cy + r)], fill=w(alpha))


def ring(draw, cx, cy, r, width, alpha=255):
    lw = max(1, round(width * SS))
    draw.ellipse([px(cx - r), px(cy - r), px(cx + r), px(cy + r)], outline=w(alpha), width=lw)


def ellipse_ring(draw, cx, cy, rx, ry, width, alpha=255):
    lw = max(1, round(width * SS))
    draw.ellipse([px(cx - rx), px(cy - ry), px(cx + rx), px(cy + ry)], outline=w(alpha), width=lw)


def rect(draw, x0, y0, x1, y1, alpha=255):
    draw.rectangle([px(x0), px(y0), px(x1), px(y1)], fill=w(alpha))


def rrect_outline(draw, x0, y0, x1, y1, radius, width, alpha=255):
    lw = max(1, round(width * SS))
    draw.rounded_rectangle([px(x0), px(y0), px(x1), px(y1)], radius=px(radius), outline=w(alpha), width=lw)


def arc(draw, cx, cy, r, a0, a1, width, alpha=255):
    lw = max(1, round(width * SS))
    draw.arc([px(cx - r), px(cy - r), px(cx + r), px(cy + r)], a0, a1, fill=w(alpha), width=lw)


# ---------------------------------------------------------------- mesh
def icon_mesh():
    # Isometric cube with per-face alpha so a single white texture still
    # reads as shaded once tinted -- top brightest, left mid, right darkest.
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 13
    s = 7.2
    top_pt = (cx, cy - s * 0.98)
    right_pt = (cx + s * 0.86, cy - s * 0.30)
    left_pt = (cx - s * 0.86, cy - s * 0.30)
    bottom_pt = (cx, cy + s * 0.38)
    bl_pt = (cx - s * 0.86, cy + s * 0.92)
    br_pt = (cx + s * 0.86, cy + s * 0.92)
    bottom2_pt = (cx, cy + s * 1.60)
    # top rhombus
    poly(d, [top_pt, right_pt, bottom_pt, left_pt], alpha=255)
    # left face
    poly(d, [left_pt, bottom_pt, bottom2_pt, bl_pt], alpha=185)
    # right face
    poly(d, [right_pt, br_pt, bottom2_pt, bottom_pt], alpha=130)
    save(img, "mesh")


# ---------------------------------------------------------- point light
def icon_point_light():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 10.5
    circle(d, cx, cy, 4.6, alpha=255)
    rect(d, cx - 1.6, cy + 4.2, cx + 1.6, cy + 5.6, alpha=255)
    rect(d, cx - 2.0, cy + 5.9, cx + 2.0, cy + 6.9, alpha=200)
    for ang in (-90, -30, 30, 90, 150, 210):
        x0, y0 = rot(cx, cy, cx, cy - 7.4, ang)
        x1, y1 = rot(cx, cy, cx, cy - 9.6, ang)
        line(d, x0, y0, x1, y1, 1.6)
    save(img, "point_light")


# ----------------------------------------------------------- spot light
def icon_spot_light():
    # Frustum-shaped shade (narrow mount at top, wide light-spread at
    # bottom) with fan rays below -- reads as a spotlight/flashlight rather
    # than a pin when the mount and shade are clearly separated.
    img = new_canvas()
    d = ImageDraw.Draw(img)
    tl, tr = (10.2, 6.6), (13.8, 6.6)
    bl, br = (6.4, 16.6), (17.6, 16.6)
    poly(d, [tl, tr, br, bl], alpha=235)
    rect(d, 10.8, 2.4, 13.2, 6.0, alpha=255)
    line(d, 9.4, 2.4, 14.6, 2.4, 1.5, alpha=255)
    for ang, length in ((-24, 3.6), (0, 4.6), (24, 3.6)):
        x0, y0 = rot(12, 16.6, 12, 17.6, ang)
        x1, y1 = rot(12, 16.6, 12, 17.6 + length, ang)
        line(d, x0, y0, x1, y1, 1.5)
    save(img, "spot_light")


# ---------------------------------------------------------------- script
def icon_script():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 6.2, 3.4, 17.0, 20.6
    fold = 4.5
    rrect_outline(d, x0, y0, x1, y1, radius=1.6, width=1.6)
    poly(d, [(x1 - fold, y0), (x1, y0 + fold), (x1 - fold, y0 + fold)], alpha=210)
    for i, yy in enumerate((9.6, 12.8, 16.0)):
        alpha = 255 if i != 2 else 255
        x_end = x1 - 2.6 if i != 2 else x1 - 5.2
        line(d, x0 + 2.6, yy, x_end, yy, 1.4)
    save(img, "script")


# ---------------------------------------------------------------- player
def icon_player():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    circle(d, 12, 7.4, 3.6, alpha=255)
    d.rounded_rectangle([px(6.4), px(12.0), px(17.6), px(21.2)], radius=px(3.6), fill=w(255))
    save(img, "player")


# ---------------------------------------------------------------- bullet
def icon_bullet():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 13.5, 12
    circle(d, cx, cy, 3.2, alpha=255)
    for i, (off, alpha, r) in enumerate([(5.4, 150, 2.4), (9.6, 80, 1.7), (13.2, 35, 1.1)]):
        circle(d, cx - off, cy, r, alpha=alpha)
    save(img, "bullet")


# ----------------------------------------------------------------- empty
def icon_empty():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 12
    ring(d, cx, cy, 6.6, 1.6, alpha=235)
    circle(d, cx, cy, 1.5, alpha=255)
    for ang in (90, 210, 330):
        x0, y0 = rot(cx, cy, cx, cy - 6.6, ang)
        x1, y1 = rot(cx, cy, cx, cy - 9.4, ang)
        line(d, x0, y0, x1, y1, 1.5)
    save(img, "empty")


# ---------------------------------------------------------------- folder
def icon_folder():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    poly(d, [(3.4, 6.4), (9.6, 6.4), (11.2, 8.4), (20.6, 8.4), (20.6, 9.6), (3.4, 9.6)], alpha=255)
    d.rounded_rectangle([px(3.4), px(9.2), px(20.6), px(19.4)], radius=px(1.2), fill=w(235))
    save(img, "folder")


# ---------------------------------------------------------------- search
def icon_search():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    ring(d, 10.2, 10.2, 5.4, 1.8, alpha=255)
    hx0, hy0 = rot(10.2, 10.2, 10.2, 15.6, 45)
    hx1, hy1 = rot(10.2, 10.2, 10.2, 20.6, 45)
    line(d, hx0, hy0, hx1, hy1, 1.9)
    save(img, "search")


# ------------------------------------------------------------------ move
def icon_move():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 12
    for ang in (0, 90, 180, 270):
        x0, y0 = rot(cx, cy, cx, cy - 2.6, ang)
        x1, y1 = rot(cx, cy, cx, cy - 8.4, ang)
        line(d, x0, y0, x1, y1, 1.6)
        tip = rot(cx, cy, cx, cy - 9.6, ang)
        lft = rot(cx, cy, cx - 2.2, cy - 6.8, ang)
        rgt = rot(cx, cy, cx + 2.2, cy - 6.8, ang)
        poly(d, [tip, lft, rgt], alpha=255)
    circle(d, cx, cy, 1.7, alpha=255)
    save(img, "move")


# ---------------------------------------------------------------- rotate
def icon_rotate():
    # PIL's arc() angles run clockwise from 3 o'clock (screen convention,
    # y-down) -- the arrowhead is built in that same frame, at the arc's
    # end angle, pointing along the circle's clockwise tangent there.
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 12.5
    r = 7.0
    end_deg = 220.0
    arc(d, cx, cy, r, -50, end_deg, 1.8, alpha=255)

    theta = math.radians(end_deg)
    ex, ey = cx + r * math.cos(theta), cy + r * math.sin(theta)
    tx, ty = -math.sin(theta), math.cos(theta)  # unit tangent, clockwise travel
    nx, ny = math.cos(theta), math.sin(theta)   # outward normal

    tip = (ex + tx * 3.6, ey + ty * 3.6)
    base1 = (ex - nx * 2.6 + tx * -0.6, ey - ny * 2.6 + ty * -0.6)
    base2 = (ex + nx * 2.6 + tx * -0.6, ey + ny * 2.6 + ty * -0.6)
    poly(d, [tip, base1, base2], alpha=255)
    save(img, "rotate")


# ---------------------------------------------------------------- scale
def icon_scale():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    rrect_outline(d, 5.0, 5.0, 19.0, 19.0, radius=1.2, width=1.6, alpha=230)
    rect(d, 4.0, 4.0, 8.0, 8.0, alpha=255)
    rect(d, 16.0, 16.0, 20.0, 20.0, alpha=255)
    line(d, 8.6, 8.6, 15.4, 15.4, 1.4, alpha=210)
    save(img, "scale")


# ---------------------------------------------------------------- local
def icon_local():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 13
    s = 6.6
    top_pt = (cx, cy - s * 0.98)
    right_pt = (cx + s * 0.86, cy - s * 0.30)
    left_pt = (cx - s * 0.86, cy - s * 0.30)
    bottom_pt = (cx, cy + s * 0.38)
    bl_pt = (cx - s * 0.86, cy + s * 0.92)
    br_pt = (cx + s * 0.86, cy + s * 0.92)
    bottom2_pt = (cx, cy + s * 1.60)
    lw = 1.4
    for a, b in [(top_pt, right_pt), (right_pt, br_pt), (br_pt, bottom2_pt),
                 (bottom2_pt, bl_pt), (bl_pt, left_pt), (left_pt, top_pt),
                 (left_pt, bottom_pt), (bottom_pt, right_pt), (bottom_pt, bottom2_pt)]:
        line(d, a[0], a[1], b[0], b[1], lw, alpha=235)
    save(img, "local")


# ---------------------------------------------------------------- world
def icon_world():
    img = new_canvas()
    d = ImageDraw.Draw(img)
    cx, cy = 12, 12
    ring(d, cx, cy, 8.0, 1.6, alpha=255)
    ellipse_ring(d, cx, cy, 3.4, 8.0, 1.3, alpha=230)
    line(d, cx - 8.0, cy, cx + 8.0, cy, 1.3, alpha=230)
    save(img, "world")


ICONS = [icon_mesh, icon_point_light, icon_spot_light, icon_script, icon_player,
         icon_bullet, icon_empty, icon_folder, icon_search, icon_move, icon_rotate,
         icon_scale, icon_local, icon_world]

if __name__ == "__main__":
    for fn in ICONS:
        fn()
