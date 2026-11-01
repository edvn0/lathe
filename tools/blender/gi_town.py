"""A large, hilly town for walking around with the probe GI on (docs/probe-gi.md).

    ~/dev/git/build_linux/bin/blender -b --python tools/blender/gi_town.py -- ~/Assets/Meshes/gi_town

writes gi_town.blend and gi_town.glb next to the given path stem. Units are metres, Z up in Blender (Lathe's world is the
model with Y and Z swapped: Blender (x, y, z) is (x, z, y) there). Run it with assets/scripts/gi_test/town_walk.lua.

The layout, and what each part tests:
  * 10 x 10 city blocks (54 m pitch, 14 m streets), about 550 m across, inside a ring of hills. Every block is a flat
    terrace; the streets are ramps between the terraces (grades up to about 12 %), so the town climbs and falls by about
    10 m: long sight lines uphill, shadowed streets downhill, probes that follow a camera that changes height.
  * "Open houses": hollow ground floor and upper floor, doors, windows, a doorway in an inner wall, a skylight, coloured
    interior walls and hanging lamps: sky light through openings, colour bleeding, emissive lighting of rooms.
  * "Towers": solid, 20 to 45 m, lit and dark windows (large emissive surfaces) and neon signs: a skyline for the far
    cascade and big coloured emitters.
  * A covered passage through one block with strip lights (a dark interior fed by emitters and bounce).
  * The plaza: a raised platform with stairs on four sides, emissive pillars, a fountain, market stalls, a ring of trees.
  * A watchtower on a hill, climbable by stairs, with a beacon on top.
  * Street lamps, strings of coloured bulbs, crates, benches, hedges and parked cars.
Everything is merged per material into a few big meshes except the bulbs, which stay separate objects: the probe GI
treats small emissive pieces as lights and larger emissive meshes as surfaces.
"""

import math
import random
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
STEM = argv[0] if argv else "/tmp/gi_town"

random.seed(20261101)

BLOCKS = 10
STREET = 14.0
BLOCK = 40.0
PITCH = BLOCK + STREET
TOWN = BLOCKS * PITCH + STREET
MARGIN = 140.0
FLOOR_H = 3.6
WALL_T = 0.3
CURB = 0.15

bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------------------------------------------------------------- materials

materials = {}


def material(name, colour, roughness=0.8, emission=None, strength=0.0):
    if name in materials:
        return materials[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    materials[name] = m
    return m


EXTERIOR = [
    (0.80, 0.72, 0.58),
    (0.78, 0.52, 0.42),
    (0.55, 0.68, 0.74),
    (0.84, 0.82, 0.76),
    (0.62, 0.74, 0.55),
    (0.82, 0.66, 0.30),
    (0.70, 0.56, 0.70),
    (0.60, 0.60, 0.64),
]
INTERIOR = [
    (0.85, 0.12, 0.10),
    (0.10, 0.65, 0.15),
    (0.12, 0.25, 0.85),
    (0.90, 0.78, 0.15),
    (0.85, 0.85, 0.82),
    (0.85, 0.35, 0.65),
]
for i, c in enumerate(EXTERIOR):
    material(f"wall_{i}", c, 0.85)
for i, c in enumerate(INTERIOR):
    material(f"room_{i}", c, 0.9)
material("asphalt", (0.07, 0.07, 0.08), 0.95)
material("grass", (0.10, 0.28, 0.07), 0.95)
material("sidewalk", (0.45, 0.45, 0.43), 0.9)
material("stone", (0.55, 0.54, 0.50), 0.85)
material("roof", (0.30, 0.17, 0.14), 0.85)
material("floor", (0.50, 0.40, 0.30), 0.7)
material("trunk", (0.28, 0.18, 0.10), 0.9)
material("leaves", (0.12, 0.40, 0.10), 0.9)
material("hedge", (0.08, 0.30, 0.08), 0.9)
material("wood", (0.45, 0.30, 0.15), 0.8)
material("metal", (0.10, 0.10, 0.11), 0.5)
material("lit_window", (0.05, 0.05, 0.05), 0.5, emission=(1.0, 0.72, 0.40), strength=4.0)
material("dark_window", (0.02, 0.03, 0.04), 0.2)
material("strip", (0.05, 0.05, 0.05), 0.5, emission=(0.85, 0.92, 1.0), strength=6.0)
material("pillar_cyan", (0.02, 0.05, 0.05), 0.4, emission=(0.1, 0.9, 1.0), strength=8.0)
material("pillar_magenta", (0.05, 0.02, 0.05), 0.4, emission=(1.0, 0.1, 0.8), strength=8.0)
material("pillar_amber", (0.05, 0.04, 0.02), 0.4, emission=(1.0, 0.55, 0.1), strength=8.0)
material("neon_pink", (0.05, 0.02, 0.04), 0.4, emission=(1.0, 0.15, 0.55), strength=7.0)
material("neon_green", (0.02, 0.05, 0.03), 0.4, emission=(0.2, 1.0, 0.35), strength=7.0)
material("neon_blue", (0.02, 0.03, 0.06), 0.4, emission=(0.2, 0.4, 1.0), strength=7.0)
material("beacon", (0.05, 0.05, 0.05), 0.4, emission=(1.0, 0.9, 0.6), strength=12.0)
for i, c in enumerate([(0.75, 0.1, 0.1), (0.1, 0.3, 0.7), (0.85, 0.8, 0.75), (0.15, 0.15, 0.17), (0.8, 0.6, 0.1)]):
    material(f"car_{i}", c, 0.35)
for i, c in enumerate([(0.8, 0.15, 0.15), (0.15, 0.55, 0.25), (0.85, 0.7, 0.15), (0.2, 0.35, 0.8)]):
    material(f"awning_{i}", c, 0.9)

BULB_COLOURS = {
    "lamp": (1.0, 0.70, 0.38),
    "red": (1.0, 0.12, 0.08),
    "green": (0.15, 1.0, 0.2),
    "blue": (0.15, 0.3, 1.0),
    "yellow": (1.0, 0.85, 0.15),
    "pink": (1.0, 0.3, 0.7),
    "white": (1.0, 0.95, 0.85),
    "orange": (1.0, 0.45, 0.1),
}
for name, c in BULB_COLOURS.items():
    material(f"bulb_{name}", (0.05, 0.05, 0.05), 0.4, emission=c, strength=30.0)

# ---------------------------------------------------------------- geometry accumulation

builders = {}
bulbs = []


def bm_of(mat):
    if mat not in builders:
        builders[mat] = bmesh.new()
    return builders[mat]


def quad(mat, p0, p1, p2, p3):
    bm = bm_of(mat)
    vs = [bm.verts.new(p) for p in (p0, p1, p2, p3)]
    try:
        bm.faces.new(vs)
    except ValueError:
        pass


def box(lo, hi, outside, inside=None, centre=None):
    """An axis aligned box. Vertical faces turned toward `centre` use the `inside` material (rooms with their own colour)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    if x1 - x0 < 1e-4 or y1 - y0 < 1e-4 or z1 - z0 < 1e-4:
        return
    faces = [
        ((1, 0, 0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1, 0, 0), ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1))),
        ((0, 1, 0), ((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1))),
        ((0, -1, 0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
        ((0, 0, 1), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0, 0, -1), ((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0))),
    ]
    mid = ((x0 + x1) / 2, (y0 + y1) / 2)
    for normal, corners in faces:
        mat = outside
        if inside is not None and centre is not None and normal[2] == 0:
            if normal[0] * (centre[0] - mid[0]) + normal[1] * (centre[1] - mid[1]) > 0:
                mat = inside
        quad(mat, *corners)


def wall(a, b, z0, z1, openings, outside, inside, centre):
    """A wall along x or y from a to b, thickness WALL_T centred on the line, with openings [(start, end, sill, head)]
    measured along the wall from its low end."""
    along_x = abs(a[1] - b[1]) < 1e-6
    length = abs(b[0] - a[0]) if along_x else abs(b[1] - a[1])
    origin = min(a[0], b[0]) if along_x else min(a[1], b[1])
    line = a[1] if along_x else a[0]
    half = WALL_T / 2

    def piece(s, e, lo, hi):
        if e - s < 1e-4 or hi - lo < 1e-4:
            return
        if along_x:
            box((origin + s, line - half, lo), (origin + e, line + half, hi), outside, inside, centre)
        else:
            box((line - half, origin + s, lo), (line + half, origin + e, hi), outside, inside, centre)

    cursor = 0.0
    for start, end, sill, head in sorted(openings):
        piece(cursor, start, z0, z1)
        piece(start, end, z0, z0 + sill)
        piece(start, end, z0 + head, z1)
        cursor = end
    piece(cursor, length, z0, z1)


def bulb(position, colour):
    bulbs.append((position, colour))


def stairs(x, y, z, direction, width, steps, rise=0.2, run=0.32, mat="stone"):
    """A straight flight rising from z at (x, y) along direction ('x+', 'x-', 'y+', 'y-'); (x, y) is the middle of the
    bottom edge. Solid steps, so they can be walked from the side as well."""
    for i in range(steps):
        d0, d1 = i * run, (i + 1) * run
        top = z + (i + 1) * rise
        if direction == "x+":
            box((x + d0, y - width / 2, z - 0.5), (x + d1, y + width / 2, top), mat)
        elif direction == "x-":
            box((x - d1, y - width / 2, z - 0.5), (x - d0, y + width / 2, top), mat)
        elif direction == "y+":
            box((x - width / 2, y + d0, z - 0.5), (x + width / 2, y + d1, top), mat)
        else:
            box((x - width / 2, y - d1, z - 0.5), (x + width / 2, y - d0, top), mat)


# ---------------------------------------------------------------- terrain

LEVELS = {}
for _bx in range(BLOCKS):
    for _by in range(BLOCKS):
        LEVELS[(_bx, _by)] = round(3.0 * math.sin(_bx * 0.45 + 1.0) * math.sin(_by * 0.38 + 0.5) + 0.3 * _bx, 1)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def axis_blend(v):
    """For a coordinate along one axis: the block index below it and how far toward the next block's level it is
    (0 inside a block, a smooth 0 to 1 across the street after it)."""
    u = v - STREET
    index = math.floor(u / PITCH)
    t = u - index * PITCH
    blend = 0.0 if t < BLOCK else smoothstep((t - BLOCK) / STREET)
    return index, blend


def level(ix, iy):
    return LEVELS[(max(0, min(BLOCKS - 1, ix)), max(0, min(BLOCKS - 1, iy)))]


def hills(x, y):
    return (
        16.0 * (0.5 + 0.5 * math.sin(x * 0.012 + 0.7) * math.cos(y * 0.010 + 1.3))
        + 7.0 * math.sin(x * 0.031 + y * 0.022 + 2.0) ** 2
        + 4.0 * math.sin(x * 0.07) * math.sin(y * 0.06)
    )


def ground(x, y):
    ix, sx = axis_blend(x)
    iy, sy = axis_blend(y)
    low = level(ix, iy) * (1 - sx) + level(ix + 1, iy) * sx
    high = level(ix, iy + 1) * (1 - sx) + level(ix + 1, iy + 1) * sx
    h = low * (1 - sy) + high * sy
    outside = max(-x, x - TOWN, -y, y - TOWN, 0.0)
    if outside > 0:
        h += hills(x, y) * smoothstep(outside / MARGIN)
    return h


def in_block(x, y):
    if not (STREET <= x <= TOWN - STREET and STREET <= y <= TOWN - STREET):
        return False
    ix, _ = axis_blend(x)
    iy, _ = axis_blend(y)
    return x - STREET - ix * PITCH < BLOCK and y - STREET - iy * PITCH < BLOCK


STEP = 2.0
LO = -MARGIN
COUNT = int((TOWN + 2 * MARGIN) / STEP)
heights = {(i, j): ground(LO + i * STEP, LO + j * STEP) for i in range(COUNT + 1) for j in range(COUNT + 1)}
for i in range(COUNT):
    for j in range(COUNT):
        cx, cy = LO + (i + 0.5) * STEP, LO + (j + 0.5) * STEP
        if in_block(cx, cy):
            continue  # under a block's slab
        mat = "asphalt" if (0 <= cx <= TOWN and 0 <= cy <= TOWN) else "grass"
        x0, x1 = LO + i * STEP, LO + (i + 1) * STEP
        y0, y1 = LO + j * STEP, LO + (j + 1) * STEP
        quad(mat, (x0, y0, heights[(i, j)]), (x1, y0, heights[(i + 1, j)]), (x1, y1, heights[(i + 1, j + 1)]),
             (x0, y1, heights[(i, j + 1)]))

# ---------------------------------------------------------------- buildings


def open_house(x, y, w, d, floors, facing, z):
    outside = f"wall_{random.randrange(len(EXTERIOR))}"
    inside = f"room_{random.randrange(len(INTERIOR))}"
    centre = (x + w / 2, y + d / 2, 0)
    corners = [(x, y), (x + w, y), (x + w, y + d), (x, y + d)]
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        length = w if side % 2 == 0 else d
        for floor in range(floors):
            z0 = z + floor * FLOOR_H
            openings = []
            windows = max(2, int(length // 4.5))
            slot = length / windows
            for i in range(windows):
                c = (i + 0.5) * slot
                if side == facing and floor == 0 and i == windows // 2:
                    openings.append((c - 0.75, c + 0.75, 0.0, 2.4))
                else:
                    openings.append((c - 0.9, c + 0.9, 0.9, 2.3))
            wall(a, b, z0, z0 + FLOOR_H, openings, outside, inside, centre)

    for floor in range(floors + 1):
        zf = z + floor * FLOOR_H
        top = floor == floors
        mat = "roof" if top else "floor"
        hole = (x + w / 2 - 1.5, y + d / 2 - 1.5, x + w / 2 + 1.5, y + d / 2 + 1.5)
        skylight = (top and random.random() < 0.7) or (not top and floor > 0)
        if skylight:
            for lo_, hi_ in [
                ((x, y, zf - 0.15), (hole[0], y + d, zf + 0.15)),
                ((hole[2], y, zf - 0.15), (x + w, y + d, zf + 0.15)),
                ((hole[0], y, zf - 0.15), (hole[2], hole[1], zf + 0.15)),
                ((hole[0], hole[3], zf - 0.15), (hole[2], y + d, zf + 0.15)),
            ]:
                box(lo_, hi_, mat, "floor", (centre[0], centre[1], zf + 5))
        else:
            box((x, y, zf - 0.15), (x + w, y + d, zf + 0.15), mat, "floor", (centre[0], centre[1], zf + 5))

    for floor in range(floors):
        z0 = z + floor * FLOOR_H
        wall((x, y + d / 2), (x + w, y + d / 2), z0, z0 + FLOOR_H, [(w / 2 - 0.8, w / 2 + 0.8, 0.0, 2.3)],
             f"wall_{random.randrange(len(EXTERIOR))}", f"room_{random.randrange(len(INTERIOR))}", centre)
        for lamp_y in (y + d * 0.25, y + d * 0.75):
            if random.random() < 0.6:
                bulb((x + w / 2 + random.uniform(-2.0, 2.0), lamp_y, z0 + FLOOR_H - 0.6), "lamp")
    if random.random() < 0.7:
        cx, cy = x + w / 2 + random.uniform(-3, 3), y + d * 0.25
        box((cx - 0.5, cy - 0.5, z), (cx + 0.5, cy + 0.5, z + 1.0), "wood")
        box((cx + 0.6, cy - 0.4, z), (cx + 1.4, cy + 0.4, z + 0.8), "wood")


def tower(x, y, w, d, height, z):
    outside = f"wall_{random.randrange(len(EXTERIOR))}"
    box((x, y, z - 3), (x + w, y + d, z + height), outside)
    box((x - 0.2, y - 0.2, z + height), (x + w + 0.2, y + d + 0.2, z + height + 0.6), "roof")
    lit_p = random.uniform(0.15, 0.55)
    for zz in [z + 3.0 + k * FLOOR_H for k in range(int((height - 3.0) // FLOOR_H))]:
        for side in range(4):
            length = w if side % 2 == 0 else d
            n = int(length // 3.0)
            for i in range(n):
                c = (i + 0.5) * length / n
                mat = "lit_window" if random.random() < lit_p else "dark_window"
                s0, s1, z0, z1 = c - 0.8, c + 0.8, zz, zz + 1.6
                # A shallow box, not a quad a few centimetres off the wall: that z-fights at a distance.
                dep = 0.25
                if side == 0:
                    box((x + s0, y - dep, z0), (x + s1, y, z1), mat)
                elif side == 1:
                    box((x + w, y + s0, z0), (x + w + dep, y + s1, z1), mat)
                elif side == 2:
                    box((x + w - s1, y + d, z0), (x + w - s0, y + d + dep, z1), mat)
                else:
                    box((x - dep, y + d - s1, z0), (x, y + d - s0, z1), mat)
    # A neon sign on the south face, a big emitter in a colour of its own.
    if random.random() < 0.6:
        neon = random.choice(["neon_pink", "neon_green", "neon_blue", "pillar_amber"])
        sz = z + random.uniform(4.0, min(height - 3.0, 14.0))
        box((x + w * 0.2, y - 0.25, sz), (x + w * 0.8, y - 0.05, sz + 1.2), neon)


def tunnel_tower(x, y, w, d, height, z):
    """A tower with a 5 m wide, 4 m high passage through it along x, lit by strip lights and a few bulbs."""
    outside = f"wall_{random.randrange(len(EXTERIOR))}"
    py0, py1 = y + d / 2 - 2.5, y + d / 2 + 2.5
    box((x, y, z - 3), (x + w, py0, z + height), outside)
    box((x, py1, z - 3), (x + w, y + d, z + height), outside)
    box((x, py0, z + 4.0), (x + w, py1, z + height), outside)
    box((x, py0, z - 0.2), (x + w, py1, z + 0.06), "floor")
    for i in range(int(w // 3)):
        cx = x + 1.5 + i * 3.0
        box((cx - 0.6, py0 + 2.3, z + 3.9), (cx + 0.6, py0 + 2.7, z + 3.97), "strip")
    for i in range(3):
        bulb((x + w * (i + 0.5) / 3, py0 + 0.5, z + 2.6), random.choice(["red", "blue", "orange"]))
    box((x - 0.2, y - 0.2, z + height), (x + w + 0.2, y + d + 0.2, z + height + 0.6), "roof")


def tree(x, y, z):
    height = random.uniform(4.0, 6.5)
    box((x - 0.2, y - 0.2, z - 0.5), (x + 0.2, y + 0.2, z + height * 0.55), "trunk")
    for k in range(3):
        bmesh.ops.create_icosphere(
            bm_of("leaves"), subdivisions=2, radius=random.uniform(1.6, 2.4),
            matrix=Matrix.Translation((x + random.uniform(-0.8, 0.8), y + random.uniform(-0.8, 0.8),
                                       z + height * 0.6 + k * 0.9)))


def lamp_post(x, y, arm_dx, arm_dy):
    z = ground(x, y)
    box((x - 0.08, y - 0.08, z - 0.3), (x + 0.08, y + 0.08, z + 5.0), "metal")
    box((x - 0.05 + min(0, arm_dx), y - 0.05 + min(0, arm_dy), z + 4.9),
        (x + 0.05 + max(0, arm_dx), y + 0.05 + max(0, arm_dy), z + 5.0), "metal")
    bulb((x + arm_dx, y + arm_dy, z + 4.75), "lamp")


def bench(x, y, z):
    box((x - 0.9, y - 0.25, z + 0.4), (x + 0.9, y + 0.25, z + 0.5), "wood")
    box((x - 0.9, y + 0.2, z + 0.5), (x + 0.9, y + 0.25, z + 0.95), "wood")
    box((x - 0.8, y - 0.2, z), (x - 0.7, y + 0.2, z + 0.4), "metal")
    box((x + 0.7, y - 0.2, z), (x + 0.8, y + 0.2, z + 0.4), "metal")


def car(x, y, along_x):
    """A parked car: a body and a cabin, resting on the highest ground under it."""
    lx, ly = (4.2, 1.8) if along_x else (1.8, 4.2)
    z = max(ground(x + sx * lx / 2, y + sy * ly / 2) for sx in (-1, 1) for sy in (-1, 1)) + 0.1
    body = f"car_{random.randrange(5)}"
    box((x - lx / 2, y - ly / 2, z + 0.25), (x + lx / 2, y + ly / 2, z + 0.85), body)
    if along_x:
        box((x - 1.2, y - ly / 2 + 0.1, z + 0.85), (x + 1.2, y + ly / 2 - 0.1, z + 1.35), "dark_window")
    else:
        box((x - lx / 2 + 0.1, y - 1.2, z + 0.85), (x + lx / 2 - 0.1, y + 1.2, z + 1.35), "dark_window")
    for sx in (-1, 1):
        for sy in (-1, 1):
            if along_x:
                box((x + sx * 1.3 - 0.3, y + sy * ly / 2 - 0.1, z), (x + sx * 1.3 + 0.3, y + sy * ly / 2 + 0.1, z + 0.5),
                    "metal")
            else:
                box((x + sx * lx / 2 - 0.1, y + sy * 1.3 - 0.3, z), (x + sx * lx / 2 + 0.1, y + sy * 1.3 + 0.3, z + 0.5),
                    "metal")


def stall(x, y, z):
    awning = f"awning_{random.randrange(4)}"
    for sx in (-1.2, 1.2):
        for sy in (-0.8, 0.8):
            box((x + sx - 0.05, y + sy - 0.05, z), (x + sx + 0.05, y + sy + 0.05, z + 2.4), "wood")
    box((x - 1.5, y - 1.1, z + 2.4), (x + 1.5, y + 1.1, z + 2.5), awning)
    box((x - 1.2, y - 0.5, z), (x + 1.2, y + 0.5, z + 0.9), "wood")
    bulb((x, y, z + 2.2), random.choice(["lamp", "yellow", "orange"]))


def watchtower(x, y, z):
    """An 8 x 8 m open tower with four stair flights up to 12 m and a beacon on top."""
    for cx in (x, x + 8):
        for cy in (y, y + 8):
            box((cx - 0.25, cy - 0.25, z - 1), (cx + 0.25, cy + 0.25, z + 13.0), "stone")
    for k in range(4):
        zl = z + 3.0 * (k + 1)
        box((x, y, zl - 0.2), (x + 8, y + 8, zl), "wood")
    # Each flight is 15 steps of 0.2 m: 3 m, from one landing to the next.
    stairs(x + 1.0, y + 1.0, z, "x+", 1.4, 15)
    stairs(x + 7.0, y + 1.5, z + 3.0, "y+", 1.4, 15)
    stairs(x + 6.5, y + 7.0, z + 6.0, "x-", 1.4, 15)
    stairs(x + 1.0, y + 6.5, z + 9.0, "y-", 1.4, 15)
    box((x - 0.5, y - 0.5, z + 12.0), (x + 8.5, y + 8.5, z + 12.2), "wood")
    box((x + 3.2, y + 3.2, z + 12.2), (x + 4.8, y + 4.8, z + 13.4), "beacon")
    bulb((x + 4.0, y + 4.0, z + 13.9), "white")


# ---------------------------------------------------------------- the city

plaza = (BLOCKS // 2, BLOCKS // 2)
tunnel_block = (2, 3)
tower_hill = (BLOCKS - 1, 1)

for bx in range(BLOCKS):
    for by in range(BLOCKS):
        x0 = STREET + bx * PITCH
        y0 = STREET + by * PITCH
        z = LEVELS[(bx, by)]
        zt = z + CURB
        box((x0, y0, z - 4.0), (x0 + BLOCK, y0 + BLOCK, zt), "sidewalk")
        lots = [(x0 + 1.5, y0 + 1.5), (x0 + BLOCK / 2 + 1.0, y0 + 1.5), (x0 + 1.5, y0 + BLOCK / 2 + 1.0),
                (x0 + BLOCK / 2 + 1.0, y0 + BLOCK / 2 + 1.0)]
        lot_w = BLOCK / 2 - 2.5

        if (bx, by) == plaza:
            cx, cy = x0 + BLOCK / 2, y0 + BLOCK / 2
            # A raised platform reached by stairs on four sides, with a fountain, pillars, stalls and trees.
            pz = zt + 2.0
            box((cx - 14, cy - 14, zt - 1), (cx + 14, cy + 14, pz), "stone")
            for k in range(10):
                top = zt + (k + 1) * 0.2
                reach = 14 + (10 - k) * 0.32
                box((cx - reach, cy - 4, zt - 0.5), (cx + reach, cy + 4, top), "stone")
                box((cx - 4, cy - reach, zt - 0.5), (cx + 4, cy + reach, top), "stone")
            for i, name in enumerate(["pillar_cyan", "pillar_magenta", "pillar_amber", "pillar_cyan"]):
                px = cx + (-9 if i % 2 == 0 else 9)
                py = cy + (-9 if i < 2 else 9)
                box((px - 1.0, py - 1.0, pz), (px + 1.0, py + 1.0, pz + 9.0), name)
            box((cx - 3.0, cy - 3.0, pz), (cx + 3.0, cy + 3.0, pz + 0.6), "stone")
            box((cx - 2.2, cy - 2.2, pz + 0.6), (cx + 2.2, cy + 2.2, pz + 0.9), "stone")
            box((cx - 0.4, cy - 0.4, pz + 0.9), (cx + 0.4, cy + 0.4, pz + 3.5), "stone")
            for k in range(6):
                a = k / 6 * 2 * math.pi + 0.3
                stall(cx + math.cos(a) * 7.5, cy + math.sin(a) * 7.5, pz)
            for k in range(14):
                a = k / 14 * 2 * math.pi
                tree(cx + math.cos(a) * 17.5, cy + math.sin(a) * 17.5, zt)
            continue

        if (bx, by) == tunnel_block:
            tunnel_tower(x0 + 2, y0 + 2, BLOCK - 4, BLOCK - 4, 30.0, zt)
            continue

        if (bx, by) == tower_hill:
            watchtower(x0 + BLOCK / 2 - 4, y0 + BLOCK / 2 - 4, zt)
            for _ in range(10):
                tree(x0 + random.uniform(3, BLOCK - 3), y0 + random.uniform(3, BLOCK - 3), zt)
            continue

        for lx, ly in lots:
            kind = random.choices(["house", "tower", "park"], weights=[5, 3, 2])[0]
            if kind == "house":
                open_house(lx, ly, lot_w, lot_w, random.choice([1, 2, 2]), random.randrange(4), zt)
            elif kind == "tower":
                tower(lx, ly, lot_w, lot_w, random.uniform(20, 45), zt)
            else:
                for _ in range(3):
                    tree(lx + random.uniform(2, lot_w - 2), ly + random.uniform(2, lot_w - 2), zt)
                bench(lx + lot_w / 2, ly + 2.0, zt)
                box((lx, ly, zt), (lx + lot_w, ly + 0.6, zt + 1.0), "hedge")
                box((lx, ly, zt), (lx + 0.6, ly + lot_w, zt + 1.0), "hedge")

# Street lamps along the block edges and strings of bulbs across the streets between them.
for bx in range(BLOCKS):
    for by in range(BLOCKS):
        x0 = STREET + bx * PITCH
        y0 = STREET + by * PITCH
        for t in (10.0, 30.0):
            lamp_post(x0 + t, y0 - 0.6, 0, -0.9)
            lamp_post(x0 - 0.6, y0 + t, -0.9, 0)
for bx in range(BLOCKS):
    for by in range(BLOCKS):
        if random.random() < 0.30:
            x0 = STREET + bx * PITCH
            y0 = STREET + by * PITCH
            t = random.choice([10.0, 30.0])
            n = 9
            z_a, z_b = ground(x0 + t, y0), ground(x0 + t, y0 - STREET)
            for i in range(n + 1):
                f = i / n
                sag = 4.0 * f * (1 - f) * 1.2
                bulb((x0 + t, y0 - STREET * f, z_a * (1 - f) + z_b * f + 5.2 - sag),
                     random.choice(list(BULB_COLOURS)[1:]))

# Parked cars along the streets, crates and barrels at street corners.
for bx in range(BLOCKS):
    for by in range(BLOCKS):
        x0 = STREET + bx * PITCH
        y0 = STREET + by * PITCH
        if random.random() < 0.55:
            car(x0 + random.uniform(6, BLOCK - 6), y0 - 3.0, True)
        if random.random() < 0.45:
            car(x0 - 3.0, y0 + random.uniform(6, BLOCK - 6), False)
        if random.random() < 0.5:
            cx, cy = x0 - 1.0, y0 - 1.0
            zc = ground(cx, cy)
            box((cx - 0.5, cy - 0.5, zc), (cx + 0.5, cy + 0.5, zc + 1.0), "wood")
            box((cx - 0.4, cy + 0.7, zc), (cx + 0.4, cy + 1.5, zc + 0.9), "wood")

# Trees on the hills around the town.
for _ in range(500):
    tx = random.uniform(-MARGIN + 10, TOWN + MARGIN - 10)
    ty = random.uniform(-MARGIN + 10, TOWN + MARGIN - 10)
    if -4 < tx < TOWN + 4 and -4 < ty < TOWN + 4:
        continue
    tree(tx, ty, ground(tx, ty))

# ---------------------------------------------------------------- objects

collection = bpy.context.scene.collection
for name, bm in builders.items():
    # Every box was built from separate quads: weld them, so the engine's LOD simplification sees connected surfaces.
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(materials[name])
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

bulb_meshes = {}
for name in BULB_COLOURS:
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.1)
    mesh = bpy.data.meshes.new(f"bulb_{name}")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(materials[f"bulb_{name}"])
    bulb_meshes[name] = mesh
for i, (position, colour) in enumerate(bulbs):
    obj = bpy.data.objects.new(f"bulb_{i}", bulb_meshes[colour])
    obj.location = Vector(position)
    collection.objects.link(obj)

plaza_x = STREET + plaza[0] * PITCH + BLOCK / 2
plaza_y = STREET + plaza[1] * PITCH + BLOCK / 2
print(f"gi_town: {len(builders)} merged meshes, {len(bulbs)} bulbs, {TOWN:.0f} m across, "
      f"plaza at {plaza_x:.0f} {plaza_y:.0f} level {LEVELS[plaza]:.1f}")

bpy.ops.wm.save_as_mainfile(filepath=STEM + ".blend")
bpy.ops.export_scene.gltf(filepath=STEM + ".glb", export_format="GLB", export_yup=True, export_apply=True)
print("gi_town: wrote", STEM + ".blend", "and", STEM + ".glb")
