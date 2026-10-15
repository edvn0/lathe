#!/usr/bin/env python3
"""Writes the low-poly snake pieces used by assets/scripts/snake/ to assets/models/snake/.

head.glb faces +X (a tapered, flattened ellipsoid), body.glb is a squashed sphere. Both are one primitive with
positions, normals, texture coordinates and 16-bit indices, centred on the origin and about one unit wide (the game
scales them to a cell). The game overrides their materials, so each carries only a plain one. No dependencies.

    python3 tools/gen_snake_models.py
"""

import json
import math
import pathlib
import struct

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "models" / "snake"


def ellipsoid(radii, segments, rings, shape=None):
    """A UV sphere scaled by `radii`; `shape(direction, position)` may bend each position. Returns (positions,
    normals, uvs, indices)."""
    positions, normals, uvs, indices = [], [], [], []

    for ring in range(rings + 1):
        polar = math.pi * ring / rings
        for segment in range(segments + 1):
            azimuth = 2.0 * math.pi * segment / segments
            direction = (math.cos(azimuth) * math.sin(polar), math.cos(polar), math.sin(azimuth) * math.sin(polar))
            position = tuple(direction[i] * radii[i] for i in range(3))

            if shape:
                position = shape(direction, position)

            # The normal of an ellipsoid at a point is its position over the squared radii.
            n = tuple(position[i] / (radii[i] ** 2) for i in range(3))
            length = math.sqrt(sum(c * c for c in n)) or 1.0

            positions.append(position)
            normals.append(tuple(c / length for c in n))
            uvs.append((segment / segments, ring / rings))

    stride = segments + 1

    for ring in range(rings):
        for segment in range(segments):
            a = ring * stride + segment
            b = a + stride
            if ring != 0:
                indices += [a, a + 1, b]
            if ring != rings - 1:
                indices += [a + 1, b + 1, b]

    return positions, normals, uvs, indices


def head():
    def shape(d, p):
        # Taper to a snout toward +X, and sit a little lower at the front.
        front = max(0.0, d[0])
        return (p[0] + 0.06, p[1] - 0.05 * front, p[2] * (1.0 - 0.34 * front))

    return ellipsoid((0.56, 0.34, 0.44), 12, 7, shape)


def body():
    return ellipsoid((0.46, 0.38, 0.46), 10, 6)


def glb(mesh, name, colour):
    positions, normals, uvs, indices = mesh

    def pack_floats(rows):
        return b"".join(struct.pack("<%df" % len(r), *r) for r in rows)

    blobs = [pack_floats(positions), pack_floats(normals), pack_floats(uvs), struct.pack("<%dH" % len(indices), *indices)]
    offsets, cursor = [], 0

    for blob in blobs:
        offsets.append(cursor)
        cursor += len(blob)

    binary = b"".join(blobs)
    binary += b"\0" * (-len(binary) % 4)

    lows = [min(p[i] for p in positions) for i in range(3)]
    highs = [max(p[i] for p in positions) for i in range(3)]

    document = {
        "asset": {"version": "2.0", "generator": "tools/gen_snake_models.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": colour, "metallicFactor": 0.0, "roughnessFactor": 0.6}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(blobs[3]), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": lows, "max": highs},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
    }

    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)

    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(text), b"JSON") + text +
            struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mesh, colour in (("head", head(), [0.35, 0.85, 0.4, 1.0]), ("body", body(), [0.15, 0.6, 0.25, 1.0])):
        data = glb(mesh, name, colour)
        (OUT / (name + ".glb")).write_bytes(data)
        print("%s.glb: %d vertices, %d triangles, %d bytes" % (name, len(mesh[0]), len(mesh[3]) // 3, len(data)))


if __name__ == "__main__":
    main()
