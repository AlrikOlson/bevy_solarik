"""Generate/check an analytic glass fixture for the separate bevy-sponza rig.

Generate into the rig assets directory, render with the commands in docs/glass.md,
then check the PNG. Requires numpy/Pillow only for checking.
"""
import argparse
import base64
import json
import math
from pathlib import Path
import struct


def generate(destination, hidden_room=False):
    """Five panes in front of white emission; red emission behind the camera."""
    data = bytearray()
    views, accessors = [], []

    def accessor(values, fmt, kind, component, bounds=None):
        offset = len(data)
        for value in values:
            data.extend(struct.pack("<" + fmt, *value))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data) - offset})
        item = {"bufferView": len(views) - 1, "componentType": component,
                "count": len(values), "type": kind}
        if bounds:
            item.update(min=bounds[0], max=bounds[1])
        accessors.append(item)
        return len(accessors) - 1

    pos = accessor([(-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0)],
                   "3f", "VEC3", 5126, ([-.5, -.5, 0], [.5, .5, 0]))
    normal = accessor([(0, 0, 1)] * 4, "3f", "VEC3", 5126)
    uv = accessor([(0, 0), (1, 0), (1, 1), (0, 1)], "2f", "VEC2", 5126)
    tangent = accessor([(1, 0, 0, 1)] * 4, "4f", "VEC4", 5126)
    indices = accessor([(0,), (1,), (2,), (0,), (2,), (3,)], "I", "SCALAR", 5125)
    materials, meshes, nodes = [], [], []

    def panel(name, position, scale, color, mode="OPAQUE", emission=(0, 0, 0),
              double_sided=True, rotation=None):
        materials.append({"name": name, "doubleSided": double_sided, "alphaMode": mode,
                          "alphaCutoff": .5, "emissiveFactor": list(emission),
                          "pbrMetallicRoughness": {"baseColorFactor": color,
                              "metallicFactor": 0, "roughnessFactor": 1}})
        if mode == "BLEND":
            materials[-1]["extensions"] = {"KHR_materials_transmission": {"transmissionFactor": 1}}
        meshes.append({"primitives": [{"attributes": {"POSITION": pos, "NORMAL": normal,
                    "TEXCOORD_0": uv, "TANGENT": tangent}, "indices": indices,
                    "material": len(materials) - 1}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1,
                      "translation": position, "scale": scale})
        if rotation:
            nodes[-1]["rotation"] = rotation

    # Large emitters fill the transmitted and reflected hemisphere at the probe points.
    panel("white backdrop", [0, 0, -2], [100, 100, 1], [0, 0, 0, 1], emission=(1, 1, 1))
    panel("red reflection", [0, 0, 4], [100, 100, 1], [0, 0, 0, 1], emission=(1, 0, 0))
    for name, x, color, mode, emission in ([] if hidden_room else [
        ("tinted", -2.4, [.25, .5, .75, 1], "BLEND", (0, 0, 0)),
        ("clear", -1.2, [1, 1, 1, 1], "BLEND", (0, 0, 0)),
        ("zero alpha", 0, [.25, .5, .75, 0], "BLEND", (0, 0, 0)),
        ("mask hole", 1.2, [0, 0, 0, 0], "MASK", (0, 1, 0)),
        ("mask solid", 2.4, [0, 0, 0, 1], "MASK", (0, 1, 0)),
    ]):
        panel(name, [x, 0, 0], [1, 2, 1], color, mode, emission)
    if hidden_room:
        # Five outward-facing walls and the glass front enclose the room.
        panel("room glass", [0, 0, 0], [4, 2, 1], [1, 1, 1, 1], "BLEND")
        panel("room back", [0, 0, -1], [4, 2, 1], [0, 0, 0, 1],
              emission=(0, .5, 0), double_sided=False, rotation=[0, 1, 0, 0])
        h = math.sqrt(.5)
        for name, position, scale, rotation in [
            ("left", [-2, 0, -.5], [1, 2, 1], [0, -h, 0, h]),
            ("right", [2, 0, -.5], [1, 2, 1], [0, h, 0, h]),
            ("top", [0, 1, -.5], [4, 1, 1], [-h, 0, 0, h]),
            ("bottom", [0, -1, -.5], [4, 1, 1], [h, 0, 0, h]),
        ]:
            panel("room " + name, position, scale, [0, 0, 0, 1],
                  double_sided=False, rotation=rotation)
    document = {"asset": {"version": "2.0", "generator": "Solarik analytic glass test"},
                "extensionsUsed": ["KHR_materials_transmission"],
                "buffers": [{"uri": "data:application/octet-stream;base64," +
                     base64.b64encode(data).decode(), "byteLength": len(data)}],
                "bufferViews": views, "accessors": accessors, "materials": materials,
                "meshes": meshes, "nodes": nodes,
                "scenes": [{"nodes": list(range(len(nodes)))}], "scene": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document), encoding="utf-8")


def check(path, hidden_room=False):
    """Compare image patches to the emitter/Fresnel closed form at EV100=0."""
    import numpy as np
    from PIL import Image
    srgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    # Bevy exposure is 2^-EV100 / 1.2.
    radiance = linear * 1.2
    height, width, _ = radiance.shape
    if hidden_room:
        measured = radiance[height // 2 - 8:height // 2 + 8,
                            width // 2 - 8:width // 2 + 8].mean(axis=(0, 1))
        expected = np.array([1 / 13, 6 / 13, 0])
        error = float(np.max(np.abs(measured - expected)))
        result = {"room": "back-facing opaque wall", "measured": measured.tolist(),
                  "expected": expected.tolist(), "max_error": error}
        print(json.dumps(result, indent=2))
        assert error < .04, result
        return
    focal = height / (2 * math.tan(math.radians(60) / 2))
    rows = []
    for name, x, tint in [("tinted", -2.4, [.25, .5, .75]), ("clear", -1.2, [1, 1, 1]),
                          ("zero alpha", 0, None), ("mask hole", 1.2, None),
                          ("mask solid", 2.4, None)]:
        px = round(width / 2 + x / 3 * focal)
        patch = radiance[height // 2 - 8:height // 2 + 8, px - 8:px + 8]
        measured = patch.mean(axis=(0, 1))
        cosine = 3 / math.sqrt(9 + x * x)
        f = .04 + .96 * (1 - cosine) ** 5
        reflection = 2 * f / (1 + f)
        expected = np.array([1, 1, 1], dtype=float)
        if tint:
            expected = reflection * np.array([1, 0, 0]) + (1 - reflection) * np.array(tint)
        if name == "mask solid":
            expected = np.array([0, 1, 0])
        error = float(np.max(np.abs(measured - expected)))
        rows.append({"pane": name, "measured": measured.tolist(), "expected": expected.tolist(),
                     "max_error": error})
        # A saturated historical baseline is not valid physical evidence.
        assert error < .04, rows[-1]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", type=Path)
    action.add_argument("--check", type=Path)
    parser.add_argument("--hidden-room", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate(args.generate, args.hidden_room)
    else:
        check(args.check, args.hidden_room)
