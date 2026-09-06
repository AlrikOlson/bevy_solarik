"""Paired colored-pane/point-light fixture for the separate local render rig.

Generate into an asset directory, capture control/tinted/opaque at camera
(0,0,0.5) looking at the origin, EV100=2, no sun/moon/IBL, reference mode.
The camera is between receiver and pane; only the light path crosses glass.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import glass_scene


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, alpha, mode in [("control", 0, "BLEND"), ("tinted", 1, "BLEND"),
                               ("opaque", 1, "OPAQUE")]:
        path = directory / f"shadow_{name}.gltf"
        glass_scene.generate(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        primitive = document["meshes"][0]["primitives"][0]
        wall = {"name": "shadow receiver", "pbrMetallicRoughness": {
            "baseColorFactor": [.05, .05, .05, 1], "metallicFactor": 0,
            "roughnessFactor": 1}, "extensions": {
                "KHR_materials_specular": {"specularFactor": 0}}}
        pane = {"name": "shadow pane", "alphaMode": mode, "doubleSided": True,
                "pbrMetallicRoughness": {"baseColorFactor": [.25, .5, .75, alpha],
                    "metallicFactor": 0, "roughnessFactor": 1},
                "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1}}}
        document["materials"] = [wall, pane]
        document["meshes"] = []
        for material in range(2):
            item = copy.deepcopy(primitive)
            item["material"] = material
            document["meshes"].append({"primitives": [item]})
        document["nodes"] = [
            {"mesh": 0, "scale": [6, 4, 1]},
            {"mesh": 1, "scale": [10, 10, 1], "translation": [0, 0, 1]},
            {"translation": [0, 0, 2], "extensions": {"KHR_lights_punctual": {"light": 0}}},
        ]
        document["scenes"] = [{"nodes": [0, 1, 2]}]
        document["extensionsUsed"].append("KHR_lights_punctual")
        document["extensions"] = {"KHR_lights_punctual": {"lights": [{
            "type": "point", "color": [1, 1, 1], "intensity": 8000 / (4 * math.pi)}]}}
        path.write_text(json.dumps(document), encoding="utf-8")


def check(directory):
    import numpy as np
    from PIL import Image

    measured = {}
    for name in ("control", "tinted", "opaque"):
        srgb = np.asarray(Image.open(directory / name / "seq_0000.png").convert("RGB"),
                          dtype=np.float64) / 255
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        h, w, _ = linear.shape
        measured[name] = linear[h//2-8:h//2+8, w//2-8:w//2+8].mean(axis=(0, 1)) * 4.8
    ratio = measured["tinted"] / measured["control"]
    expected = np.array([.25, .5, .75]) * 12 / 13
    result = {"radiance": {k: v.tolist() for k, v in measured.items()},
              "transmission": ratio.tolist(), "expected": expected.tolist()}
    print(json.dumps(result, indent=2))
    # Wall albedo .05 keeps pane/receiver interreflection small; tolerance includes
    # finite point-light sphere sampling, PNG quantization and this extra transport.
    assert np.max(np.abs(ratio - expected)) < .03, result
    assert np.max(measured["opaque"]) < .01, result
    assert np.min(measured["control"]) > 2.3, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", type=Path)
    action.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.generate:
        generate(args.generate)
    else:
        check(args.check)

