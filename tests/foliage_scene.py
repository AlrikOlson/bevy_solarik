"""Controlled backlit diffuse cards for the bevy-sponza capture rig."""
import argparse
import json
import math
from pathlib import Path
from glass_scene import generate as glass_generate

def generate(path):
    glass_generate(path)
    scene = json.loads(path.read_text())
    scene["materials"][1]["emissiveFactor"] = [0, 0, 0]
    for index, (name, x) in enumerate([("leaf zero", -1.2), ("leaf half", 0), ("leaf full", 1.2)], 2):
        mat = scene["materials"][index]
        mat.update(name=name, alphaMode="MASK", emissiveFactor=[0, 0, 0])
        mat["pbrMetallicRoughness"]["baseColorFactor"] = [.2, .5, .8, 1]
        scene["nodes"][index]["translation"] = [x, 0, 0]
    scene["nodes"] = scene["nodes"][:5]
    scene["scenes"][0]["nodes"] = list(range(5))
    path.write_text(json.dumps(scene), encoding="utf-8")

def check(path):
    import numpy as np
    from PIL import Image
    srgb = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255
    image = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4) * 1.2
    height, width, _ = image.shape
    focal = height / (2 * math.tan(math.radians(60) / 2))
    rows = []
    for x, transmission in [(-1.2, 0), (0, .5), (1.2, 1)]:
        px = round(width / 2 + x / 3 * focal)
        measured = image[height//2-8:height//2+8, px-8:px+8].mean(axis=(0, 1))
        cosine = 3 / math.sqrt(9 + x*x)
        entry = .96 * (1 - (1-cosine)**5)
        expected = np.array([.2, .5, .8]) * transmission * entry * .96 * 20/21
        rows.append(dict(transmission=transmission, measured=measured.tolist(), expected=expected.tolist()))
        assert np.max(np.abs(measured-expected)) < .04, rows[-1]
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", type=Path)
    action.add_argument("--check", type=Path)
    args = p.parse_args()
    generate(args.generate) if args.generate else check(args.check)

