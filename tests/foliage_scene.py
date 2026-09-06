"""Controlled backlit diffuse cards for the bevy-sponza capture rig."""
import argparse
import json
import math
from pathlib import Path
from glass_scene import generate as glass_generate

def generate(path, reflected=False):
    glass_generate(path)
    scene = json.loads(path.read_text())
    scene["materials"][1]["emissiveFactor"] = [0, 0, 0]
    for index, (name, x) in enumerate([("leaf zero", -1.2), ("leaf half", 0), ("leaf full", 1.2)], 2):
        mat = scene["materials"][index]
        mat.update(name=name, alphaMode="MASK", emissiveFactor=[0, 0, 0])
        mat["pbrMetallicRoughness"]["baseColorFactor"] = [.2, .5, .8, 1]
        scene["nodes"][index]["translation"] = [x, 0, 0]
    scene["nodes"] = scene["nodes"][:5]
    if reflected:
        # The camera at z=3 sees the cards at z=6 only through a mirror at z=0.
        # A white emitter at z=8 lights the opposite side of each card.
        import copy
        scene["nodes"][0]["translation"] = [0, 0, 8]
        scene["nodes"][1]["translation"] = [0, 0, 10]
        for node in scene["nodes"][2:5]:
            node["translation"][2] = 6
            node["rotation"] = [0, 1, 0, 0]
        mirror = copy.deepcopy(scene["materials"][0])
        mirror.update(name="mirror", emissiveFactor=[0, 0, 0])
        mirror["pbrMetallicRoughness"].update(
            baseColorFactor=[1, 1, 1, 1], metallicFactor=1, roughnessFactor=0)
        scene["materials"].append(mirror)
        mesh = copy.deepcopy(scene["meshes"][0])
        mesh["primitives"][0]["material"] = len(scene["materials"]) - 1
        scene["meshes"].append(mesh)
        scene["nodes"].append(dict(name="mirror", mesh=len(scene["meshes"])-1,
                                   translation=[0, 0, 0], scale=[100, 100, 1]))
    scene["scenes"][0]["nodes"] = list(range(len(scene["nodes"])))
    path.write_text(json.dumps(scene), encoding="utf-8")

def check(path, reflected=False, reference=None):
    import numpy as np
    from PIL import Image
    srgb = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255
    image = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4) * 1.2
    reference_image = None
    if reflected:
        if reference is None:
            raise ValueError("--reflected --check requires --reference: the mirror illuminates both hemispheres")
        srgb_reference = np.asarray(Image.open(reference).convert("RGB"), dtype=float) / 255
        reference_image = np.where(srgb_reference <= .04045, srgb_reference / 12.92,
                                   ((srgb_reference + .055) / 1.055) ** 2.4) * 1.2
        assert reference_image.shape == image.shape
    height, width, _ = image.shape
    focal = height / (2 * math.tan(math.radians(60) / 2))
    distance = 9 if reflected else 3
    rows = []
    for x, transmission in [(-1.2, 0), (0, .5), (1.2, 1)]:
        px = round(width / 2 + x / distance * focal)
        measured = image[height//2-8:height//2+8, px-8:px+8].mean(axis=(0, 1))
        cosine = distance / math.sqrt(distance*distance + x*x)
        entry = .96 * (1 - (1-cosine)**5)
        expected = np.array([.2, .5, .8]) * transmission * entry * .96 * 20/21
        if reflected and transmission > 0:
            expected = reference_image[height//2-8:height//2+8, px-8:px+8].mean(axis=(0, 1))
        # t=0 retains the pre-existing opaque glossy estimator (black in this fixture).
        rows.append(dict(transmission=transmission, measured=measured.tolist(), expected=expected.tolist()))
        assert np.max(np.abs(measured-expected)) < .04, rows[-1]
    print(json.dumps(rows, indent=2))

def check_motion(path):
    """Measure all 60 frames from foliage_reflection_scene.toml at 640x320."""
    import numpy as np
    from PIL import Image
    paths = sorted(path.glob("seq_*.png"))
    assert len(paths) == 60, "capture all 60 fixture frames without skipping"
    values = []
    for index, image_path in enumerate(paths):
        srgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=float) / 255
        assert srgb.shape == (320, 640, 3), "documented fixture resolution"
        image = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4) * 1.2
        u = index / 60
        camera_x = -.5 + u*u*(3-2*u)
        focal = 320 / (2 * math.tan(math.pi/6))
        row = []
        for x in [-1.2, 0, 1.2]:
            px = round(320 + (x-camera_x)/9*focal)
            row.append(image[152:168, px-8:px+8].mean(axis=(0, 1)))
        values.append(row)
    values = np.asarray(values)
    changes = np.abs(np.diff(values, axis=0))
    assert values[:, 0].max() < .01, "opaque control changed during motion"
    assert changes[:, 1:].max() < .04, "leaf patch energy jumps between adjacent frames"
    # A stable black image is not a successful transmission check.
    assert values[:, 1:, 2].min() > .65, "reflected leaf transmission is missing"
    print(json.dumps(dict(frames=len(paths), rgb_min=values.min(axis=0).tolist(),
                         rgb_max=values.max(axis=0).tolist(), rgb_mean=values.mean(axis=0).tolist(),
                         max_frame_delta=changes.max(axis=0).tolist()), indent=2))


def check_bistro(path, baseline):
    """Check opaque controls and confirm the canopy estimator actually changed."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(baseline).convert("RGB"), dtype=float) / 255
    b = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255
    assert a.shape == b.shape == (720, 1280, 3), "documented Bistro frame0 camera"
    rows = []
    for name, (x, y, w, h) in {"trunk": (420, 450, 65, 150), "wall": (900, 330, 300, 130),
                             "canopy": (350, 0, 650, 160)}.items():
        aa, bb = a[y:y+h, x:x+w], b[y:y+h, x:x+w]
        error = float(np.abs(aa-bb).mean())
        rows.append(dict(region=name, baseline=aa.mean(axis=(0, 1)).tolist(),
                         foliage=bb.mean(axis=(0, 1)).tolist(), mean_absolute_difference=error))
        if name != "canopy":
            assert error < .01, rows[-1]
        else:
            assert error > .02, "foliage estimator did not affect the canopy"
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", type=Path)
    action.add_argument("--check", type=Path)
    p.add_argument("--bistro-baseline", type=Path)
    p.add_argument("--reflected", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--motion", action="store_true")
    args = p.parse_args()
    if args.generate:
        generate(args.generate, args.reflected)
    elif args.motion:
        check_motion(args.check)
    elif args.bistro_baseline:
        check_bistro(args.check, args.bistro_baseline)
    else:
        check(args.check, args.reflected, args.reference)

