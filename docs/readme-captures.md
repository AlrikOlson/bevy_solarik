# README capture record

## Current images with the corrected Bistro interior

September 6, 2026 (local): refreshed from the final integrated release build
after repairing the Bistro interior import. Source bases: Solarik `df9a05a`,
DLSS5 `e8b2703` and playground `740e78a`, plus the paired-import fix.
Executable SHA-256:
`3a996f4bad28d5fc5ed428e41546ec78a9c65924406b2450d9844cf25b514056`.
Windows/Vulkan, RTX 4090, NVIDIA Studio driver 616.56. Build with
`cargo build --release --locked` in the playground.

The qian-o exterior needs import scale **0.625**; its interior needs **1.0**.
The old recipe incorrectly applied 0.625 to both. Reimporting aligns eight
shared facade anchors within 0.004 mm and makes the audited window rays hit
real room surfaces. Those earlier bright window patches were empty-world
visibility, not emissive interiors. Run
`python tools/import_bistro.py "C:/scenes/GLTF-Assets/Bistro"` in the playground
to repair an old local import. Its `docs/bistro-glass.md` contains the
before/after crops and ray-distance regression.

Solarik's GPU atmosphere and native-resolution DLSS Ray Reconstruction remain
active. The atmosphere uses High quality, density scales 1, Mie g 0.8,
2,000 m aerial distance and IBL scale 1. Exposure uses the preset's
12.5/5.0 EV100 day/night curve; scene lamps are unchanged. Extra grading and
local artistic fog are disabled (`SCENE_FX=blur`); GPU timestamp diagnostics
are off. Camera and lighting stay fixed throughout warmup. Historical
`night` filenames contain dusk views.

| README asset | Resolution | Pose | Fixed warmup frames |
| --- | --- | --- | ---: |
| bistro-day.png | 3840×2160 | Timeline 750 / 1800 | 512 |
| bistro-night.png (dusk) | 3840×2160 | Timeline 1520 / 1800 | 1024 |
| bistro-temporal-response.png | 1280×720 | Independent −14° sun control | 4096 |

Neural Rendering is off in all three images. The temporal illustration is a
fresh static night control; the earlier sunset-response measurements remain
in [temporal response](temporal-response.md).

Independent warmups use identical camera and lighting. The metric is regional
mean sRGB-decoded display-linear luminance, with weights (0.2126, 0.7152, 0.0722).
Absolute changes are on the same 0–1 display-linear scale.

| View / region | Rectangle x, y, width, height | Warmup comparison | Luminance change | Absolute change |
| --- | --- | --- | ---: | ---: |
| Day pavement | 430, 1930, 550, 190 | 128 to 512 | +2.68% | +0.000358 |
| Day facade | 2250, 100, 900, 450 | 128 to 512 | +2.86% | +0.000899 |
| Day wall | 1110, 1180, 500, 330 | 128 to 512 | +5.51% | +0.000425 |
| Dusk pavement | 100, 1720, 760, 250 | 512 to 1024 | +1.12% | +0.000113 |
| Dusk facade | 130, 150, 670, 450 | 512 to 1024 | -2.64% | -0.002523 |
| Dusk wall | 2150, 1660, 510, 280 | 512 to 1024 | -2.20% | -0.000067 |
| Night pavement | 40, 560, 230, 120 | 512 to 1024 | -2.38% | -0.001226 |
| Night facade | 40, 40, 200, 140 | 512 to 1024 | +0.37% | +0.001001 |
| Night cafe wall | 900, 500, 220, 100 | 512 to 1024 | -3.38% | -0.000163 |
| Night pavement | 40, 560, 230, 120 | 1024 to 2048 | +1.50% | +0.000756 |
| Night facade | 40, 40, 200, 140 | 1024 to 2048 | -0.20% | -0.000537 |
| Night cafe wall | 900, 500, 220, 100 | 1024 to 2048 | +11.65% | +0.000543 |
| Night pavement | 40, 560, 230, 120 | 2048 to 4096 | -0.86% | -0.000441 |
| Night facade | 40, 40, 200, 140 | 2048 to 4096 | -0.94% | -0.002534 |
| Night cafe wall | 900, 500, 220, 100 | 2048 to 4096 | -5.82% | -0.000303 |

The day wall still changes +5.51%. The night control was extended to 4,096
frames after the cafe wall changed +11.65% from 1,024 to 2,048; its final
change is −5.82% (−0.000303). Pavement and facade change less than 0.95% in
that last pair. This remaining dark-region variation is recorded with the
existing night-image-quality follow-up, not treated as resolved by warmup.

From the playground in Git Bash, after importing the corrected pair and
building the release executable:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=750 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=512 SCENE_FX=blur SCENE_DIAG=0 NR_OUT=../artifacts/bevy-sponza/glass-leak-readme-day-512 python tools/launch.py capture off
SCENE=bistro NR_RES=3840x2160 NR_START=1520 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=1024 SCENE_FX=blur SCENE_DIAG=0 NR_OUT=../artifacts/bevy-sponza/glass-leak-readme-dusk-1024 python tools/launch.py capture off
SCENE=bistro NR_RES=1280x720 NR_START=0 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=4096 SCENE_FX=blur SCENE_DIAG=0 SCENE_SUN=-14,-14,90,90 SCENE_CAM_FROM=4,1.6,12 SCENE_LOOK_FROM=-3,3,-3 SCENE_FOV=50 NR_OUT=../artifacts/bevy-sponza/glass-leak-temporal-4096 python tools/launch.py capture off
```

All final images were inspected. These bounded regional comparisons do not
establish full convergence; dark interiors, foliage softness and stochastic
spatial variation remain.

All 18 serial captures produced fresh PNGs, exited normally and had
no shader/validation errors or Windows NVIDIA driver events. Both NR ON logs
confirm active evaluation and explicit feature release before runtime shutdown.
All runtime source and imported geometry hashes match the capture manifest.
The only removed manifest entry was an unused standalone interior preset
created by the importer; the assembled Bistro loads the interior glTF directly.

The organizer's local `artifacts/bevy-sponza/glass_leak_readme_capture.py`
retains the recipes and `glass-leak-*/capture-result.json` the exact settings.
`glass-leak-build.json`, `glass-leak-settling.json`,
`glass-leak-driver-audit.json` and `glass-leak-final-audit.json` record the
source/asset hashes, measurements and checks. `glass-leak-readme-inventory.json`
verifies all **12** README image files across the three active repositories,
including details blocks and matching duplicates. The four comparison/HUD
images are byte-identical in the playground and DLSS5 repositories.

## Earlier GPU atmosphere images (undersized interior)

The following record is historical. Its incorrect interior scale allowed
empty-world visibility behind some cafe panes. The images referenced by the
README have all been replaced by the corrected assembly above.

September 6, 2026 (local): captured from the integrated release build containing
the GPU clear-sky atmosphere, stable fixed-camera interpolation and shared
HUD/diagnostic registration, normal capture shutdown and explicit NR resource
cleanup before runtime unload. Source bases are Solarik `cc1f9e6`, playground
`460be24` and DLSS5 `4f37244`, plus this iteration's changes.
Executable SHA-256:
`b5004c45dea2b72df878318d95b88f13b9144838c55f3d53693725259ef21235`.
Windows/Vulkan on RTX 4090, NVIDIA Studio driver 616.56; GPU timestamp
instrumentation disabled for these images. Build with
`cargo build --release --locked` in the playground. After capture, the staged
whitespace check removed exactly one trailing newline from `model.wgsl`;
all other source bytes match the build manifest. The preserved captured shader
and final artifact audit verify that sole formatting difference.

The original Bistro exterior and matching cafe interior are loaded. Both
render modes use Solarik and native-resolution DLSS Ray Reconstruction.
The atmosphere uses the High preset quality, density scales 1, Mie g 0.8,
2,000 m aerial distance and IBL scale 1. Camera exposure follows the authored
12.5/5.0 EV100 day/night curve. Local artistic fog and extra grading are off;
`SCENE_FX=blur` retains the usual camera effects. Camera and lighting stay
fixed during warmup. Historical `night` filenames below contain dusk views.

| README asset | Resolution | Pose | Fixed warmup frames |
| --- | --- | --- | ---: |
| bistro-day.png | 3840×2160 | Timeline 750 / 1800 | 512 |
| bistro-night.png (dusk) | 3840×2160 | Timeline 1520 / 1800 | 1024 |
| bistro-temporal-response.png | 1280×720 | Independent −14° sun control | 2048 |

All three images have Neural Rendering off. The temporal-response illustration
is a fresh static night control, not a new measurement of sunset response.

Independent warmups hold camera and lighting fixed. The metric is regional
mean sRGB-decoded display-linear luminance with weights (0.2126, 0.7152, 0.0722).
Absolute changes use the same 0–1 display-linear scale.

| View / region | Rectangle x, y, width, height | Warmup comparison | Luminance change | Absolute change |
| --- | --- | --- | ---: | ---: |
| Day pavement | 430, 1930, 550, 190 | 128 to 512 | +1.36% | +0.000181 |
| Day facade | 2250, 100, 900, 450 | 128 to 512 | +0.07% | +0.000022 |
| Day wall | 1110, 1180, 500, 330 | 128 to 512 | +5.40% | +0.000425 |
| Dusk pavement | 100, 1720, 760, 250 | 512 to 1024 | +1.31% | +0.000132 |
| Dusk facade | 130, 150, 670, 450 | 512 to 1024 | +2.16% | +0.001998 |
| Dusk wall | 2150, 1660, 510, 280 | 512 to 1024 | +3.76% | +0.000111 |
| Night pavement | 40, 560, 230, 120 | 512 to 1024 | +1.59% | +0.000831 |
| Night facade | 40, 40, 200, 140 | 512 to 1024 | -0.64% | -0.001730 |
| Night cafe wall | 900, 500, 220, 100 | 512 to 1024 | -8.35% | -0.000450 |
| Night pavement | 40, 560, 230, 120 | 1024 to 2048 | -2.42% | -0.001282 |
| Night facade | 40, 40, 200, 140 | 1024 to 2048 | +1.57% | +0.004248 |
| Night cafe wall | 900, 500, 220, 100 | 1024 to 2048 | +4.43% | +0.000219 |

The day wall still changes +5.40% (+0.000425). The night wall changes −8.35%
from 512 to 1,024 frames, then +4.43% (+0.000219) from 1,024 to 2,048.
These independent warmups do not establish full convergence. All final images
were inspected: street geometry, the Vespa, balconies, lamps, glass and shrubs
remain present; dark shade, foliage softness and spatial variation remain.

The complete set of 16 serial captures exited normally, produced fresh PNGs
and had no shader/validation errors or Windows NVIDIA driver events. Both NR
ON logs confirm active evaluation and explicit feature release before shutdown.

From the playground repository in Git Bash:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=750 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=512 SCENE_FX=blur SCENE_DIAG=0 NR_OUT=../artifacts/bevy-sponza/sky-readme-day-512 python tools/launch.py capture off
SCENE=bistro NR_RES=3840x2160 NR_START=1520 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=1024 SCENE_FX=blur SCENE_DIAG=0 NR_OUT=../artifacts/bevy-sponza/sky-readme-dusk-1024 python tools/launch.py capture off
SCENE=bistro NR_RES=1280x720 NR_START=0 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=2048 SCENE_FX=blur SCENE_DIAG=0 SCENE_SUN=-14,-14,90,90 SCENE_CAM_FROM=4,1.6,12 SCENE_LOOK_FROM=-3,3,-3 SCENE_FOV=50 NR_OUT=../artifacts/bevy-sponza/sky-temporal-2048 python tools/launch.py capture off
```

The local `artifacts/bevy-sponza/sky-build.json` identifies every source file
and the executable. `sky-settling.json` records regional measurements;
`sky-readme-inventory.json` verifies all 12 referenced image files across the
three repositories, including details blocks and identical duplicates.
See [atmosphere validation](atmosphere.md) for numerical GPU tests, shared
radiance units, reference rendering and repeated 4K timing.

## Earlier GI foliage endpoint images

September 6, 2026: all three README images are refreshed with Solarik
`e76ec36` plus the GI endpoint change in this commit, and playground `90f532a`.
Build: `cargo build --release --locked` in the separate rig. All views include
the original Bistro exterior/interior, preset lights and exposure, fog/blur,
RR at native resolution, no extra grading and no Neural Rendering.

| README asset | Resolution | Pose | Fixed warmup frames |
| --- | --- | --- | ---: |
| bistro-day.png | 3840x2160 | Timeline 750 / 1800 | 512 |
| bistro-night.png (dusk) | 3840x2160 | Timeline 1520 / 1800 | 1024 |
| bistro-temporal-response.png | 1280x720 | Independent -14° sun night control | 2048 |

Independent warmups keep the same camera and lighting. Regional means use
sRGB-decoded display-linear RGB with luminance weights (0.2126, 0.7152, 0.0722).

| View / region | Rectangle x, y, width, height | Warmup comparison | Luminance change |
| --- | --- | --- | ---: |
| Day pavement | 430, 1930, 550, 190 | 128 to 512 | -2.53% |
| Day facade | 2250, 100, 900, 450 | 128 to 512 | -0.37% |
| Day wall | 1110, 1180, 500, 330 | 128 to 512 | +1.25% |
| Dusk pavement | 100, 1720, 760, 250 | 512 to 1024 | -2.79% |
| Dusk facade | 130, 150, 670, 450 | 512 to 1024 | +0.62% |
| Dusk wall | 2150, 1660, 510, 280 | 512 to 1024 | -0.58% |
| Night pavement | 40, 560, 230, 120 | 1024 to 2048 | +2.32% |
| Night facade | 40, 40, 200, 140 | 1024 to 2048 | +1.16% |
| Night wall | 900, 500, 220, 100 | 1024 to 2048 | -5.84% |

The night run was extended after a -4.90% wall shift from 512 to 1024 frames.
The later wall change is -0.000376 in absolute display-linear luminance.
Longer warmup has not established full convergence. Final images were inspected
at full resolution: the Vespa, shrubs, balconies, cafe and street geometry
remain present; dark regions, spatial variation and foliage softness remain.
The night image is a fresh static control, not a new sunset-response measurement.

From the playground repository in Git Bash:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=750 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=512 SCENE_FX=fog,blur NR_OUT=../artifacts/bevy-sponza/fe-readme-day-512 python tools/launch.py capture off
SCENE=bistro NR_RES=3840x2160 NR_START=1520 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=1024 SCENE_FX=fog,blur NR_OUT=../artifacts/bevy-sponza/fe-readme-dusk-1024 python tools/launch.py capture off
SCENE=bistro NR_RES=1280x720 NR_START=0 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=2048 SCENE_FX=fog,blur SCENE_SUN=-14,-14,90,90 SCENE_CAM_FROM=4,1.6,12 SCENE_LOOK_FROM=-3,3,-3 SCENE_FOV=50 NR_OUT=../artifacts/bevy-sponza/fe-temporal-2048 python tools/launch.py capture off
```

Local `artifacts/bevy-sponza/fe_capture.py`, `fe_metrics.py`, `fe-metrics.json`
and capture logs record the commands and measurements. The local
`fe-readme-inventory.json` verifies every one of the 12 README image files
across the three active repositories against its new capture/composite hash.
See [GI endpoint validation](foliage.md#restir-gi-endpoints) for the separate
analytical GPU tests and fixed/moving renderer checks.

## Earlier foliage cache propagation images

September 6, 2026: refreshed with Solarik `37794eb` plus the cache-propagation
change in this commit, and playground `90f532a`. Both images use native
3840x2160 with RR, the matching Bistro interior, preset exposure and lighting,
fog/blur, no additional grading and no Neural Rendering. The day image uses
timeline frame 750 and 512 warmup frames; dusk uses frame 1520 and 1,024 warmup
frames. The timeline remains 1,800 frames.

Independent fixed-camera and fixed-lighting warmups give these display-linear
mean-luminance changes:

| View / region | Rectangle x, y, width, height | 128 to 512 | 512 to 1024 |
| --- | --- | --- | --- |
| Cafe pavement | 430, 1930, 550, 190 | -2.48% | — |
| Cafe upper facade | 2250, 100, 900, 450 | -1.13% | — |
| Cafe wall | 1110, 1180, 500, 330 | +1.77% | — |
| Dusk pavement | 100, 1720, 760, 250 | +6.62% | +0.59% |
| Dusk facade | 130, 150, 670, 450 | +1.27% | +0.63% |
| Dusk cafe wall | 2150, 1660, 510, 280 | +7.92% | +2.80% |

The initial dusk comparison prompted the longer warmup. All five images were
inspected at full resolution; the Vespa, shrubs, balconies and cafe geometry
remain present. Spatial variation, dark regions and foliage softness remain.
These regional mean comparisons bound settling in these views; they do not
establish full convergence or reference-pathtracer accuracy.

From the separate rig after `cargo build --release --locked`:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=750 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=512 \
  SCENE_FX=fog,blur NR_OUT=../artifacts/bevy-sponza/fc-readme-day-512 ./capture.sh off
SCENE=bistro NR_RES=3840x2160 NR_START=1520 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=1024 \
  SCENE_FX=fog,blur NR_OUT=../artifacts/bevy-sponza/fc-readme-dusk-1024 ./capture.sh off
```

Use warmups 128/512 for the earlier comparisons. Logs and JSON measurements
remain with the local `fc-*` artifacts. See [foliage cache validation](foliage.md#world-cache-propagation)
for numerical GPU tests, the separate camera-motion comparison and the
remaining ReSTIR receiver work.

## Earlier reflected foliage images

September 6, 2026: both 4K images are refreshed with the reflected-foliage
handoff built on Solarik `fbcabb3` and playground `90f532a`. The same
frame 750 cafe and frame 1520 dusk poses, 1800-frame timeline, scene exposure,
lights and fog/blur settings are retained. RR is on at native resolution;
Neural Rendering and additional grading are off. Both published images use
512 warmup frames. Reproduce the earlier commands below with output directories
`fg-readme-day-512` and `fg-readme-dusk-512`.

Each scene was captured after independent 128/512-frame warmups at fixed
camera and lighting. Display-linear luminance changes (512 relative to 128):

| View / region | Rectangle x, y, width, height | Change |
| --- | --- | --- |
| Cafe pavement | 430, 1930, 550, 190 | +1.25% |
| Cafe upper facade | 2250, 100, 900, 450 | +2.21% |
| Cafe wall | 1110, 1180, 500, 330 | +1.89% |
| Dusk pavement | 100, 1720, 760, 250 | +3.98% |
| Dusk facade | 130, 150, 670, 450 | -0.20% |
| Dusk cafe wall | 2150, 1660, 510, 280 | -0.41% |

Images were inspected at full resolution: the Vespa, shrubs, cafe geometry
and broad lighting remain present. Residual softness and settling remain;
the dusk pavement comparison does not establish a noise-free plateau.
See [foliage validation](foliage.md#reflected-foliage-validation) for the
separate analytic, reference and moving-camera evidence.

## Earlier renderer visibility fix

September 6, 2026: the README now shows the cafe at timeline frame 750 and
the terrace at dusk at frame 1520, both on the 1800-frame Bistro timeline.
These use the renderer fork in this commit, based on Solarik `b7dd057`, and
the playground's root Cargo patch (based on `6b91c83`). Both include the
matching original Bistro interior. No diagnostic rendering switches are active.

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=750 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=512 \
  SCENE_FX=fog,blur NR_OUT=../artifacts/bevy-sponza/visibility-day ./capture.sh off
SCENE=bistro NR_RES=3840x2160 NR_START=1520 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=512 \
  SCENE_FX=fog,blur NR_OUT=../artifacts/bevy-sponza/visibility-dusk ./capture.sh off
```

Images were inspected at full resolution. Independent 128/512-frame warmups
retain the Vespa and shrubs. The dusk pavement/facade/cafe-wall display-linear
mean luminance differences are -1.29%, +0.40% and -0.99%; the earlier long-settling
capture findings below describe older code. This bounded comparison does not
establish a noise-free plateau. See [the renderer fork record](renderer-fork.md)
for the deterministic regression and runtime evidence.

DLSS Ray Reconstruction is on at native resolution; DLSS 5 Neural Rendering
and extra grading are off. Lens, exposure, lights and effects use the scene
preset. The historical `bistro-night.png` filename now contains the dusk image.

## Earlier aperture-light audit

September 6, 2026: native 3840×2160 Bistro captures using renderer
`bb4104d` and local rig `6eb7fcf` plus the generated lantern fixture.
The day image uses 2,048 warm-up frames; night uses 4,096. The earlier
[convergence audit](pathtracer-convergence.md) records the night comparison
between 240 and 2,048 frames and its unresolved visual limitations.

Reproduce using the separate local `bevy-sponza` rig after a release build:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=100 NR_CAPTURE=1 NR_WARMUP=2048 \
  NR_OUT=../artifacts/bevy-sponza/aperture-readme-day ./capture.sh off
SCENE=bistro SCENE_DIAG=0 NR_RES=3840x2160 NR_START=400 NR_CAPTURE=1 NR_WARMUP=4096 \
  NR_OUT=../artifacts/bevy-sponza/aperture-readme-night ./capture.sh off
```

The scene preset enables Solarik and DLSS Ray Reconstruction at native
resolution. Exposure, lights and effects use preset defaults; extra grading
is disabled. Neural Rendering is off. These are presentation screenshots,
not linear photometry or convergence measurements. Inspect the saved images
after shader/presentation changes before replacing the README assets.

The aperture-light audit refresh uses renderer `bb4104d`. The day image is
recaptured at frame 100 with 2,048 warm-up frames (previously 240), the same
fixed camera, EV100 12 and preset lights. Display-linear whole-image RMSE
is 0.007505; means are 0.268100 and 0.266480. Visual inspection shows stable
large-scale lighting with residual soft foliage. This pair bounds change
between these warm-ups; it does not prove a noise-free GI plateau.

The night refresh keeps frame 400, EV100 5, preset lights and the logged
camera pose from the earlier audit. Against the 2,048-frame image, the
4,096-frame image has display-linear whole-image RMSE 0.014414; means are
0.124400 and 0.128171. Inspection shows the same broad lighting and remaining
shadow blotching and soft foliage. Longer warm-up has not eliminated these
artifacts, and this pair does not establish a settled GI plateau. Diagnostic
logging (`SCENE_DIAG=0`) records timings without changing presentation.

The README retains the Bistro asset attribution and license. The first
frame-0 trial was rejected for presentation because a tree hides the street.
