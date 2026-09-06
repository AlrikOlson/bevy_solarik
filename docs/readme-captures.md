# README capture record

## Reflected foliage — current images

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
SCENE=bistro SCENE_DIAG=1 NR_RES=3840x2160 NR_START=400 NR_CAPTURE=1 NR_WARMUP=4096 \
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
logging (`SCENE_DIAG=1`) records timings without changing presentation.

The README retains the Bistro asset attribution and license. The first
frame-0 trial was rejected for presentation because a tree hides the street.
