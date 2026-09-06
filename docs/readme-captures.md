# README capture record

September 6, 2026: native 3840×2160 Bistro captures from the development
renderer including the glass-transport integration and pane range correction. The source is the
same commit as this record; the sibling rig is `86ec147`.

Reproduce using the separate local `bevy-sponza` rig after a release build:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=100 NR_CAPTURE=1 NR_WARMUP=240 \
  NR_OUT=../artifacts/bevy-sponza/integration-readme-street ./capture.sh off
SCENE=bistro NR_RES=3840x2160 NR_START=400 NR_CAPTURE=1 NR_WARMUP=240 \
  NR_OUT=../artifacts/bevy-sponza/integration-readme-night ./capture.sh off
```

The scene preset enables Solarik and DLSS Ray Reconstruction at native
resolution. Exposure, lights and effects use preset defaults; extra grading
is disabled. Neural Rendering is off. These are presentation screenshots,
not linear photometry or convergence measurements. Inspect the saved images
after shader/presentation changes before replacing the README assets.

The README retains the Bistro asset attribution and license. The first
frame-0 trial was rejected for presentation because a tree hides the street.
