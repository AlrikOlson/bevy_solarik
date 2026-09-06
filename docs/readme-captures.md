# README capture record

September 6, 2026: native 3840×2160 Bistro captures from the development
renderer including the glass-transport integration and pane range correction. The source is the
glass integration commit `cc1ca83`; the sibling rig was `86ec147` for the day
capture. The night image was refreshed during the convergence audit with
the same rendering code, a diagnostic-only shader addition and rig pose
logging. Its warm-up is now 2,048 frames. The [audit](pathtracer-convergence.md)
records the comparison with 240 frames and its unresolved visual limitations.

Reproduce using the separate local `bevy-sponza` rig after a release build:

```bash
SCENE=bistro NR_RES=3840x2160 NR_START=100 NR_CAPTURE=1 NR_WARMUP=240 \
  NR_OUT=../artifacts/bevy-sponza/integration-readme-street ./capture.sh off
SCENE=bistro NR_RES=3840x2160 NR_START=400 NR_CAPTURE=1 NR_WARMUP=2048 \
  NR_OUT=../artifacts/bevy-sponza/night-settle-2048 ./capture.sh off
```

The scene preset enables Solarik and DLSS Ray Reconstruction at native
resolution. Exposure, lights and effects use preset defaults; extra grading
is disabled. Neural Rendering is off. These are presentation screenshots,
not linear photometry or convergence measurements. Inspect the saved images
after shader/presentation changes before replacing the README assets.

The README retains the Bistro asset attribution and license. The first
frame-0 trial was rejected for presentation because a tree hides the street.
