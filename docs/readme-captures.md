# README capture record

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
