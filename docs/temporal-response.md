# GI response to changing lighting

Development change beyond v0.1.0. Exposure, sky intensity, lamp intensity and
radiance clamping are unchanged.

Two history mechanisms delayed sunset response:

- GI capped confidence at 8 but never limited the selected endpoint's age.
  Bright old radiance could survive repeated spatial and temporal selection.
  Endpoints now expire after 16 temporal transfers. Spatial reuse preserves
  their age; fresh draws start at zero. Merge weight sums are local, so the
  existing 48-byte reservoir layout is unchanged.
- The world cache blended up to 32 successful cell updates, but its 40,000-cell
  budget updates only a fraction of active cells each frame. The new blend
  accounts for that probability, targeting a 16-frame time constant. This is
  an expected response, not a hard deadline for every stochastic cell. Below
  a 1/16 update probability, full replacement is already the fastest response
  possible without increasing the ray budget.

The adaptive luminance-delta blend may respond faster. Less history trades raw
stability for response. Recursive bounces and the denoiser add their own history;
this does not mean the whole image converges after 16 frames.

## Regression tests

```text
cargo test --test cache_history --test gi_history --test gi_shadow -- --ignored
```

These Vulkan tests execute production WGSL functions. GI tests preserve endpoint
age through repeated spatial merges and expire bright history at the age bound.
They retain the connection-energy assertions. The cache test checks constant
energy and less than 2% expected residual after 64 frames at update probabilities
1, 1/2, 1/4 and 1/8. The old blend retained 13.1% even at probability 1, with a
longer tail under throttling. These are filter tests, not whole-scene convergence.

## Refreshed README night control

September 6, 2026: the README illustration is refreshed with Solarik `e76ec36`
plus the GI foliage endpoint change in this commit and playground `90f532a`.
It is an independent fixed night run at the camera below, sun elevation -14
and azimuth 90 degrees, 1280x720, RR on, preset exposure and lamps, fog/blur,
no additional grading and no Neural Rendering. The selected image has 2,048
warmup frames. Between 1,024 and 2,048 frames, pavement/facade/cafe-wall means
change +2.32%/+1.16%/-5.84%; the wall's absolute luminance change is -0.000376.
Residual night variation remains. See [the full capture record](readme-captures.md).
This static illustration does not replace or re-measure the earlier sunset
experiment below.

## Earlier Bistro capture protocol

Use the assembled exterior/interior in playground revision 6b91c83. Fixed camera:
position (4,1.6,12), look at (-3,3,-3), vertical FOV 50 degrees. Sun elevation
moves from 5 to -14 degrees over 240 frames, azimuth 90, then holds for another
240 frames. Compare to an independent -14-degree run warmed for 1024 frames.
Transition warmup stays at 256 frames.

Mean error over the first 32 frames after the sun stops, relative to each
version's eight-frame fresh-night control:

| Region | Before | After |
| --- | ---: | ---: |
| Pavement | +19.8% | -1.0% |
| Shaded facade | +14.0% | -1.3% |
| Cafe wall | +3.8% | +3.7% |

The age-only intermediate did not eliminate the tail; update-cadence correction
was needed. These are single-run, noisy PNG diagnostics. Fresh-control means
changed by roughly 2–5% between builds, so the result is evidence of temporal
response rather than proof of identical steady-state bias. The normal RR
sequence was inspected throughout sunset and the eight-second night hold.
Some shadow/lighting variation remains.

Raw diagnostics use 640x360, SCENE_LINEAR=1, SCENE_RR=0, SCENE_FX=none, saving
every eighth frame. Exposure reaches the same EV100 5 floor in night comparisons.
The PNGs retain clipped bright samples, so region means are displayed-energy
diagnostics, not unclipped radiometry. A 1280x720 sequence checks normal Ray
Reconstruction presentation. Local captures, environment manifests and the
reproducible capture/measurement scripts live in Documents/Solarik-Short/night-fix.
