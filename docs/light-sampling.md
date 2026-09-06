# Light sampling measurements

Measured 2026-09-06 on an RTX 4090, Vulkan, Bevy 0.19.1, 1920×1080.
Uniform baseline: commit `a80cc2a` (rendering code from `054d84a`).
Scene: Bistro, frame 140, default preset, linear capture, NR off.

The CPU now builds a Walker alias table each frame. Its marginal PMF is
used by both next-event candidates and emissive-hit MIS. The presampled
tiles, world-cache DI, specular NEE and pathtracer share those functions.
Sky misses retain weight 1 because the sky is not in the light list.
The algorithm follows [PBRT's power sampler](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling)
and [alias construction](https://pbr-book.org/4ed/Sampling_Algorithms/The_Alias_Method).

## Mean brightness

Patch rectangles are `(x, y, width, height)`. Linear RGB is decoded from
sRGB PNGs, then Rec. 709 luminance is averaged. Exposure at frame 140 is
EV100 **11.74**. `photometry.sh` currently prints the last exposure in the
log, **11.58**, after the camera advances during flushing; that number is
not the captured pose's exposure.

| Patch | Rectangle | Uniform 8192, nits | Uniform 512, EV difference | Power 512, EV difference | Power 8192, EV difference |
|---|---|---:|---:|---:|---:|
| Near cobbles | 780,975,180,80 | 124.697 | -0.038 | +0.023 | +0.011 |
| Right cobbles | 1700,950,180,80 | 90.950 | -0.046 | +0.012 | -0.010 |
| Facade | 1220,150,120,80 | 325.507 | -0.020 | -0.009 | +0.001 |

All differences above use the uniform 8192-frame capture as reference.
The requested 0.2 EV cobble threshold passes. Uniform 512 already passes,
so this does not establish a 16× improvement. The largest clipped fraction
in these patches is below 0.1%; tiny bright outliers remain.

The numbers 512 and 8192 name the rig's warmup frames. Its first timeline
frame resets accumulation after assets settle. The existing sample-count
diagnostic at 512 warmup produced white at all 2,073,600 pixels, confirming
at least 512 accumulated samples; it saturates and does not resolve an
off-by-one difference at the capture boundary.

## Noise and runtime

Relative RMSE is `sqrt(mean((candidate_luminance-reference_luminance)^2))`
divided by the reference patch's mean. Here the reference is **power 8192**.

| Patch | Uniform 512 RMSE | Power 512 RMSE | Raw realtime uniform | Raw realtime power |
|---|---:|---:|---:|---:|
| Near cobbles | 2.112 | 1.350 | 0.381 | 0.383 |
| Right cobbles | 2.404 | 1.535 | 0.481 | 0.481 |
| Facade | 1.176 | 0.726 | 1.227 | 1.228 |

Pathtracer patch RMSE falls about 36–38%. This is one run per configuration
against a reference that still has noise, not a statistical convergence
study. Raw realtime is captured with `SCENE_RR=0`; it includes GI and
specular, so it does not isolate DI variance. There is no measurable raw
realtime quality win in these patches.

Representative SCENE_DIAG readings near capture: direct lighting about
1.6 ms uniform versus 1.1–1.3 ms weighted; presampling about 0.038 versus
0.025 ms. Diffuse indirect remains about 12–14 ms. These are individual
GPU timings, not repeated performance distributions. CPU alias construction
and cached-edge memory overhead were not separately profiled.

## Reproducing

These commands are a record of the separate, unpublished measurement rig;
the rig and scene assets are not included in this plugin repository.
In that development workspace, run from its root and save the uniform
captures before rebuilding with this change. Never run a GPU capture and
cargo build concurrently.

```bash
PHOTO_MODES=pathtracer PT_SPP=8192 PHOTO_TAG=uniform_8192_140 bevy-sponza/tools/photometry.sh bistro 140
PHOTO_MODES="pathtracer solari_raw" PT_SPP=512 PHOTO_TAG=uniform_512_140 SCENE_DIAG=1 bevy-sponza/tools/photometry.sh bistro 140
# Rebuild the rig after applying the change, then:
PHOTO_MODES="pathtracer solari_raw" PT_SPP=512 PHOTO_TAG=power_512_140 SCENE_DIAG=1 bevy-sponza/tools/photometry.sh bistro 140
PHOTO_MODES=pathtracer PT_SPP=8192 PHOTO_TAG=power_8192_140 bevy-sponza/tools/photometry.sh bistro 140
PHOTO_MODES=pathtracer PT_SPP=512 PHOTO_TAG=sample_count_140 SOLARIK_PATHTRACER_DEBUG_SAMPLE_COUNT=1 bevy-sponza/tools/photometry.sh bistro 140
```

Captures are local artifacts under `bevy-sponza/preview/photo_bistro_*`;
logs are under `bevy-sponza/run/photo_bistro_*`. PNGs and logs are ignored
by git. Use the existing `tools/photometry.py` with `--ev100 11.74` and the
rectangles above to reproduce the mean comparison.

Seven new Rust tests cover alias PMFs, dark/invalid/overflowing weights,
two-light inverse-probability expectation against uniform selection,
indexed geometry, transformed area, and point/spot power. Check, nextest
and clippy pass in both crates: 15 library tests and 25 rig tests.
