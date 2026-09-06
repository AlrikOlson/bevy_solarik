# Thin diffuse foliage

The reference pathtracer supports scalar `StandardMaterial.diffuse_transmission`
on explicitly double-sided `AlphaMode::Mask` materials. Default zero, other alpha
modes and single-sided materials keep opaque diffuse behavior. Nonfinite values
become zero and finite values clamp to [0,1]. A unorm16 coefficient uses the
unused high half of Material.flags without changing its 64-byte GPU layout.

The diffuse lobe splits its original energy: reflection receives (1-t), transmission
receives t. Both sides use base-color tint and the existing Fresnel layering;
metallic energy stays specular. This is a thin-sheet approximation, without
thickness, subsurface scattering, a transmission texture or colored shadows.
The sampler and light-sample MIS share one solid-angle PDF. Continuation and
direct-light visibility rays start on the outgoing side.

Realtime uses a dedicated primary-foliage pass after opaque shading and before
glass compositing. A camera ray must match the raster depth (2 mm or 0.001% of
distance) and normal before replacement; only authored nonzero-transmission
foliage is shaded. It traces four independent paths with at most four scattering
events each, using two-sided NEE and solid-angle MIS for emissive meshes.
DLSS receives the original primary leaf depth, normal, albedo and motion.

Glossy rays, including reflections launched by primary glass, hand authored
transmissive leaf hits to the same two-sided surface estimator. The incoming
path retains its emissive MIS weight and mirror surface replacement before
the handoff. Four scattering vertices follow the leaf hit, with at most 32
thin-pane events between vertices. Primary glass also uses this estimator for
foliage hidden behind an opaque raster-depth boundary. Thin glass and diffuse
coverage keep their existing transport within these surface paths.

World-cache GI propagation combines separately cached front and back
irradiance at leaf endpoints (see below). Existing primary ReSTIR receivers
and ReSTIR GI endpoint samples remain one-sided. Leaf surface-path handoffs
still bypass those caches; zero-transmission materials keep the existing
glossy estimator. Reused GI samples taken directly at a leaf still lack its
transmission. This change does not reduce primary-foliage query
cost. Bounce truncation can lose indirect light, raw output is noisy, and
reflections add GPU work. Coplanar,
unregistered raster-only surfaces can remain ambiguous without material IDs.

Explicitly select `OpaqueRendererMethod::Deferred` when authoring foliage for
Solarik: Bevy's Auto chooses forward for transmission. Pipeline warmup and
unavailable TLAS resources retain the existing opaque shading.

## Validation

`cargo test --test foliage -- --ignored` runs 65,536 production diffuse BSDF
samples for each t=0,0.25,0.5,1, checking hemisphere frequencies, finite endpoints,
sample/MIS PDF agreement and energy. Specular-only functions are stubbed in this
isolated numerical fixture; the full shader is exercised by the rig capture.

From the organizer root:
```text
python bevy_solarik/tests/foliage_scene.py --generate bevy-sponza/assets/scenes/bistro/foliage_probe.gltf
```
From bevy-sponza in Git Bash after a release build:
```bash
SCENE=bistro SCENE_GLTF=scenes/bistro/foliage_probe.gltf \
SCENE_FOLIAGE='leaf zero=0,leaf half=0.5,leaf full=1' \
SCENE_SOLARI=1 SCENE_PATHTRACE=1 SCENE_RR=0 SCENE_LINEAR=1 \
SCENE_CAM_FROM=0,0,3 SCENE_LOOK_FROM=0,0,0 SCENE_FOV=60 \
SCENE_DAY_EV=0 SCENE_NIGHT_EV=0 SCENE_SUN_SCALE=0 SCENE_MOON_SCALE=0 SCENE_IBL_SCALE=0 \
NR_WARMUP=512 NR_CAPTURE=1 NR_START=400 NR_RES=640x320 \
NR_OUT=../artifacts/bevy-sponza/foliage_reference ./capture.sh off
python ../bevy_solarik/tests/foliage_scene.py --check ../artifacts/bevy-sponza/foliage_reference/seq_0000.png
```

RTX 4090/Vulkan, 512 samples: half-transmission RGB measured
(0.0896,0.2237,0.3571), analytic (0.0878,0.2194,0.3511).
Full transmission measured (0.1757,0.4387,0.7018), analytic
(0.1755,0.4389,0.7022). The zero card's blue channel is 0.0127 from
finite-scene interreflection; the checker permits 0.04 absolute error.
Realtime (same analytic scene, SCENE_PATHTRACE=0 and RR=0) measures half
transmission (0.0799,0.1993,0.3188), full (0.1758,0.4391,0.7021);
maximum absolute error is 0.0323 with only four samples per pixel.
The zero card remains exactly black on the unchanged realtime opaque path.


## Reflected foliage validation

The glossy handoff runs one bounded surface path per leaf hit. It keeps the
incoming path's emission MIS and RR surface replacement, and never writes
two-sided leaf radiance into the one-sided world cache. The existing
`shade_surface_path` wrapper still includes initial emission for primary callers.

`cargo test --test glossy_foliage -- --ignored` executes the production glossy
path, surface path, diffuse BSDF and PDF with synthetic scene I/O. Eight cases
run with RR guides disabled and enabled, using 32,768 samples each: opaque,
half/full transmission, backlit NEE, initial emission owned by the path or DI,
rough-path cache bypass, and a tinted pane before foliage. They check energy,
outgoing offsets, finite values, traversal bounds and guide ownership.
The pre-change shader fails the half-transmission test with zero energy.
Existing foliage, glossy-glass, primary-glass and surface-shadow GPU suites pass.

Generate the reflected fixture with `foliage_scene.py --generate <path> --reflected`.
For motion, copy `tests/foliage_reflection_scene.toml` to the separate rig's
`assets/scenes/foliage-reflection/scene.toml` and generate its `scene.gltf` beside it.
After a release build, use these Git Bash settings in the rig:

```bash
SCENE=foliage-reflection SCENE_LINEAR=1 SCENE_RR=1 \
SCENE_SUN_SCALE=0 SCENE_MOON_SCALE=0 NR_RES=640x320 \
NR_START=0 NR_TIMELINE=60 NR_CAPTURE=60 NR_WARMUP=128 \
NR_OUT=../artifacts/bevy-sponza/fg-motion ./capture.sh off
python ../bevy_solarik/tests/foliage_scene.py \
  --check ../artifacts/bevy-sponza/fg-motion --motion
```

A fixed view adds `SCENE_CAM_FROM=0,0,3 SCENE_LOOK_FROM=0,0,0 SCENE_FOV=60`
and uses `NR_CAPTURE=1`. Capture raw realtime with `SCENE_RR=0`, and the reference
with `SCENE_PATHTRACE=1 SCENE_RR=0 NR_WARMUP=2048`. Check raw/RR against the
reference with `--check <PNG> --reflected --reference <reference PNG>`.
The large mirror illuminates both hemispheres, so this image comparison uses
the pathtracer; the GPU probe separately supplies the one-hemisphere analytic test.

September 6, 2026, RTX 4090/Vulkan, 640×320, EV100 0, no post effects or Neural
Rendering: the original glossy path renders all three reflected cards black.
The new t=0.5 card measures RGB (0.18349, 0.44179, 0.69914) raw versus
(0.18336, 0.44076, 0.69894) in the 2,048-sample reference. The t=1 card measures
(0.18391, 0.44309, 0.70035) versus (0.18677, 0.44862, 0.71045).
Maximum channel error across both leaf patches is 0.01010 raw and 0.00512 with RR.
This is a small scene comparison, not a general convergence claim.

All 60 RR motion frames retain lit leaf patches. Their largest adjacent-frame
patch-channel change is 0.00523, while the opaque control remains below 0.00016.
The glossy-pass GPU median before screenshot readback rises from 0.054 ms
(range 0.044–0.065, 23 samples) to 0.077 ms (0.067–0.104, 16 samples), excluding
the first three diagnostic measurements. This tiny fixture does not predict
Bistro cost; the primary-foliage pass is unchanged.

The t=0 control deliberately retains its old black realtime result even though
the reference lights it. Opaque secondary diffuse transport is separate backlog.
RR also shows broad arcs in this fixture's flat mirror background, including
an all-zero-transmission control. The patch metrics do not establish whole-image
quality or freedom from temporal artifacts.

## World-cache propagation

Cache GI rays that hit authored foliage now consume
`(1-t) * E(front) + t * E(back)` before applying the leaf's base color.
Each hemisphere retains its own position/normal key, geometry, visibility,
irradiance samples and history. The material coefficient never enters a cache
key or a stored hemisphere field. Existing cache-side ray offsets place direct
and indirect queries on that hemisphere's side of the leaf; connection
attenuation remains outside the cached irradiance.

This preserves the cache's existing Lambertian approximation. It does not add
angular Fresnel layering, directional radiance, emission handling or a new
sampling PDF. It only redistributes the existing diffuse energy between two
hemispheres. Partial transmission makes two cache queries; t=0 and t=1 make
one. Zero transmission preserves the original query and random sequence.
Both queried entries inherit the caller's lifetime with atomic-max refresh.
Each entry keeps the existing update budget and frame-limited history blend.

`cargo test --test cache_foliage -- --ignored --test-threads=1` runs two GPU
tests. The production `sample_gi` test covers 28 combinations of coefficients
0/0.25/0.5/1, reversed normals, clear/tinted/black connections, sky escape,
distance bounds, throttling and dispatch bounds. It checks endpoint exclusion,
ray offsets and inherited lifetime. Twenty further cases execute the production
cache hash, allocation and query code with deterministic LOD and no position
jitter, checking independent opposite-side fields, lifetime refresh and opaque
energy/RNG preservation. Real scene traversal and jitter are exercised by rig
captures. The old shader fails at t=0.25: red irradiance contribution 0.4 versus
the analytic 0.8 for the fixture's front/back fields.

This is the cache-propagation portion of foliage integration. Primary ReSTIR
material payloads, two-sided receiver targets/PDFs, GI endpoint reuse and removal
of the dedicated primary-foliage pass remain follow-up work.

### Cache-propagation capture checks

September 6, 2026, RTX 4090/Vulkan. Compare the old cache at Solarik
`37794eb` with this change, using the same authored t=0.5 leaves, RR,
frame-0 pose, 1280x720 resolution, grade-only effects and 128 warmup frames.
The initial primary and reflected foliage estimators are identical in both runs.
Display-linear RGB RMSE is 0.00056 in the trunk region, 0.00323 on the wall,
and 0.00900 in the canopy. Wall/canopy mean luminance changes are -0.23%/-0.22%.
Independent 128/512 warmups of the new renderer still change those means by
+3.57%/+3.54%. This scene comparison cannot separate the small cache effect
from stochastic variation or establish reference-pathtracer accuracy.

The world-cache GPU span during fixed-camera warmup, after scene loading and
before screenshot readback, measures a median 0.444 ms before (0.410–0.502)
and 0.418 ms after (0.417–0.424), with only three samples in each run.
The overlapping ranges do not establish a speedup or a general performance
bound. The primary-foliage pass still dominates this view.

A 24-frame, 640x360 comparison at timeline frames 0–23 retains the same scene
geometry. Its before/after display-linear RGB RMSE is 0.00847. Median
adjacent-frame RMSE is 0.00448 before and 0.00441 after; maximum whole-image
mean-luminance step is 0.000126/0.000149. This is a short, slow camera-motion
probe with changing sunlight, not an isolated flicker measurement. Dark
under-canopy regions and foliage softness remain visible.

From the rig, after a release build, reproduce the fixed view with:

```bash
SCENE=bistro SCENE_FX=grade SCENE_DIAG=1 NR_RES=1280x720 \
NR_START=0 NR_TIMELINE=1800 NR_CAPTURE=1 NR_WARMUP=128 \
NR_OUT=../artifacts/bevy-sponza/fc-after ./capture.sh off
```

Use `NR_WARMUP=512` for the settling comparison; use `NR_RES=640x360`,
`NR_CAPTURE=24` and `NR_WARMUP=128` for motion. Local captures and measurement
reports use the `fc-*` prefix in the organizer's `artifacts/bevy-sponza`.
The [README capture record](readme-captures.md) documents the refreshed 4K pair.

## Bistro frame 0

The rig preset authors t=0.5 only on
`Foliage_Linde_Tree_Large_Orange_Leaves.DoubleSided` and
`Foliage_Linde_Tree_Large_Green_Leaves.DoubleSided`.
Bark, hedges, walls and other masked materials keep their original settings.
`[[render.foliage]]` entries match exact names; `SCENE_FOLIAGE` overrides them.
Material overrides apply at load, so restart after editing them.

Captures use camera (52,3,27), look (15,3,5), FOV45, 1280x720, frame0,
grade only, 128 warmup frames and RR on. The reference uses RR off and
512 samples. Set both exact-name coefficients to zero for the baseline.

The canopy has lit yellow-green backfaces and more variation, qualitatively
consistent with the reference. It is darker overall than the previous flat
one-sided result; this is not a uniform brightness boost or a convergence claim.
Opaque-control mean absolute RGB differences are 0.00106 for the trunk and
0.00161 for the wall, including stochastic/denoiser variation. Validate with:
```text
python ../bevy_solarik/tests/foliage_scene.py --check ../artifacts/bevy-sponza/foliage_bistro_timed/seq_0000.png --bistro-baseline ../artifacts/bevy-sponza/foliage_bistro_baseline/seq_0000.png
```

The dedicated pass costs 2.40–2.65 ms on RTX4090/Vulkan at this pose/resolution
(before capture readback); it is skipped when the TLAS has no authored foliage.
Reservoir integration, denoising and primary-pass performance improvements
remain follow-up work. Reflected-leaf transport now uses the bounded handoff
described above.

PDF conversion follows [PBRT light sampling](https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources):
area density becomes solid-angle density through distance squared divided by
the emitter cosine. The legacy reference/glossy emitter MIS still needs a separate
measure-consistency audit; the new foliage pass performs the conversion.

