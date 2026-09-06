# Thin glass transport

Thin panes require both `AlphaMode::Blend` and `specular_transmission = 1.0`.
The corresponding glTF authoring is `KHR_materials_transmission.transmissionFactor = 1`
plus `alphaMode: BLEND`. Alpha mode alone never identifies glass.
Ordinary Blend, Premultiplied, Add and Multiply currently retain Bevy raster
shading/compositing and are excluded from the TLAS (including ray shadows,
reflections and pathtracer visibility). Fractional or invalid specular transmission
also uses this fallback. Diffuse Blend ray transport is a separate follow-up.
Raster-only cameras retain Bevy's behavior.

The unreleased pathtracer treats explicitly authored glass as smooth thin panes.
Camera and subsequent BSDF rays can reflect from the pane or transmit through
its base-color tint. Realtime Solarik also resolves panes inside glossy reflection
paths. Camera-visible panes now use a primary compositor after opaque and sky
rendering, before the remaining raster transparency.

## Model and implementation

`scene/thin_glass.wgsl` uses Schlick Fresnel with Bevy's normal-incidence
`F0 = clamp(0.16 * reflectance², 0, 1)`. It sums internal reflections between
the pane's two interfaces as `R = 2F / (1 + F)`. At normal incidence the
default reflectance control of 0.5 gives F0=0.04 and R=1/13.

This follows the thin, parallel-interface construction in
[PBRT's thin dielectric BSDF](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF),
with Schlick Fresnel replacing its exact dielectric Fresnel. Transmission exits
along the incident direction: this is a zero-thickness pane, with no lateral
shift, volumetric absorption, rough refraction, dispersion, or configurable IOR.

Alpha is coverage `a`, clamped after multiplying the base-color factor and
texture alpha. Linear base color `c` is clamped to [0,1]. The reflected and
transmitted energies are:

- Reflection: `a R`.
- Transmission: `(1-a) + a (1-R) c`.

Sampling reflection with probability `a R` gives unit reflection throughput.
Transmission divides its energy by `1-a R` exactly once. Thus alpha zero is
an untinted hole and a clear white pane conserves energy. Glass emission is
coverage weighted.

`trace_glass_ray` confirms blended candidates with nonzero alpha, while
retaining the texture cutoff for masked geometry. Existing `trace_ray` callers
keep their prior behavior. No material-buffer layout changed. The pathtracer
handles glass before the ordinary BRDF, using a two-sided geometric normal and
offsetting the outgoing ray to the correct side. A delta reflection stops
competing with the previous direct-light sample; straight transmission retains
that competition. A chain is capped at 32 consecutive glass interactions.
Reaching that cap discards remaining energy rather than treating glass as diffuse.

## Realtime glossy paths

The glossy path uses the same glass-aware traversal and thin-pane sampler.
Glass does not consume any of its three opaque bounces; each consecutive chain
is capped at 32 glass interactions. A delta reflection owns subsequent emission,
while straight transmission retains the previous light-sampling competition,
including the primary ReSTIR DI ownership rule. Black transmitted throughput
terminates immediately. Shadow and diffuse GI rays remain unattenuated.

DLSS primary-surface replacement stops when a glossy path encounters glass.
This retains the opaque primary guides instead of replacing them with a surface
chosen by a stochastic glass branch. It is a conservative fallback, not a
complete guide model for transparent reflections.

The ignored `glossy_glass_gpu` test executes the production glossy transport
and shared sampler against deterministic synthetic scene I/O. Ten cases run
with and without DLSS guide code: tint, clear and zero-alpha panes, reflection,
four panes retaining the opaque budget, chain truncation, black transmission,
transmission/reflection emission ownership, and an opaque PSR control. It checks
radiance within 1e-5 and counts replacement calls. The original shader failed
the tint case with [1,1,1] instead of [0.2,0.5,0.8]. This isolates transport;
it does not test actual acceleration structures, texture alpha, or convergence.

```text
cargo test --test glossy_glass -- --ignored --nocapture
```

## Primary camera panes

The camera pass splits reflection and transmission deterministically using the
same thin-pane energy weights as the stochastic sampler. It walks up to 32 panes
front to back, traces each reflected direction with the existing glossy path,
and applies the accumulated transmission to the already shaded opaque/sky pixel.
An exact 32-pane chain may reach its background; a 33rd pane truncates remaining
energy. Near-plane ray construction works for perspective and orthographic views.

Only blended instances actually bound in the current TLAS lose their raster
draws, and only when the pipelines and view resources are ready. The phase uses
main-world entity IDs (its render entity may be a placeholder), so suppression
resolves that identity explicitly. Items are removed before batching and restored
before the next visibility/material queue update, preserving reload fallback and
other cameras. Alpha testing disabled keeps the raster path.

Depth and ordinary motion remain the opaque background. Pane pixels restore
background albedo, normal/roughness and specular motion after glossy PSR, keeping
all guides on the same deterministic surface. This is a conservative ownership
policy, not an exact reconstruction model for two independently moving layers.

The ignored `primary_glass_gpu` test executes the production compositor against
synthetic scene I/O: 15 cases cover tint, clear/zero-alpha panes, opaque occlusion,
hidden black/green walls and raster ownership controls,
two panes, black transmission, perfect reflection, emission and both sides of
the 32-pane limit. The existing thin-glass and glossy GPU regressions also pass.

A complete 640x320 realtime capture at the analytic camera, RR off, matches the
tinted/clear/hole radiance oracle within 0.0031. The original integration failed
with black panes because raster draws still covered the traced result. The
opaque masked green control exposed a Bevy 0.19.1 render-order bug: its
forward alpha-mask draw overwrote Solarik's deferred result. Solarik now resolves
after the opaque/sky pass and before primary compositing and transparency.
The control measures 0.9969 for expected radiance 1.0, versus a clipped value
of at least 1.2 before the fix. Every pane now uses the physical oracle:

```text
python tests/glass_scene.py --check ../artifacts/bevy-sponza/mask_ownership_analytic/seq_0000.png
```

## Opaque interiors behind glass

A ray-visible wall can be absent from the raster background when its material
culls backfaces. Previously the compositor found that wall but returned the
raster sky or more distant building. It now shades the actual opaque hit using
the same bounded four-path estimator as primary foliage. The query stops before
raster depth, so a visible opaque background still reuses its existing shading.
Pixels without an intervening pane retain their raster ownership. Authored
double-sided flags and normals are unchanged.

The `--hidden-room` variant of `tests/glass_scene.py` encloses a room with a
glass front and five outward-facing, single-sided walls. A green emissive back
wall lies in front of a bright white emitter; a red emitter tests reflection.
At normal incidence the expected radiance is [1/13, 6/13, 0]. Before the fix,
the room measured [0.9969, 0.9221, 0.9218], revealing the white emitter through
the wall. Afterward it measures [0.0778, 0.4637, 0.00002], maximum error 0.00218.

Generate with `--generate <asset path> --hidden-room`, use the analytic camera
and exposure below with `SCENE_SOLARI=1 SCENE_RR=0 NR_WARMUP=240`, then verify:

```text
python tests/glass_scene.py --hidden-room --check ../artifacts/bevy-sponza/glass_closed_room_fixed/seq_0000.png
```

Missing interior geometry cannot be reconstructed. A single-sided backface
blocks transmission but retains its authored shading orientation. Hidden
surfaces use bounded paths without temporal reservoirs; raster depth/motion
and DLSS background guides remain unchanged, so moving glass can still ghost.

## Validation, 2026-09-06

Windows, Vulkan, RTX 4090, NVIDIA 591.86, Bevy 0.19.1.

`tests/thin_glass.rs` executes the production WGSL on the GPU. The test first
failed because the sampler did not exist, then passed with the implementation.
It checks 24 configurations (12 cases on both sides), each with 65,536 stratified
branch samples: white-pane energy conservation, RGB tint, alpha endpoints and
clamping, black tint, grazing incidence, zero and saturated reflectance, oblique
incidence, outgoing directions, and origin offsets.

Run the ignored GPU test explicitly, with scene captures and other builds stopped:

```text
cargo test --test thin_glass -- --ignored --nocapture
```

`tests/glass_scene.py` also builds a complete ray-query fixture for the separate
local bevy-sponza rig. White emission behind the panes tests transmission; red
emission behind the camera tests reflection. It contains tinted and clear glass,
a zero-alpha pane, a masked hole and a solid masked pane. The checker inverts
sRGB and EV100=0 exposure and compares image patches against emitter/Fresnel
expectations. The test would fail if blended geometry were still skipped.

| Pane | Expected RGB radiance | Measured RGB radiance | Maximum absolute error |
|---|---|---|---|
| Tinted | 0.3084, 0.4611, 0.6916 | 0.3128, 0.4626, 0.6931 | 0.00441 |
| Clear | 1.0000, 0.9231, 0.9231 | 1.0118, 0.9242, 0.9240 | 0.01178 |
| Zero alpha | 1, 1, 1 | 1.0117, 0.9972, 0.9969 | 0.01167 |
| Mask hole | 1, 1, 1 | 1.0113, 0.9994, 0.9969 | 0.01133 |
| Mask solid | 0, 1, 0 | 0.0159, 0.9969, 0.0003 | 0.01585 |

The 0.04 absolute tolerance includes finite sampling, 8-bit PNG quantization,
patch-angle variation and ordinary specular shading of the emissive surfaces.
The standalone GPU test isolates the pane model more tightly (3e-5 energy error).

Local fixture reproduction, starting from the Solarik checkout with a sibling
rig and its Bistro preset available:

```bash
python tests/glass_scene.py --generate ../bevy-sponza/assets/scenes/bistro/glass_probe.gltf
cd ../bevy-sponza
cargo build --release
SCENE=bistro SCENE_GLTF=scenes/bistro/glass_probe.gltf \
SCENE_PATHTRACE=1 SCENE_LINEAR=1 SCENE_CAM_FROM=0,0,3 SCENE_LOOK_FROM=0,0,0 \
SCENE_FOV=60 SCENE_DAY_EV=0 SCENE_NIGHT_EV=0 SCENE_SUN_SCALE=0 \
SCENE_MOON_SCALE=0 SCENE_IBL_SCALE=0 NR_WARMUP=2048 NR_CAPTURE=1 \
NR_START=400 NR_RES=640x320 NR_OUT=../artifacts/bevy-sponza/glass_analytic \
./capture.sh off
python ../bevy_solarik/tests/glass_scene.py --check ../artifacts/bevy-sponza/glass_analytic/seq_0000.png
```

Bistro integration captures at frames 140 and 400 also completed without shader
errors, at 1920×1080 with 2048 warmup frames and grade-only postprocessing.
Fixed cameras reproduce the preset poses:

| Frame | Camera | Look target | FOV |
|---|---|---|---|
| 140 | -10.5, 1.7, -1 | 0, 3.5, 0 | 50 |
| 400 | 5.8892128, 4.8116618, 12 | -3, 3.4723032, -3 | 46.2215743 |

Set `SCENE_PATHTRACE=1 SCENE_FX=grade NR_WARMUP=2048 NR_CAPTURE=1 NR_RES=1920x1080`,
then the row's `NR_START`, `SCENE_CAM_FROM`, `SCENE_LOOK_FROM`, and `SCENE_FOV`.
Do not carry the analytic fixture's light, exposure or model overrides into Bistro.

Images and logs stay outside the source repository, in the local organizer's
`artifacts/bevy-sponza/glass_fixed_140`, `glass_fixed_400`, and `glass_analytic`.
Bistro windows and the Vespa view were inspected, but remain visibly noisy.
Neither the 512-warmup moving-pose captures nor the 2048-warmup fixed-pose captures
establish convergence or quantitative reflection quality. The analytic scene is
the numerical acceptance evidence. Saved sample counts and Bistro convergence
need a separate audit.

Realtime Bistro captures at the same fixed poses also compile and run the full
ray-query shader with DLSS RR enabled: 1280x720, 128 warmup frames, grade-only,
`SCENE_SOLARI=1 SCENE_PATHTRACE=0 SCENE_RR=1`. Images and logs are in the
local organizer's `artifacts/bevy-sponza/glossy_glass_140` and
`glossy_glass_400`. Both captures were inspected and have no shader errors.
They validate the earlier secondary-path integration, not glass photometry or
temporal convergence.

Primary-compositor Bistro captures at frames 140 and 400 use the same fixed
poses, 1280x720, 128 warmup frames and RR. Images and logs are in
`primary_glass_140` and `primary_glass_400`. They were compared visually with
`glass_fixed_140` and `glass_fixed_400`: windows, lamps and the Vespa render,
but the noisy reference and differing resolution do not establish convergence
or numerical equality. The analytic fixture is the quantitative glass evidence.

## Limits

- Shadow rays still pass through glass without Fresnel loss or tint. This makes
  next-event lighting through panes approximate; do not use it as a complete
  dielectric photometric reference. Emissive light sampling also retains its
  existing coverage treatment.
- Analytic point, spot and directional lights cannot be hit by delta reflected
  rays. Reflections see emissive geometry and sky; analytic-light reflections
  need a separate sampling strategy.
- Every Blend material gets the thin-pane interpretation, including blended
  foliage or curtains. Alpha alone cannot identify physically authored glass.
- Roughness and metallic controls do not change this smooth dielectric model.
  A mesh with two modeled pane faces is treated as two thin panes.
- `SolarikAlphaTesting(false)` retains the existing opaque-geometry fallback.
- Primary transmission reuses opaque/sky shading. Non-raytraced transparent
  objects are still drawn later and cannot interleave correctly with traced panes.
- A single background DLSS guide cannot describe both reflection and transmission;
  moving reflections may ghost. Layer-aware reconstruction remains future work.
