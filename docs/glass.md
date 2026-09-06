# Thin glass in the reference pathtracer

The unreleased pathtracer treats `AlphaMode::Blend` surfaces as smooth thin panes.
Camera and subsequent BSDF rays can reflect from the pane or transmit through
its base-color tint. Realtime Solarik still skips blended panes in its rays and
draws them with the raster forward pass. Realtime primary-glass compositing and
DLSS guide ownership are separate work.

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
- Realtime camera-visible panes need a new primary-glass path: its current
  specular pass begins at the opaque G-buffer, so changing secondary rays alone
  cannot produce that result. Raster reflection suppression must ship with that
  compositor to avoid double reflections.
