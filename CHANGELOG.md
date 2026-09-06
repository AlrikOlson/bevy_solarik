# Changelog

## Unreleased

- Solarik now includes its own `bevy_render` fork. The CPU-visible entity lookup uses the list's main-entity ordering, preserving visible meshes while their materials load. Applications using development checkouts must apply the root Cargo patch documented in the README; tagged `v0.1.0` is unchanged.

- Realtime primary foliage now uses a depth-matched, four-path/four-vertex estimator with two-sided diffuse transport and stable leaf DLSS guides. Existing opaque pixels and caches retain their shading; [Bistro validation and measured cost](docs/foliage.md) document the limits.

- Reference foliage transmission: authored two-sided masked materials split diffuse energy across both hemispheres, with matching sampling/MIS PDFs and outgoing visibility offsets. CPU packing, production diffuse GPU contracts and a three-card backlight capture validate the implementation.

- Primary realtime glass now composites tinted transmission and ray-traced reflections after opaque rendering. Ready views suppress only TLAS-present pane draws, retaining raster fallback and deterministic background DLSS guides. Eleven production WGSL cases and an analytic scene validate transport.

- Realtime glossy reflections now resolve thin glass with tinted transmission, bounded pane chains and emission weighting. Glass stops DLSS primary-surface replacement; primary camera glass is handled by the compositor. The production transport GPU test covers both DLSS guide variants.

- Reference-pathtracer thin glass: Fresnel reflection, tinted straight-through transmission, texture-alpha coverage and two-sided outgoing offsets. [GPU and integrated scene validation, reproduction and limits](docs/glass.md).

- Sky cubemap importance sampling with cosine-hemisphere MIS for the first realtime GI bounce; black-sky fallback and counted zero samples.
- GPU sampling validation and [Bistro brightness and cost measurements](docs/sky-sampling.md). The measured baseline already meets the brightness target; faster convergence is not established.

## 0.1.0 — 2026-09-06

First tagged release, based on `bevy_solari` 0.19.1.

- Sky cubemap lighting on rays that leave the scene.
- Point and spot lights with Bevy's units and falloff.
- Alpha-tested foliage and light rays that pass through blended materials.
- A reference pathtracer that accumulates while the camera is still.
- Power-weighted light selection, with [measurements](docs/light-sampling.md) from the Bistro scene.
- Optional DLSS Ray Reconstruction.

Tested on Windows, Vulkan and an RTX 4090. Glass refraction, leaf transmission and sky importance sampling are still missing. See the [README](README.md) for setup and the current limits.
