# Changelog

## Unreleased

- Rough surfaces seen at grazing angles are no longer dark. Diffuse light is now reduced by the specular lobe's directional albedo from the split-sum table instead of the Fresnel reflectance of a smooth interface, which the table reproduces at zero roughness; the smooth value removed about 90% of the diffuse light at a view cosine of 0.02 whatever the roughness. A shading normal that faces away from the direction it is seen from is mirrored into the visible hemisphere instead of shading black, which had outlined the silhouettes of normal-mapped surfaces in dark pixels. A production-WGSL GPU test covers both.

- Planetary clouds can be driven by an application-supplied cube weather map (cover, cloud-top height, extinction scale) in a configurable shell, with an adiabatic vertical profile, independent-column treatment of partial cover, mip selection by footprint and sub-texel detail. `PlanetaryAtmosphere` gains `cloud_base`, `cloud_top`, `weather` and `cloud_detail`; the built-in noise cover remains when no map is given. [Details](docs/atmosphere.md#optional-spherical-world-mode-development).

- Planetary cloud lighting: droplet phase from the Jendersie & d'Eon (2023) HG + Draine fit with the forward peak delta-scaled, a marched sun optical depth, and a closed-form delta-Eddington diffuse field for multiple scattering. Thick cloud now reflects like thick cloud; a GPU test compares slab albedo and radiance with Monte Carlo. [Model, measured error and limits](docs/atmosphere.md#optional-spherical-world-mode-development).

- Planetary clouds are integrated on their own ray intervals instead of sharing the clear-air quadrature, and cloud extinction accepts physical liquid-cloud values (up to 0.2 m⁻¹). Cloud-march convergence error fell from 11.2% to 0.51%. Lookup fields are now clear air only. [Details and cost](docs/atmosphere.md#optional-spherical-world-mode-development).

- Optional GPU clear-sky atmosphere: Rayleigh/Mie scattering, ozone, multiple scattering, attenuated sun/moon lighting, photometric stars and aerial perspective. The visible sky, raster environment and ray-traced transport share physical radiance units. [Controls, numerical validation and limits](docs/atmosphere.md).

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
