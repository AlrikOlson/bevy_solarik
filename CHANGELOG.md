# Changelog

## 0.1.0 — 2026-09-06

First tagged release, based on `bevy_solari` 0.19.1.

- Sky cubemap lighting on rays that leave the scene.
- Point and spot lights with Bevy's units and falloff.
- Alpha-tested foliage and light rays that pass through blended materials.
- A reference pathtracer that accumulates while the camera is still.
- Power-weighted light selection, with [measurements](docs/light-sampling.md) from the Bistro scene.
- Optional DLSS Ray Reconstruction.

Tested on Windows, Vulkan and an RTX 4090. Glass refraction, leaf transmission and sky importance sampling are still missing. See the [README](README.md) for setup and the current limits.
