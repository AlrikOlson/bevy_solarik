# bevy_solarik

Ray-traced lighting for Bevy 0.19. This is a fork of `bevy_solari` 0.19.1 (the experimental ray tracer that ships with Bevy: ReSTIR direct and indirect light, a world-space radiance cache, DLSS Ray Reconstruction as the denoiser) with the things I needed to light real scenes with it: a sky, alpha-tested foliage and glass, and a reference pathtracer that accumulates.

It is a separate crate with its own name so it can sit next to Bevy's own `bevy_solari` without a `[patch]`. Shader import paths are `bevy_solarik::...`, the public types are `SolarikPlugins`, `SolarikLighting`, `SolarikSkyLight` and `SolarikAlphaTesting`; `RaytracingMesh3d`, `Pathtracer` and `PathtracingPlugin` keep their upstream names.

Built and tested against Bevy 0.19.1 on an RTX 4090, Vulkan, Windows. That is the only place it has run.

## What it adds over upstream

**A sky.** Upstream lights a scene from directional lights and emissive meshes only; a ray that leaves the scene contributes nothing, so anything lit by the sky alone renders black. Here a `SolarikSkyLight` resource (a cubemap in the scene's radiance units, plus an intensity) is bound into the ray-traced scene and read wherever a traced ray escapes: the ReSTIR GI first bounce (the escaped ray becomes a far sample point and is reused like any other), the world cache's GI rays (traced to the end of the world so distant geometry is not mistaken for sky), the specular GI paths and the pathtracer.

The sky is not in the direct-light list on purpose. A first version put it there as uniform-sphere light samples; in a street canyon almost every sample that wins the resampling is occluded, and the spiky estimate lost about two stops through the tonemapper and the denoiser. Rays that escape are visible by construction, so the miss paths are the better estimator. Importance sampling the cubemap is the obvious follow-up.

**Alpha-masked and blended materials.** Upstream builds every BLAS opaque and the rays see leaf cards and window glass as solid quads. Here the binder looks at the materials on each mesh: a mesh with a `Mask` or `Blend` material gets its BLAS rebuilt without the `OPAQUE` flag, and `trace_ray` walks the candidate hits (`rayQueryProceed` / `rayQueryConfirmIntersection`), alpha-testing `Mask` candidates against the base colour texture and never confirming `Blend` ones (glass is transparent to light). Double-sided materials hit from behind are shaded from the side the ray arrived on, as the rasteriser already does. The material flags live in what used to be `GpuMaterial`'s padding, so the layout is unchanged. `SolarikAlphaTesting(false)` turns it all off for a scene that cannot afford the any-hit work (a dense tree in view adds about 2 ms to the lighting passes at 1080p and at 4K). No transmission yet: a leaf lit from the other side stays dark.

**Point and spot lights.** Upstream ignores `PointLight` and `SpotLight`; a lamp-lit scene had to fake its lamps as emissive meshes, and the numbers never matched the raster path. Here every point and spot light in the scene is a light source in the ReSTIR DI list, sampled as a sphere of the light's `radius` (floored at 1 cm, since glTF and Bevy default to 0). The units are the raster path's: `bevy_pbr` already turns lumens into candela with its 4π (for spots too), the sphere's surface radiance is that intensity over π r², so a receiver facing the light sees I / d² either way, and Filament's spot cone and range window are applied exactly as `pbr_lighting.wgsl` does. Shadow rays go to the sample point; the sample is a point on the whole sphere (not the cap a receiver would see), because Solari resolves light samples before it knows the receiver (the presampled tiles, and the resampling re-resolves them at other pixels); the far half of the sphere has zero weight and falls out of the resampling without a ray. The sphere is not geometry, so a BRDF ray can never hit it and next-event estimation gets the full weight. Temporal reuse follows a light across frames by its entity, like a directional light. Diffuse BRDFs differ by design: Solari's Lambert carries a (1 - F) layering term the raster Burley diffuse does not, which is about 0.1 EV on a rough dielectric. `tools/photometry.py` in the rig measures all of this against the pathtracer and a closed-form scene (see the rig README).

**Pathtracer accumulation.** Upstream resets the accumulation whenever `Ref<GlobalTransform>::is_changed()` says the camera moved, and read it from the extract system that fires every frame for a camera that has not moved, so the reference never got past one sample. This fork compares the transform by value. Two diagnostics came with the fix: the pathtracer node logs (at debug level) how many of the last 100 frames reset the accumulation, and `SOLARIK_PATHTRACER_DEBUG_SAMPLE_COUNT` in the environment at startup makes it draw the per-pixel sample count instead of the image (white at 512 samples).

**Power-weighted light selection.** Each frame the CPU builds a Walker alias table over estimated emitted flux. Emissive meshes use the material's luminance times world-space triangle area times π; point lights use candela times 4π; spots integrate Bevy's squared cone ramp. Directional lights use illuminance times a nominal 1 m² collection area. Emissive textures are not averaged into the estimate. These choices affect variance; each sample is divided by its actual selection probability, and emissive-hit MIS reads that same probability from the instance. The sky remains on the traced miss paths. Mesh edges are cached at upload (24 bytes per triangle) so area follows nonuniform scale and mesh edits; a dark light list falls back to uniform selection.

On the Bistro at frame 140, the 512-frame pathtracer capture is within 0.03 EV of the uniform 8192-frame reference on two cobblestone patches. Against a weighted 8192-frame reference, patch RMSE drops about 36% from uniform selection at the same warmup. Uniform selection already passed the 0.2 EV mean threshold, and the raw realtime output did not measurably improve. These are single-run measurements, not a 16× convergence claim; see [the measurement record](docs/light-sampling.md).

The deferred path is still forced, one emissive mesh may have at most 65,535 triangles, and the scene at most 65,535 light sources of every kind together.

## Usage

```toml
[dependencies]
bevy = { version = "0.19.1", features = ["dlss"] }
bevy_solarik = { git = "https://github.com/AlrikOlson/bevy_solarik", features = ["dlss"] }
```

```rust,ignore
use bevy::prelude::*;
use bevy_solarik::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SolarikPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, sky: Res<MySkyCubemap>) {
    commands.spawn((
        Camera3d::default(),
        SolarikLighting::default(), // pulls in Hdr and the prepasses it needs
        Msaa::Off,
        bevy::anti_alias::dlss::Dlss::<bevy::anti_alias::dlss::DlssRayReconstructionFeature> {
            perf_quality_mode: bevy::anti_alias::dlss::DlssPerfQualityMode::Dlaa,
            reset: false,
        },
    ));
    commands.insert_resource(SolarikSkyLight {
        image: Some(sky.0.clone()),
        intensity: 1.0,
    });
    // every mesh that should be lit, and light others, also needs RaytracingMesh3d
}
```

Meshes need exactly `POSITION`, `NORMAL`, `UV_0` and `TANGENT` with u32 indices to go into the ray-traced scene; anything else stays raster-only. The camera's `Msaa` has to be off. `SolarikLighting::reset` drops the temporal history for one frame on a camera cut. `SolarikPlugins::required_wgpu_features()` is what the GPU has to support; if it does not, the plugins log a warning and do nothing.

This crate is exercised in a separate, unpublished reel rig. The rig converts u16 index buffers and generates tangents so imported glTF fits the mesh contract. Solarik and [bevy_dlss5](https://github.com/AlrikOlson/bevy_dlss5), the DLSS 5 Neural Rendering plugin, can share one NGX instance; the two plugins are separate repositories.

## Requirements

- A GPU with ray query support on Vulkan (tested: RTX 4090). DX12 is untested.
- Bevy 0.19.1. The crate depends on Bevy's sub-crates directly, the way upstream `bevy_solari` does, so it unifies with whatever `bevy` your app pulls in.
- A zstd backend for the KTX2 lookup tables `bevy_pbr` needs. `zstd_rust` is the default feature; `default-features = false` plus `zstd_c` picks the C decoder.
- The `dlss` feature wants `DLSS_SDK`, `VULKAN_SDK` and libclang at build time (`dlss_wgpu` runs bindgen over the SDK headers) and `nvngx_dlssd.dll` next to the exe at runtime. Without the feature there is no denoiser and the image is raw ReSTIR output.

## Building this crate on its own

```
cargo check --workspace --all-targets
cargo nextest run --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

A standalone clone builds into its own `target/` directory. To share a build cache with another local Bevy project, set `CARGO_TARGET_DIR` explicitly.

## Provenance

The diff against the crates.io `bevy_solari` 0.19.1 sources is `docs/upstream-0.19.1.diff`; `tools/upstream_diff.sh` regenerates it from the local cargo registry. The rename is in there too, so it reads as provenance, not as a patch to send upstream.

Upstream is `bevy_solari` by JMS55 and the Bevy contributors, MIT OR Apache-2.0. This fork keeps that licence (see LICENSE-MIT and LICENSE-APACHE).
