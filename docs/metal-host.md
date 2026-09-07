# Metal and host-renderer integration

Metal support and host-renderer integration were exercised
on Apple M5 Max / Metal (macOS 26.5.2, Bevy 0.19.1, wgpu/Naga 29).
It is based on Solarik `c531ee02ae85b26506625cf8e9b9c091dbf60a9e`.
The bundled `bevy_render` fork is unchanged.

## What changes

- Metal uses two word-addressed geometry pages instead of buffer binding
  arrays. Scene records and the cache counter share buffers to fit Metal's
  31-buffer limit; the indirect-dispatch allocation includes Metal's vec3
  padding. Each page respects both adapter limits. Uploads and vertex/index
  reads can cross the page boundary without repacking source attributes.
- Metal's Naga intersector completes at initialization; its proceed flag does
  not clear. The Metal shader consumes that result once, then recasts past
  rejected coverage at most 128 times. A saturated stack conservatively
  occludes. Vulkan keeps its candidate-confirm traversal. This is an alpha
  traversal workaround, not unbounded transparent-layer support.
- BLAS uses the source vertex stride and retains optional streams. Replacing
  a BLAS cancels its old compaction ticket; compaction runs exclusively to
  prevent lock-order deadlocks with meshlet submissions.
- Emissive samples replay the full primitive draw from the reservoir seed,
  retaining identical spatial/temporal reuse past 65,535 triangles. The
  existing light-selection distribution is unchanged.
- Hidden instances are excluded. `SolarikMaterial3d` supplies a separate
  StandardMaterial handle for rays without replacing the raster extension.
  `SolarikLightOff` excludes an analytical light represented by a mesh emitter.
  `scene_stats()` exposes requested/resident counts, missing resources,
  geometry bytes, attributes, lights and actual dispatches.

## Host denoising and ordering

`SolarikLighting::denoise_guides` defaults to false. When enabled, each render
view exposes `SolarikDenoiseGuideTextures`: diffuse/specular albedo, world-space
normal, perceptual roughness, and specular first-hit distance. All five use
RGBA16Float; scalar values use red. The host owns the denoiser, motion/depth
inputs, exposure, and camera-cut reset. MetalFX is not a new dependency here.
DLSS's own guide path takes precedence when active.

Lighting runs before opaque and optional meshlet shading; primary panes run
after opaque/meshlet/atmosphere and before raster transmission/transparency.
The exported `realtime::solarik_lighting::<true>` lets a host order its denoiser
after the primary pass. Lighting is ordered between Bevy's public prepass and
main-pass stages; primary panes follow the main opaque pass, which itself
follows meshlet shading. This needs no private Bevy entry point. The optional
`meshlet` feature forwards to Bevy's meshlet feature.

Ray material textures still use UV0. Preserving UV1, skin weights and other
streams does not implement procedural extension shaders, alternate texture
UVs, texture transforms or animated skinning in secondary rays.

## Reproduction and evidence

From this standalone checkout, run the normal CONTRIBUTING gates. The Metal
regressions are self-contained and opt-in:

```sh
cargo test --test metal_traversal
cargo test --test metal_traversal -- --ignored --test-threads=1
cargo check --workspace --all-targets --features meshlet
```

Run the CPU translation test before the ray probe. GPU readbacks use bounded
polling. The three small GPU tests exercise production accessors across five
page boundaries, eight opaque/masked/glass/miss ray cases, and 4,096 replayed
samples from a 130,001-triangle emitter. The RNG fixture is attributed Bevy
0.19.1 code; it does not require another repository beside this checkout.

The README's Windows/Bistro captures were not regenerated: that rig and
NVIDIA/DLSS hardware are unavailable on this machine. The images retain their
existing capture provenance. Vulkan traversal, the optional DLSS runtime,
large-scene performance and foliage appearance still need upstream platform
and visual validation; no claim of visual parity is made.
