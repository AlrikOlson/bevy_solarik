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

Existing ReSTIR receivers and world caches remain one-sided. Foliage paths
bypass those caches; other pixels retain their old lighting. Secondary foliage
seen exclusively through those old caches/glossy paths still lacks transmission.
Paths in the new pass skip blended glass as the old diffuse GI does; primary
glass can still composite over the finished foliage. Bounce truncation can lose
indirect light, raw output is noisy, and the pass adds GPU cost. Coplanar,
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
This is a correctness-first fallback. Reservoir integration, reflected-leaf
transport, denoising and performance improvements remain follow-up work.

PDF conversion follows [PBRT light sampling](https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources):
area density becomes solid-angle density through distance squared divided by
the emitter cosine. The legacy reference/glossy emitter MIS still needs a separate
measure-consistency audit; the new foliage pass performs the conversion.

