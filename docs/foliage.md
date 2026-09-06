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

Realtime estimators currently leave the coefficient disabled; their G-buffer has
no transmission field. Explicitly select `OpaqueRendererMethod::Deferred` when
authoring foliage for Solarik: Bevy's Auto chooses forward for transmission.

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
This establishes controlled reference transport, not Bistro realtime acceptance.

