This directory vendors the published wgpu-core 29.0.4 source (upstream revision
e99f5305ded96ff7006f0714d043a7f735bd45c2), with its original MIT and Apache licenses.

The local change is limited to src/command/ray_tracing.rs. Command encoding caches
BLAS resolution by the complete generation-bearing ID. TLAS encoding tracks and
retains each distinct BLAS once, rather than once per instance. Instance ordering,
transforms, masks, custom-index checks, serialization, barriers and build modes are
unchanged. Each unique dependency still undergoes submission and use validation.
The companion wgpu-hal patch appends native instance bytes without an allocation
per instance. Resource identity uses the owned Arc's address, which remains stable
through encoding even when device-local tracker indices overlap.

Reason: the production forest's 789,000-instance sparse-update probe measured
29–30 ms encoding, 48–53 ms finishing, and 13–16 ms submitting with the original
crate. The corresponding GPU build in the game measured less than 2 ms.
The combined patch measured 28.6–29.3 ms encoding, 23.8–24.9 ms finishing, and
0.126–0.159 ms submitting with mixed shared geometries: about 43% less CPU work.
The release ray-hit controls passed sparse motion/restoration, pending/empty/reuse,
invalid custom indices, unbuilt distinct BLAS and BLAS rebuilt after their TLAS.
Upstream context: https://github.com/gfx-rs/wgpu/issues/9946 and
https://github.com/gfx-rs/wgpu/issues/10450 (reviewed 2026-10-10).

Controls live in ../../tests/scene_tlas_scale.rs and ../../tests/scene_tlas.rs.
This patch does not introduce the proposed GPU-instance-buffer API or unsafe HAL
access. Remove it when an upstream version provides equivalent behavior and passes
the same production-path controls.
