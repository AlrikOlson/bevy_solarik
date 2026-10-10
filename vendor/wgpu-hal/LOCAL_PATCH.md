Published wgpu-hal 29.0.4, upstream revision
e99f5305ded96ff7006f0714d043a7f735bd45c2. Original licenses are retained.

The Device and dynamic Device traits add tlas_instance_to_bytes_into. Its default
preserves the previous backend implementation. Vulkan overrides it to append the
same RawTlasInstance bytes directly from the stack into the caller's vector. The
old method remains available. There are no layout, synchronization, build-mode,
or unsafe-code changes. This removes one allocation per instance from the safe
wgpu-core TLAS build path; see ../wgpu-core/LOCAL_PATCH.md and the renderer's
tests/scene_tlas_scale.rs and tests/scene_tlas_validation.rs.

Upstream context: https://github.com/gfx-rs/wgpu/issues/9946 (2026-10-10).
