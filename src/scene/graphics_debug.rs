//! Capture-only scene ownership and native buffer handles. No added default systems.
use alloc::vec::Vec;
use bevy_render::render_resource::Buffer;

/// Exact prepared scene receipt retained on settled frames.
#[derive(Default)]
pub struct SceneSnapshot {
    /// Roles and handles for bounded readback of actual GPU data, not CPU shadows.
    pub buffers: Vec<(&'static str, Buffer)>,
    /// Extracted ray instances, including unavailable BLAS candidates.
    pub instances: usize,
    /// Actual BLAS readiness/opacity/compaction generation.
    pub blas_generation: u64,
    /// Change tick of the extracted material dependency.
    pub material_change_tick: u32,
    /// Retained TLAS allocation slots.
    pub tlas_capacity: usize,
    /// Nonempty current TLAS slots, excluding cleared tail and unavailable BLAS.
    pub tlas_active: usize,
    /// Exact compact slot order, populated only with `super::CaptureSceneIndices`.
    pub active_indices: Vec<u32>,
}
