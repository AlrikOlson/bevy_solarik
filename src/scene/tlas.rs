//! Persistent allocation with a full, exact TLAS build for each changed scene.
use bevy_render::{render_resource::*, renderer::RenderDevice};

#[derive(Default)]
pub(crate) struct SceneTlas {
    allocation: Option<Tlas>,
}

impl SceneTlas {
    pub(crate) fn prepare(&mut self, device: &RenderDevice, required: usize) -> (&mut Tlas, bool) {
        let reused = self
            .allocation
            .as_ref()
            .is_some_and(|t| t.get().len() >= required);
        if !reused {
            self.allocation = Some(device.wgpu_device().create_tlas(&CreateTlasDescriptor {
                label: Some("persistent_scene_tlas"),
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
                max_instances: required.max(1).next_multiple_of(4096) as u32,
            }));
        }
        let tlas = self.allocation.as_mut().unwrap();
        let capacity = tlas.get().len();
        // A dense prefix is filled by the caller. Removed and unavailable
        // instances must not survive in the retained capacity's tail.
        tlas.get_mut_slice(0..capacity).unwrap().fill(None);
        (tlas, reused)
    }
}
