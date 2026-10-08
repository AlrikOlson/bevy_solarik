//! Persistent allocation with a full, exact TLAS build for each changed scene.
use bevy_render::{render_resource::*, renderer::RenderDevice};

#[derive(Default)]
pub(crate) struct SceneTlas {
    allocation: Option<Tlas>,
}

impl SceneTlas {
    pub(crate) fn retained(&mut self, required: usize) -> Option<&mut Tlas> {
        self.allocation
            .as_mut()
            .filter(|t| t.get().len() >= required)
    }
    pub(crate) fn get(&self) -> Option<&Tlas> {
        self.allocation.as_ref()
    }
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
        // The full path republishes a compact descriptor prefix. Stable shader
        // IDs are custom instance data, independent of these physical addresses.
        tlas.get_mut_slice(0..capacity).unwrap().fill(None);
        (tlas, reused)
    }
}
/// Changed compact descriptors, including swap-removals and appended ready slots.
/// Stable shader indices remain values; physical descriptor positions may move.
pub(crate) fn changed_dense_slots(
    old: &[u32],
    next: impl Iterator<Item = u32>,
    dirty: impl Fn(u32) -> bool,
) -> Vec<(u32, u32)> {
    next.enumerate()
        .filter(|(index, slot)| old.get(*index) != Some(slot) || dirty(*slot))
        .map(|(index, slot)| {
            (
                u32::try_from(index).expect("TLAS descriptor capacity"),
                slot,
            )
        })
        .collect()
}
/// Publish a physical descriptor; its custom shader index remains independent.
/// wgpu still performs a full Build; sparse CPU publication is not a refit.
pub(crate) fn publish_slot(tlas: &mut Tlas, slot: u32, instance: Option<TlasInstance>) {
    assert!(slot < (1 << 24), "TLAS custom slot capacity");
    *tlas
        .get_mut_single(slot as usize)
        .expect("preflighted TLAS capacity") = instance;
}
