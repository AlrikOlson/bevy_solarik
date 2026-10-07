//! Mutation receipts for persistent render scene material membership.
use crate::RenderMaterialInstance;
use bevy_render::sync_world::MainEntityHashMap;
use core::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_MAP: AtomicU64 = AtomicU64::new(1);

/// A material membership map whose public mutable access invalidates scene caches.
/// Replacing the complete map also changes its unique identity. Read access does
/// not change the receipt, including when the resource's sweep tick advances.
pub struct MaterialInstanceMap {
    values: MainEntityHashMap<RenderMaterialInstance>,
    identity: u64,
    revision: u64,
}

impl Default for MaterialInstanceMap {
    fn default() -> Self {
        Self {
            values: MainEntityHashMap::default(),
            identity: NEXT_MAP
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("material map identity capacity"),
            revision: 0,
        }
    }
}

impl MaterialInstanceMap {
    /// Exact invalidation receipt for public writes or complete map replacement.
    pub fn receipt(&self) -> (u64, u64) {
        (self.identity, self.revision)
    }
}

impl Deref for MaterialInstanceMap {
    type Target = MainEntityHashMap<RenderMaterialInstance>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl DerefMut for MaterialInstanceMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("material map revision capacity");
        &mut self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DUMMY_MESH_MATERIAL, RenderMaterialInstances};
    use bevy_ecs::{change_detection::Tick, entity::Entity};

    #[test]
    fn persistent_material_receipt_tracks_public_mutations_and_replacement_not_reads_or_sweep() {
        let mut materials = RenderMaterialInstances::default();
        let initial = materials.instances.receipt();
        let entity = Entity::from_raw_u32(7).unwrap().into();
        materials.instances.insert(
            entity,
            RenderMaterialInstance {
                asset_id: DUMMY_MESH_MATERIAL.untyped(),
                last_change_tick: Tick::new(1),
            },
        );
        let inserted = materials.instances.receipt();
        assert_ne!(initial, inserted);
        assert!(materials.instances.get(&entity).is_some());
        materials.current_change_tick.set(99);
        assert_eq!(inserted, materials.instances.receipt());
        materials
            .instances
            .get_mut(&entity)
            .unwrap()
            .last_change_tick
            .set(2);
        assert_ne!(inserted, materials.instances.receipt());
        let before_remove = materials.instances.receipt();
        materials.instances.remove(&entity);
        assert_ne!(before_remove, materials.instances.receipt());
        let removed = materials.instances.receipt();
        materials.instances = MaterialInstanceMap::default();
        assert_ne!(removed.0, materials.instances.receipt().0);
    }
}
