//! Mutation receipts for persistent render scene material membership.
use crate::RenderMaterialInstance;
use alloc::collections::VecDeque;
use bevy_render::sync_world::{MainEntity, MainEntityHashMap};
use core::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_MAP: AtomicU64 = AtomicU64::new(1);
const CHANGE_LIMIT: usize = 8192;

/// A material membership map whose public mutable access invalidates scene caches.
/// Replacing the complete map also changes its unique identity. Read access does
/// not change the receipt, including when the resource's sweep tick advances.
pub struct MaterialInstanceMap {
    values: MainEntityHashMap<RenderMaterialInstance>,
    identity: u64,
    revision: u64,
    invalidated_through: u64,
    changes: VecDeque<(u64, MainEntity)>,
}

impl Default for MaterialInstanceMap {
    fn default() -> Self {
        Self {
            values: MainEntityHashMap::default(),
            identity: NEXT_MAP
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("material map identity capacity"),
            revision: 0,
            invalidated_through: 0,
            changes: VecDeque::new(),
        }
    }
}

impl MaterialInstanceMap {
    fn advance(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("material map revision capacity");
    }

    fn record(&mut self, entity: MainEntity) {
        self.advance();
        if self.changes.len() == CHANGE_LIMIT {
            self.invalidated_through = self.changes.pop_front().unwrap().0;
        }
        self.changes.push_back((self.revision, entity));
    }

    /// Insert through the bounded membership journal. Unknown mutable access
    /// still invalidates every consumer through `DerefMut`.
    pub fn insert(
        &mut self,
        entity: MainEntity,
        value: RenderMaterialInstance,
    ) -> Option<RenderMaterialInstance> {
        let old = self.values.insert(entity, value);
        self.record(entity);
        old
    }

    /// Remove a membership and retain its change until bounded journal eviction.
    pub fn remove(&mut self, entity: &MainEntity) -> Option<RenderMaterialInstance> {
        let old = self.values.remove(entity);
        if old.is_some() {
            self.record(*entity);
        }
        old
    }

    /// Exact changed owners since a consumer receipt. A replaced map, missed
    /// journal entries or arbitrary public mutation requires a complete refresh.
    pub fn changes_since(
        &self,
        receipt: (u64, u64),
    ) -> Option<impl Iterator<Item = MainEntity> + '_> {
        if receipt.0 != self.identity
            || receipt.1 < self.invalidated_through
            || receipt.1 > self.revision
        {
            return None;
        }
        Some(
            self.changes
                .iter()
                .filter(move |(revision, _)| *revision > receipt.1)
                .map(|(_, entity)| *entity),
        )
    }

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
        self.advance();
        self.invalidated_through = self.revision;
        self.changes.clear();
        &mut self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DUMMY_MESH_MATERIAL, RenderMaterialInstances};
    use bevy_ecs::{change_detection::Tick, entity::Entity};

    #[test]
    fn journal_tracks_insert_remove_reinsert_and_fails_closed_on_gaps() {
        let mut map = MaterialInstanceMap::default();
        let a = Entity::from_raw_u32(7).unwrap().into();
        let b = Entity::from_raw_u32(8).unwrap().into();
        let value = || RenderMaterialInstance {
            asset_id: DUMMY_MESH_MATERIAL.untyped(),
            last_change_tick: Tick::new(1),
        };
        let start = map.receipt();
        map.insert(a, value());
        let first = map.receipt();
        map.insert(b, value());
        map.remove(&a);
        map.insert(a, value());
        assert_eq!(
            map.changes_since(start).unwrap().collect::<Vec<_>>(),
            [a, b, a, a]
        );
        assert_eq!(
            map.changes_since(first).unwrap().collect::<Vec<_>>(),
            [b, a, a]
        );
        let before_unknown = map.receipt();
        map.get_mut(&a).unwrap().last_change_tick.set(9);
        assert!(map.changes_since(before_unknown).is_none());
        let after_unknown = map.receipt();
        assert_eq!(map.changes_since(after_unknown).unwrap().count(), 0);
        for _ in 0..CHANGE_LIMIT {
            map.insert(b, value());
        }
        assert_eq!(
            map.changes_since(after_unknown).unwrap().count(),
            CHANGE_LIMIT
        );
        map.insert(a, value());
        assert!(map.changes_since(after_unknown).is_none());
        assert_eq!(map.changes.len(), CHANGE_LIMIT);
        assert!(
            map.changes_since(MaterialInstanceMap::default().receipt())
                .is_none()
        );
        assert!(
            map.changes_since((map.identity, map.revision + 1))
                .is_none()
        );
    }

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
