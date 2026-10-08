//! Renderer-owned identity, active indirection and completed-GPU retirement.
use crate::renderer::RenderQueue;
use alloc::{sync::Arc, vec::Vec};
use bevy_ecs::entity::Entity;
use bevy_platform::collections::HashMap;
use core::sync::atomic::{AtomicU64, Ordering};

const SLOT_PAGE: usize = 4096;

/// A renderer address paired with its non-aliasing CPU generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneSlot {
    /// Address in persistent shader record arrays.
    pub index: u32,
    /// Incremented only when a completed retired address is reused.
    pub generation: u32,
}
/// Stable source identity within a shared assembly. Part zero is the ordinary entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneInstance {
    pub entity: Entity,
    pub part: u32,
}
impl From<Entity> for SceneInstance {
    fn from(entity: Entity) -> Self {
        Self { entity, part: 0 }
    }
}
struct Entry {
    entity: Option<SceneInstance>,
    generation: u32,
    seen: u64,
    active: Option<usize>,
}

/// Reclamation waits for actual completion of previously submitted consumers,
/// rather than assuming a number of frames. Pending assets retain their identity
/// but never enter the compact work list.
#[derive(Default)]
pub struct SceneSlots {
    entities: HashMap<SceneInstance, SceneSlot>,
    entries: Vec<Entry>,
    active: Vec<u32>,
    free: Vec<u32>,
    retired: Vec<(u32, u64)>,
    completed: Arc<AtomicU64>,
    epoch: u64,
    fence: u64,
    /// Monotonic revision of active membership or explicitly changed records.
    pub publication: u64,
}
impl SceneSlots {
    /// Begin an extraction epoch and reclaim only completed GPU retirements.
    pub fn begin(&mut self) {
        let complete = self.completed.load(Ordering::Acquire);
        self.retired.retain(|&(index, fence)| {
            if fence > complete {
                return true;
            }
            self.free.push(index);
            false
        });
        self.epoch = self
            .epoch
            .checked_add(1)
            .expect("scene extraction epoch exhausted");
    }
    /// Observe an entity without changing its slot or readiness.
    pub fn touch(&mut self, entity: Entity) -> SceneSlot {
        self.touch_part(entity.into())
    }
    /// Assemblies keep stable part identities without a game-world entity per leaf.
    pub fn touch_part(&mut self, entity: SceneInstance) -> SceneSlot {
        let slot = match self.entities.get(&entity).copied() {
            Some(slot) => slot,
            None => self.allocate(entity),
        };
        self.entries[slot.index as usize].seen = self.epoch;
        slot
    }
    /// Observe a retained query address only when its owner and generation still match.
    pub fn touch_known(&mut self, slot: SceneSlot, entity: Entity) -> bool {
        self.touch_part_known(slot, entity.into())
    }
    /// Observe a shared part only if both owner and GPU generation still match.
    pub fn touch_part_known(&mut self, slot: SceneSlot, entity: SceneInstance) -> bool {
        let Some(entry) = self.entries.get_mut(slot.index as usize) else {
            return false;
        };
        if entry.entity != Some(entity) || entry.generation != slot.generation {
            return false;
        }
        entry.seen = self.epoch;
        true
    }
    fn allocate(&mut self, entity: SceneInstance) -> SceneSlot {
        if self.free.is_empty() {
            // Match persistent GPU record allocation. Exact pages avoid both
            // per-arrival CPU record growth and geometric Entry over-allocation.
            let start = self.entries.len();
            let end = start.checked_add(SLOT_PAGE).expect("scene slot capacity");
            u32::try_from(end).expect("scene slot capacity");
            self.entries.reserve_exact(SLOT_PAGE);
            self.entries.resize_with(end, || Entry {
                entity: None,
                generation: 0,
                seen: 0,
                active: None,
            });
            self.free.extend((start as u32..end as u32).rev());
        }
        let index = self.free.pop().unwrap();
        let entry = &mut self.entries[index as usize];
        entry.generation = entry
            .generation
            .checked_add(1)
            .expect("scene generation exhausted");
        entry.entity = Some(entity);
        let slot = SceneSlot {
            index,
            generation: entry.generation,
        };
        self.entities.insert(entity, slot);
        slot
    }
    /// Return the retained address of an observed entity.
    pub fn get(&self, entity: Entity) -> Option<SceneSlot> {
        self.get_part(entity.into())
    }
    /// Return a stable part address, including not-yet-ready source geometry.
    pub fn get_part(&self, entity: SceneInstance) -> Option<SceneSlot> {
        self.entities.get(&entity).copied()
    }
    /// Return the live owner of a persistent address.
    pub fn entity(&self, index: u32) -> Option<Entity> {
        self.entries
            .get(index as usize)?
            .entity
            .map(|owner| owner.entity)
    }
    /// Check both generation and current readiness.
    pub fn is_active(&self, slot: SceneSlot) -> bool {
        self.entries.get(slot.index as usize).is_some_and(|e| {
            e.generation == slot.generation && e.entity.is_some() && e.active.is_some()
        })
    }
    /// Append a ready slot to the compact GPU work list.
    pub fn activate(&mut self, slot: SceneSlot) -> bool {
        let entry = &mut self.entries[slot.index as usize];
        assert_eq!(entry.generation, slot.generation);
        assert!(entry.entity.is_some(), "retired scene slot");
        if entry.active.is_some() {
            return false;
        }
        entry.active = Some(self.active.len());
        self.active.push(slot.index);
        self.changed();
        true
    }
    /// Suppress a slot while retaining its identity for late asset readiness.
    pub fn deactivate(&mut self, slot: SceneSlot) -> bool {
        let entry = &mut self.entries[slot.index as usize];
        assert_eq!(entry.generation, slot.generation);
        assert!(entry.entity.is_some(), "retired scene slot");
        let Some(position) = entry.active.take() else {
            return false;
        };
        self.active.swap_remove(position);
        if let Some(&moved) = self.active.get(position) {
            self.entries[moved as usize].active = Some(position);
        }
        self.changed();
        true
    }
    /// Find unobserved addresses only in the caller's full-snapshot subset.
    /// Other owners may use delta extraction in the same slot pool.
    pub fn unobserved_from(&self, slots: &[SceneSlot]) -> Vec<(SceneInstance, SceneSlot)> {
        slots
            .iter()
            .filter_map(|&slot| {
                let entry = self.entries.get(slot.index as usize)?;
                (entry.generation == slot.generation && entry.seen != self.epoch)
                    .then_some((entry.entity?, slot))
            })
            .collect()
    }

    /// Remove unobserved entities and return their old generational identities.
    pub fn finish(&mut self, queue: &RenderQueue) -> Vec<SceneSlot> {
        let removed = self.unobserved();
        self.retire_completed(queue, &removed);
        removed
    }
    /// Retire explicitly removed parts in a delta epoch without scanning or
    /// touching other roots. Stale generations and duplicate requests are ignored.
    pub fn remove_parts(
        &mut self,
        queue: &RenderQueue,
        parts: impl IntoIterator<Item = (SceneInstance, SceneSlot)>,
    ) -> Vec<SceneSlot> {
        let removed = self.removal_slots(parts);
        self.retire_completed(queue, &removed);
        removed
    }
    fn removal_slots(
        &self,
        parts: impl IntoIterator<Item = (SceneInstance, SceneSlot)>,
    ) -> Vec<SceneSlot> {
        let mut removed: Vec<_> = parts
            .into_iter()
            .filter_map(|(owner, slot)| (self.get_part(owner) == Some(slot)).then_some(slot))
            .collect();
        removed.sort_unstable_by_key(|slot| slot.index);
        removed.dedup_by_key(|slot| slot.index);
        removed
    }
    fn retire_completed(&mut self, queue: &RenderQueue, removed: &[SceneSlot]) {
        if removed.is_empty() {
            return;
        }
        self.retire(removed);
        let (completed, fence) = (self.completed.clone(), self.fence);
        queue.on_submitted_work_done(move || {
            completed.fetch_max(fence, Ordering::Release);
        });
    }
    fn unobserved(&self) -> Vec<SceneSlot> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.entity.is_some() && e.seen != self.epoch)
            .map(|(i, e)| SceneSlot {
                index: i as u32,
                generation: e.generation,
            })
            .collect()
    }
    fn retire(&mut self, removed: &[SceneSlot]) {
        self.fence = self
            .fence
            .checked_add(1)
            .expect("scene retirement fence exhausted");
        for &slot in removed {
            self.deactivate(slot);
            let entry = &mut self.entries[slot.index as usize];
            self.entities.remove(&entry.entity.take().unwrap());
            self.retired.push((slot.index, self.fence));
        }
    }

    /// Record an actual membership or shader-record change.
    pub fn changed(&mut self) {
        self.publication = self
            .publication
            .checked_add(1)
            .expect("scene publication exhausted");
    }
    /// Compact indices for ready instances; all later consumers use these slots.
    pub fn active_indices(&self) -> &[u32] {
        &self.active
    }
    /// Live, reusable and GPU-retired addresses partition the allocated pages.
    pub fn counts(&self) -> (usize, usize, usize) {
        let counts = (self.entities.len(), self.free.len(), self.retired.len());
        debug_assert_eq!(counts.0 + counts.1 + counts.2, self.entries.len());
        counts
    }

    /// Persistent address capacity, including holes and retirements.
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paged_slots_partition_addresses_and_reuse_only_completed_generations() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let mut rotating: Vec<_> = (1..=SLOT_PAGE as u32 + 1)
            .map(|id| slots.touch(Entity::from_raw_u32(id).unwrap()))
            .collect();
        let capacity = slots.capacity();
        assert_eq!(capacity, SLOT_PAGE * 2);
        assert_eq!(slots.counts(), (SLOT_PAGE + 1, SLOT_PAGE - 1, 0));
        let mut next_entity = SLOT_PAGE as u32 + 2;
        for _ in 0..8 {
            let retired = rotating.drain(..512).collect::<Vec<_>>();
            slots.retire(&retired);
            assert_eq!(slots.counts().2, 512);
            let pending: Vec<_> = (0..512)
                .map(|_| {
                    let slot = slots.touch(Entity::from_raw_u32(next_entity).unwrap());
                    next_entity += 1;
                    assert!(!retired.iter().any(|old| old.index == slot.index));
                    slot
                })
                .collect();
            slots.completed.store(slots.fence, Ordering::Release);
            slots.begin();
            assert_eq!(slots.counts().2, 0);
            for old in retired.iter().rev() {
                let replacement = slots.touch(Entity::from_raw_u32(next_entity).unwrap());
                next_entity += 1;
                assert_eq!(replacement.index, old.index);
                assert_eq!(replacement.generation, old.generation + 1);
                assert!(!slots.touch_known(*old, slots.entity(old.index).unwrap()));
                rotating.push(replacement);
            }
            slots.retire(&pending);
            slots.completed.store(slots.fence, Ordering::Release);
            slots.begin();
            assert_eq!(slots.capacity(), capacity);
            assert_eq!(slots.counts(), (SLOT_PAGE + 1, SLOT_PAGE - 1, 0));
        }
    }

    #[test]
    fn root_and_shared_parts_keep_independent_generation_safe_addresses() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let entity = Entity::from_bits(11);
        let root = slots.touch(entity);
        let a = SceneInstance { entity, part: 1 };
        let b = SceneInstance { entity, part: 2 };
        let sa = slots.touch_part(a);
        let sb = slots.touch_part(b);
        assert_ne!(root.index, sa.index);
        assert_ne!(sa.index, sb.index);
        slots.activate(sa);
        slots.activate(sb);
        slots.begin();
        assert!(slots.touch_part_known(sb, b));
        assert!(
            !slots.touch_part_known(sa, b),
            "part ownership must not alias"
        );
        slots.retire(&[sa]);
        assert!(slots.is_active(sb));
        slots.completed.store(slots.fence, Ordering::Release);
        slots.begin();
        let replacement = slots.touch_part(a);
        assert_eq!(replacement.index, sa.index);
        assert_ne!(replacement.generation, sa.generation);
        assert!(!slots.touch_part_known(sa, a));
    }
    #[test]
    fn active_indirection_preserves_holes_and_readiness_identity() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let a = slots.touch(Entity::from_bits(1));
        let b = slots.touch(Entity::from_bits(2));
        slots.activate(a);
        slots.activate(b);
        slots.deactivate(a);
        assert_eq!(slots.active_indices(), &[b.index]);
        assert_eq!(slots.touch(Entity::from_bits(1)), a);
        slots.activate(a);
        assert!(slots.is_active(a));
        assert_eq!(slots.active_indices(), &[b.index, a.index]);
    }
    #[test]
    fn known_slot_observation_rejects_aliases_and_marks_only_the_right_owner() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let entity = Entity::from_bits(1);
        let slot = slots.touch(entity);
        slots.begin();
        assert!(!slots.touch_known(slot, Entity::from_bits(2)));
        assert_eq!(slots.unobserved(), [slot]);
        let stale = SceneSlot {
            generation: slot.generation + 1,
            ..slot
        };
        assert!(!slots.touch_known(stale, entity));
        assert_eq!(slots.unobserved(), [slot]);
        assert!(slots.touch_known(slot, entity));
        assert!(slots.unobserved().is_empty());
        slots.retire(&[slot]);
        assert!(!slots.touch_known(slot, entity));
        slots.completed.store(slots.fence, Ordering::Release);
        slots.begin();
        let replacement = slots.touch(entity);
        assert_ne!(slot.generation, replacement.generation);
        assert!(!slots.touch_known(slot, entity));
        assert!(slots.touch_known(replacement, entity));
    }
    #[test]
    fn partial_retirement_preserves_unvisited_roots_and_rejects_stale_requests() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let a = SceneInstance {
            entity: Entity::from_bits(1),
            part: 1,
        };
        let b = SceneInstance {
            entity: Entity::from_bits(2),
            part: 1,
        };
        let sa = slots.touch_part(a);
        let sb = slots.touch_part(b);
        slots.activate(sa);
        slots.activate(sb);
        slots.begin(); // Neither was visited: delta retirement must leave b alone.
        let removed = slots.removal_slots([(a, sa), (a, sa), (b, sa)]);
        assert_eq!(removed, [sa]);
        slots.retire(&removed);
        assert_eq!(slots.active_indices(), [sb.index]);
        assert!(slots.removal_slots([(a, sa)]).is_empty());
        let pending = slots.touch_part(a);
        assert_ne!(pending.index, sa.index);
        slots.completed.store(slots.fence, Ordering::Release);
        slots.begin();
        let c = SceneInstance {
            entity: Entity::from_bits(3),
            part: 1,
        };
        let reused = slots.touch_part(c);
        assert_eq!(reused.index, sa.index);
        assert_ne!(reused.generation, sa.generation);
        assert!(slots.removal_slots([(c, sa), (a, sa)]).is_empty());
        assert!(slots.is_active(sb));
    }
    #[test]
    fn snapshot_subset_does_not_retire_unvisited_assembly_parts() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let root = Entity::from_bits(19);
        let ordinary = slots.touch(root);
        let part = slots.touch_part(SceneInstance {
            entity: root,
            part: 1,
        });
        slots.activate(ordinary);
        slots.activate(part);
        slots.begin();
        assert!(slots.touch_known(ordinary, root));
        assert!(slots.unobserved_from(&[ordinary]).is_empty());
        slots.begin();
        assert_eq!(
            slots.unobserved_from(&[ordinary]),
            [(root.into(), ordinary)]
        );
        let stale = SceneSlot {
            generation: ordinary.generation + 1,
            ..ordinary
        };
        assert!(slots.unobserved_from(&[stale]).is_empty());
        assert!(slots.is_active(part));
    }
    #[test]
    fn retired_generations_cannot_alias_before_completion() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let a = slots.touch(Entity::from_bits(1));
        slots
            .entities
            .remove(&SceneInstance::from(Entity::from_bits(1)));
        slots.entries[a.index as usize].entity = None;
        slots.retired.push((a.index, 7));
        slots.begin();
        let b = slots.touch(Entity::from_bits(2));
        assert_ne!(a.index, b.index);
        slots.completed.store(7, Ordering::Release);
        slots.begin();
        let c = slots.touch(Entity::from_bits(3));
        assert_eq!(c.index, a.index);
        assert_eq!(c.generation, a.generation + 1);
        assert!(!slots.is_active(a));
    }
}
