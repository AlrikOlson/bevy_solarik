//! Renderer-owned identity, active indirection and completed-GPU retirement.
use crate::renderer::RenderQueue;
use alloc::{sync::Arc, vec::Vec};
use bevy_ecs::entity::{Entity, EntityHashMap};
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
struct Entry {
    entity: Option<Entity>,
    generation: u32,
    seen: u64,
    active: Option<usize>,
}

/// Reclamation waits for actual completion of previously submitted consumers,
/// rather than assuming a number of frames. Pending assets retain their identity
/// but never enter the compact work list.
#[derive(Default)]
pub struct SceneSlots {
    entities: EntityHashMap<SceneSlot>,
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
        let slot = match self.entities.get(&entity).copied() {
            Some(slot) => slot,
            None => self.allocate(entity),
        };
        self.entries[slot.index as usize].seen = self.epoch;
        slot
    }
    /// Observe a retained query address only when its owner and generation still match.
    pub fn touch_known(&mut self, slot: SceneSlot, entity: Entity) -> bool {
        let Some(entry) = self.entries.get_mut(slot.index as usize) else {
            return false;
        };
        if entry.entity != Some(entity) || entry.generation != slot.generation {
            return false;
        }
        entry.seen = self.epoch;
        true
    }
    fn allocate(&mut self, entity: Entity) -> SceneSlot {
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
        self.entities.get(&entity).copied()
    }
    /// Return the live owner of a persistent address.
    pub fn entity(&self, index: u32) -> Option<Entity> {
        self.entries.get(index as usize)?.entity
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
    /// Remove unobserved entities and return their old generational identities.
    pub fn finish(&mut self, queue: &RenderQueue) -> Vec<SceneSlot> {
        let removed = self.unobserved();
        if !removed.is_empty() {
            self.retire(&removed);
            let (completed, fence) = (self.completed.clone(), self.fence);
            queue.on_submitted_work_done(move || {
                completed.fetch_max(fence, Ordering::Release);
            });
        }
        removed
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
    fn retired_generations_cannot_alias_before_completion() {
        let mut slots = SceneSlots::default();
        slots.begin();
        let a = slots.touch(Entity::from_bits(1));
        slots.entities.remove(&Entity::from_bits(1));
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
