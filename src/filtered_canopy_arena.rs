//! Bounded immutable region uploads with stable offsets and queue-ordered growth.
use super::{MAX_ARENA_BYTES, REGION_ALLOCATION_BYTES};
use alloc::sync::{Arc, Weak};
use bevy_ecs::resource::Resource;
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
};
use core::ops::Range;
pub(super) struct Allocation {
    pub range: Range<u32>,
}
#[derive(Default)]
struct Slots {
    end: u32,
    free: Vec<Range<u32>>,
    live: Vec<(Range<u32>, Weak<Allocation>)>,
}
impl Slots {
    fn allocate(&mut self, rows: u32, limit: u32) -> Option<Arc<Allocation>> {
        self.collect();
        let range = if let Some(i) = self.free.iter().position(|r| r.end - r.start >= rows) {
            let start = self.free[i].start;
            self.free[i].start += rows;
            if self.free[i].is_empty() {
                self.free.remove(i);
            }
            start..start + rows
        } else {
            let end = self.end.checked_add(rows)?;
            if end > limit {
                return None;
            }
            let range = self.end..end;
            self.end = end;
            range
        };
        let allocation = Arc::new(Allocation {
            range: range.clone(),
        });
        self.live.push((range, Arc::downgrade(&allocation)));
        Some(allocation)
    }
    fn collect(&mut self) {
        self.live.retain(|(range, owner)| {
            if owner.strong_count() > 0 {
                return true;
            }
            self.free.push(range.clone());
            false
        });
        self.free.sort_unstable_by_key(|r| r.start);
        let mut merged: Vec<Range<u32>> = Vec::new();
        for range in self.free.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.end == range.start
            {
                last.end = range.end;
            } else {
                merged.push(range);
            }
        }
        while merged.last().is_some_and(|r| r.end == self.end) {
            self.end = merged.pop().expect("checked last free range").start;
        }
        self.free = merged;
    }
}
#[derive(Resource, Default)]
pub(super) struct Arena {
    pub buffer: Option<Buffer>,
    slots: Slots,
}
impl Arena {
    pub fn ensure(&mut self, device: &RenderDevice) {
        if self.buffer.is_some() {
            return;
        }
        self.buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("finite canopy region arena"),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    pub fn upload(
        &mut self,
        rows: &[[u32; 4]],
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Option<Arc<Allocation>> {
        self.ensure(device);
        if rows.len().checked_mul(16)? > REGION_ALLOCATION_BYTES {
            return None;
        }
        let limit = (MAX_ARENA_BYTES as u64)
            .min(device.limits().max_storage_buffer_binding_size)
            .min(device.limits().max_buffer_size);
        let allocation = self
            .slots
            .allocate((REGION_ALLOCATION_BYTES / 16) as u32, (limit / 16) as u32)?;
        let needed = allocation.range.end as u64 * 16;
        let old = self.buffer.as_ref()?;
        if old.size() < needed {
            let capacity = needed.next_power_of_two().min(limit);
            let new = device.create_buffer(&BufferDescriptor {
                label: Some("finite canopy region arena"),
                size: capacity,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(old, 0, &new, 0, old.size());
            queue.submit([encoder.finish()]);
            self.buffer = Some(new);
        }
        queue.write_buffer(
            self.buffer.as_ref()?,
            allocation.range.start as u64 * 16,
            bytemuck::cast_slice(rows),
        );
        Some(allocation)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_ranges_survive_retirement_and_fragment_reuse() {
        let mut slots = Slots::default();
        let a = slots.allocate(10, 32).unwrap();
        let b = slots.allocate(12, 32).unwrap();
        let c = slots.allocate(10, 32).unwrap();
        assert!(slots.allocate(1, 32).is_none());
        drop(b);
        let d = slots.allocate(11, 32).unwrap();
        assert_eq!(d.range, 10..21);
        assert_eq!(a.range, 0..10);
        assert_eq!(c.range, 22..32);
        drop(a);
        drop(c);
        drop(d);
        assert_eq!(slots.allocate(32, 32).unwrap().range, 0..32);
    }
    #[test]
    fn fixed_regional_slots_reuse_arbitrary_retirement_without_fragmentation() {
        let unit = (REGION_ALLOCATION_BYTES / 16) as u32;
        let limit = (MAX_ARENA_BYTES / 16) as u32;
        let mut slots = Slots::default();
        let mut live: Vec<_> = (0..limit / unit)
            .map(|_| Some(slots.allocate(unit, limit).unwrap()))
            .collect();
        assert!(slots.allocate(unit, limit).is_none());
        for i in (0..live.len()).step_by(2) {
            live[i] = None;
        }
        for i in (0..live.len()).step_by(2) {
            let allocation = slots.allocate(unit, limit).unwrap();
            assert_eq!(allocation.range.start % unit, 0);
            live[i] = Some(allocation);
        }
        assert!(slots.allocate(unit, limit).is_none());
        assert_eq!(slots.end, limit);
    }
}
