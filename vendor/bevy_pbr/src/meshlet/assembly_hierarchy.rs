//! A conservative root above the existing per-part BVHs. This changes traversal,
//! never the selected source representation or the ray scene.
use super::*;
use bevy_math::{Affine3A, Vec3};
use bevy_render::render_resource::ShaderType;
use core::ops::Range;
use range_alloc::RangeAllocator;

#[derive(Clone, Copy, Default, PartialEq, ShaderType)]
pub struct AssemblyGroup {
    pub center: Vec3,
    pub first: u32,
    pub half_extent: Vec3,
    // The high bit requests fail-open traversal for a singular source transform.
    pub count: u32,
}

#[derive(Clone)]
struct Allocation {
    group: u32,
    members: Range<u32>,
}

pub(super) struct Hierarchy {
    allocations: HashMap<SceneInstance, Allocation>,
    ordinary: HashSet<SceneInstance>,
    free_groups: Vec<u32>,
    next_group: u32,
    members: RangeAllocator<u32>,
}
impl Default for Hierarchy {
    fn default() -> Self {
        Self {
            allocations: HashMap::default(),
            ordinary: HashSet::default(),
            free_groups: Vec::new(),
            next_group: 0,
            members: RangeAllocator::new(0..0),
        }
    }
}
impl Hierarchy {
    fn range(&mut self, count: u32) -> Range<u32> {
        if let Ok(range) = self.members.allocate_range(count) {
            return range;
        }
        let end = self.members.initial_range().end;
        self.members
            .grow_to(end.checked_add(end.max(count)).expect("hierarchy capacity"));
        self.members
            .allocate_range(count)
            .expect("grown hierarchy range")
    }
    fn allocate(&mut self, owner: SceneInstance, count: u32) -> Allocation {
        if let Some(old) = self.allocations.get(&owner) {
            if old.members.end - old.members.start >= count {
                return old.clone();
            }
        }
        let group = if let Some(old) = self.allocations.remove(&owner) {
            self.members.free_range(old.members);
            old.group
        } else {
            self.free_groups.pop().unwrap_or_else(|| {
                let group = self.next_group;
                self.next_group += 1;
                group
            })
        };
        let allocation = Allocation {
            group,
            members: self.range(count),
        };
        self.allocations.insert(owner, allocation.clone());
        allocation
    }
}

/// Group IDs and spans persist across sparse root changes. GPU queues refer to
/// them only within one frame; staged writes precede that frame's dispatches.
pub(super) fn update(manager: &mut InstanceManager, roots: &[Entity]) {
    update_ordinary(manager);
    let mut prototypes: HashMap<usize, Option<MeshletAabb>> = HashMap::default();
    for &root in roots {
        let owner = SceneInstance {
            entity: root,
            part: 1,
        };
        let Some(input) = manager.assembly_inputs.get(&root) else {
            remove(manager, owner);
            continue;
        };
        let (bounds, members) = root_data(manager, input, &mut prototypes);
        publish(manager, owner, bounds, &members);
    }
    manager.instance_upload_dirty |=
        !manager.group_dirty_indices.is_empty() || !manager.member_dirty_indices.is_empty();
    bevy_render::diagnostic::profile_value(
        "meshlet.hierarchy_roots",
        manager.hierarchy.allocations.len() as f64,
        "count",
    );
}
fn root_data(
    manager: &InstanceManager,
    input: &assembly_extract::AssemblyInput,
    prototypes: &mut HashMap<usize, Option<MeshletAabb>>,
) -> (Option<MeshletAabb>, Vec<u32>) {
    let members: Vec<_> = input
        .slots
        .iter()
        .filter(|slot| manager.slots.is_active(**slot))
        .map(|slot| slot.index)
        .collect();
    if !input.ready || members.is_empty() || members.len() != input.slots.len() {
        return (None, members);
    }
    let bounds = *prototypes
        .entry(input.parts.as_ptr() as usize)
        .or_insert_with(|| {
            let anchor_from_root = input.parts[0].transform.inverse();
            union_bounds(input.parts.iter().zip(&input.slots).map(|(part, slot)| {
                (
                    anchor_from_root * part.transform,
                    manager.instance_aabbs.get()[slot.index as usize],
                )
            }))
        });
    (bounds, members)
}
fn update_ordinary(manager: &mut InstanceManager) {
    let mut previous = core::mem::take(&mut manager.hierarchy.ordinary);
    for slot in manager.query_slots.clone() {
        if !manager.slots.is_active(slot) {
            continue;
        }
        let owner = SceneInstance {
            entity: manager.slots.entity(slot.index).unwrap(),
            part: 0,
        };
        previous.remove(&owner);
        manager.hierarchy.ordinary.insert(owner);
        publish(
            manager,
            owner,
            Some(manager.instance_aabbs.get()[slot.index as usize]),
            &[slot.index],
        );
    }
    for owner in previous {
        remove(manager, owner);
    }
}
fn remove(manager: &mut InstanceManager, owner: SceneInstance) {
    let Some(old) = manager.hierarchy.allocations.remove(&owner) else {
        return;
    };
    manager.hierarchy.members.free_range(old.members);
    manager.hierarchy.free_groups.push(old.group);
    manager.assembly_groups.get_mut()[old.group as usize].count = 0;
    manager.group_dirty_indices.push(old.group);
}
fn publish(
    manager: &mut InstanceManager,
    owner: SceneInstance,
    bounds: Option<MeshletAabb>,
    members: &[u32],
) {
    if members.is_empty() {
        remove(manager, owner);
        return;
    }
    let allocation = manager.hierarchy.allocate(owner, members.len() as u32);
    let group = AssemblyGroup {
        center: bounds.map_or(Vec3::ZERO, |b| b.center),
        half_extent: bounds.map_or(Vec3::ZERO, |b| b.half_extent),
        first: allocation.members.start,
        count: members.len() as u32 | if bounds.is_none() { 1 << 31 } else { 0 },
    };
    publish_group(manager, allocation.group, group);
    publish_members(manager, allocation.members.start as usize, members);
}
fn publish_group(manager: &mut InstanceManager, group_index: u32, group: AssemblyGroup) {
    let groups = manager.assembly_groups.get_mut();
    let index = group_index as usize;
    if groups.len() <= index {
        groups.resize(index + 1, AssemblyGroup::default());
    }
    if groups[index] != group {
        groups[index] = group;
        manager.group_dirty_indices.push(group_index);
    }
}
fn publish_members(manager: &mut InstanceManager, start: usize, members: &[u32]) {
    let data = manager.assembly_members.get_mut();
    let old_len = data.len();
    if data.len() < start + members.len() {
        data.resize(start + members.len(), 0);
    }
    for (offset, &value) in members.iter().enumerate() {
        let index = start + offset;
        if index >= old_len || data[index] != value {
            data[index] = value;
            manager.member_dirty_indices.push(index as u32);
        }
    }
}

#[cfg(test)]
#[path = "assembly_hierarchy_tests.rs"]
mod publication_tests;
fn union_bounds(parts: impl Iterator<Item = (Affine3A, MeshletAabb)>) -> Option<MeshletAabb> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for (transform, aabb) in parts {
        if !transform.is_finite() {
            return None;
        }
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let p = transform
                        .transform_point3(aabb.center + aabb.half_extent * Vec3::new(x, y, z));
                    if !p.is_finite() {
                        return None;
                    }
                    minimum = minimum.min(p);
                    maximum = maximum.max(p);
                }
            }
        }
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return None;
    }
    let center = (minimum + maximum) * 0.5;
    let epsilon = maximum.abs().max(minimum.abs()).max(Vec3::ONE) * (8.0 * f32::EPSILON);
    Some(MeshletAabb {
        center,
        half_extent: (maximum - minimum) * 0.5 + epsilon,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn root_bounds_enclose_rotated_scaled_parts_and_singular_fails_open() {
        let aabb = MeshletAabb {
            center: Vec3::new(2.0, -1.0, 4.0),
            half_extent: Vec3::new(1.0, 3.0, 2.0),
        };
        let t = Affine3A::from_scale_rotation_translation(
            Vec3::new(-2.0, 0.5, 3.0),
            bevy_math::Quat::from_rotation_z(0.8),
            Vec3::new(5.0, 7.0, -3.0),
        );
        let b = union_bounds([(t, aabb)].into_iter()).unwrap();
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let p = t.transform_point3(aabb.center + aabb.half_extent * Vec3::new(x, y, z));
                    assert!((p - b.center).abs().cmple(b.half_extent).all());
                }
            }
        }
        assert!(
            union_bounds([(Affine3A::from_scale(Vec3::ZERO).inverse(), aabb)].into_iter())
                .is_none()
        );
    }
}
