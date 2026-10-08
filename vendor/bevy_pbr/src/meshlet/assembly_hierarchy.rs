//! A conservative root above the existing per-part BVHs. This changes traversal,
//! never the selected source representation or the ray scene.
use super::*;
use bevy_math::{Affine3A, Vec3};
use bevy_render::render_resource::ShaderType;

#[derive(Clone, Copy, PartialEq, ShaderType)]
pub struct AssemblyGroup {
    pub center: Vec3,
    pub first: u32,
    pub half_extent: Vec3,
    // The high bit requests fail-open traversal for a singular source transform.
    pub count: u32,
}

pub(super) fn rebuild(manager: &mut InstanceManager) {
    let mut groups = Vec::new();
    let mut members = Vec::with_capacity(manager.scene_instance_count as usize);
    for slot in &manager.query_slots {
        if manager.slots.is_active(*slot) {
            singleton(manager, slot.index, &mut groups, &mut members);
        }
    }
    let mut roots: Vec<_> = manager.assembly_inputs.iter().collect();
    roots.sort_unstable_by_key(|(entity, _)| entity.to_bits());
    // Bounds depend only on immutable prototype geometry and transforms.
    let mut prototypes: HashMap<usize, Option<MeshletAabb>> = HashMap::default();
    for (_, input) in roots {
        if !input.ready {
            for slot in &input.slots {
                if manager.slots.is_active(*slot) {
                    singleton(manager, slot.index, &mut groups, &mut members);
                }
            }
            continue;
        }
        if input.slots.is_empty() {
            continue;
        }
        let key = input.parts.as_ptr() as usize;
        let bounds = *prototypes.entry(key).or_insert_with(|| {
            let anchor_from_root = input.parts[0].transform.inverse();
            union_bounds(input.parts.iter().zip(&input.slots).map(|(part, slot)| {
                (
                    anchor_from_root * part.transform,
                    manager.instance_aabbs.get()[slot.index as usize],
                )
            }))
        });
        let first = members.len() as u32;
        members.extend(input.slots.iter().map(|slot| slot.index));
        let count = input.slots.len() as u32;
        groups.push(AssemblyGroup {
            center: bounds.map_or(Vec3::ZERO, |b| b.center),
            first,
            half_extent: bounds.map_or(Vec3::ZERO, |b| b.half_extent),
            count: count | if bounds.is_none() { 1 << 31 } else { 0 },
        });
    }
    let old_groups = manager.assembly_groups.get();
    let dirty: Vec<_> = groups
        .iter()
        .enumerate()
        .filter(|(index, group)| old_groups.get(*index) != Some(*group))
        .map(|(index, _)| index as u32)
        .collect();
    manager.instance_upload_dirty |= old_groups.len() != groups.len();
    *manager.assembly_groups.get_mut() = groups;
    manager.group_dirty_indices.extend(dirty);
    manager
        .member_dirty_indices
        .extend(manager.assembly_members.set_indices(&members));
    manager.instance_upload_dirty |=
        !manager.group_dirty_indices.is_empty() || !manager.member_dirty_indices.is_empty();
    bevy_render::diagnostic::profile_value(
        "meshlet.hierarchy_roots",
        manager.assembly_groups.get().len() as f64,
        "count",
    );
}
fn singleton(
    manager: &InstanceManager,
    slot: u32,
    groups: &mut Vec<AssemblyGroup>,
    members: &mut Vec<u32>,
) {
    let aabb = manager.instance_aabbs.get()[slot as usize];
    groups.push(AssemblyGroup {
        center: aabb.center,
        half_extent: aabb.half_extent,
        first: members.len() as u32,
        count: 1,
    });
    members.push(slot);
}
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
