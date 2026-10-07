//! Conservative perspective queue admission, independent of occlusion and frustum culling.
use super::asset::{BvhNode, MeshletMesh};
use bevy_math::Mat4;

/// Sorted distance thresholds for a shared prototype's possible queue writes.
/// This is an upper bound, not a prediction of visible triangles or GPU time.
#[derive(Clone, Debug)]
pub struct MeshletWorkset {
    thresholds: Vec<(f64, u64)>,
    lower_thresholds: Vec<(f64, u64)>,
    full: u64,
    root_radius: f64,
}

impl MeshletMesh {
    /// Bound candidates after early own-LOD rejection for a known scale range.
    /// The near plane is required because the shader clamps perspective distance.
    pub fn workset_profile_range(
        &self,
        root_from_mesh: Mat4,
        minimum_scale: f64,
        maximum_scale: f64,
        focal_pixels: f64,
        near_plane: f64,
    ) -> Option<MeshletWorkset> {
        let mut result = self.workset_profile(root_from_mesh, maximum_scale, focal_pixels)?;
        if !minimum_scale.is_finite()
            || minimum_scale <= 0.0
            || minimum_scale > maximum_scale
            || !near_plane.is_finite()
            || near_plane <= 0.0
        {
            return None;
        }
        let part = root_from_mesh
            .x_axis
            .truncate()
            .length()
            .max(root_from_mesh.y_axis.truncate().length())
            .max(root_from_mesh.z_axis.truncate().length()) as f64;
        for node in self.bvh.iter() {
            for i in 0..8 {
                let count = node.child_counts[i];
                if count == 0 || count == u8::MAX {
                    continue;
                }
                let parent = node.lod_bounds[i];
                let extent = root_from_mesh.transform_point3(parent.center).length() as f64
                    + parent.radius as f64 * part;
                let upper = (extent * maximum_scale
                    + node.aabbs[i].error as f64 * part * maximum_scale * focal_pixels)
                    * ROUNDING_GUARD;
                for child in 0..u32::from(count) {
                    let data = self
                        .meshlet_cull_data
                        .get((node.aabbs[i].child_offset + child) as usize)?;
                    let sphere = data.lod_group_sphere;
                    if !sphere.center.is_finite()
                        || !sphere.radius.is_finite()
                        || sphere.radius < 0.0
                        || !data.aabb.error.is_finite()
                        || data.aabb.error < 0.0
                    {
                        return None;
                    }
                    let extent = root_from_mesh.transform_point3(sphere.center).length() as f64
                        + sphere.radius as f64 * part;
                    let error_reach = data.aabb.error as f64 * part * minimum_scale * focal_pixels
                        / ROUNDING_GUARD;
                    // A very large near plane can make even nearby errors imperceptible.
                    let lower = if error_reach <= near_plane * ROUNDING_GUARD {
                        0.0
                    } else {
                        (error_reach - extent * maximum_scale * ROUNDING_GUARD)
                            .max(0.0)
                            .min(upper)
                    };
                    result.lower_thresholds.push((lower, 2));
                }
            }
        }
        suffix_costs(&mut result.lower_thresholds);
        Some(result)
    }

    /// Bound all perspective views whose focal length and uniform instance scale
    /// do not exceed these values. `root_from_mesh` includes the part transform.
    /// Orthographic views require the all-LOD bound instead. Rebuild the profile
    /// when the projection, resolution, part transform or maximum scale changes.
    /// Returns `None` for invalid inputs. No frustum/occlusion savings are assumed.
    pub fn workset_profile(
        &self,
        root_from_mesh: Mat4,
        maximum_scale: f64,
        focal_pixels: f64,
    ) -> Option<MeshletWorkset> {
        profile(&self.bvh, root_from_mesh, maximum_scale, focal_pixels)
    }
}

impl MeshletWorkset {
    /// Encloses every active BVH AABB corner after part transform and maximum
    /// instance scale. A separating frustum plane therefore rejects all children.
    pub fn root_radius(&self) -> f64 {
        self.root_radius
    }
    /// Bound all cameras within a root-distance interval, including caller motion error.
    pub fn slots_in_range(&self, minimum: f64, maximum: f64) -> u64 {
        if !minimum.is_finite() || !maximum.is_finite() || minimum < 0.0 || maximum < minimum {
            return self.full;
        }
        let first = self
            .lower_thresholds
            .partition_point(|(limit, _)| *limit <= maximum);
        self.slots(minimum)
            - self
                .lower_thresholds
                .get(first)
                .map_or(0, |(_, cost)| *cost)
    }
    /// Upper bound for any camera at least `distance_metres` from the instance
    /// root. The caller must account for camera-relative conversion error and
    /// every render view. Invalid distances conservatively use every LOD.
    pub fn slots(&self, distance_metres: f64) -> u64 {
        if !distance_metres.is_finite() || distance_metres < 0.0 {
            return self.full;
        }
        let first = self
            .thresholds
            .partition_point(|(limit, _)| *limit < distance_metres);
        2 + self.thresholds.get(first).map_or(0, |(_, slots)| *slots)
    }

    /// Bound without any projection or distance assumptions.
    pub fn all_lod_slots(&self) -> u64 {
        self.full
    }
}

fn profile(nodes: &[BvhNode], transform: Mat4, scale: f64, focal: f64) -> Option<MeshletWorkset> {
    if !transform.is_finite()
        || transform.row(3) != bevy_math::Vec4::W
        || !scale.is_finite()
        || scale <= 0.0
        || !focal.is_finite()
        || focal <= 0.0
    {
        return None;
    }
    let part_scale = transform
        .x_axis
        .truncate()
        .length()
        .max(transform.y_axis.truncate().length())
        .max(transform.z_axis.truncate().length()) as f64;
    if !part_scale.is_finite() {
        return None;
    }
    let mut thresholds = thresholds(nodes, transform, part_scale, scale, focal)?;
    let sum = suffix_costs(&mut thresholds);
    Some(MeshletWorkset {
        thresholds,
        lower_thresholds: Vec::new(),
        full: sum + 2,
        root_radius: root_radius(nodes, transform, scale)?,
    })
}

const ROUNDING_GUARD: f64 = 1.0 + 64.0 * f32::EPSILON as f64;

fn root_radius(nodes: &[BvhNode], transform: Mat4, scale: f64) -> Option<f64> {
    let mut radius = 0.0_f64;
    for node in nodes {
        for (aabb, count) in node.aabbs.iter().zip(node.child_counts) {
            if count == 0 {
                continue;
            }
            if !aabb.center.is_finite()
                || !aabb.half_extent.is_finite()
                || aabb.half_extent.min_element() < 0.0
            {
                return None;
            }
            for corner in 0..8 {
                let sign = bevy_math::Vec3::new(
                    if corner & 1 == 0 { -1.0 } else { 1.0 },
                    if corner & 2 == 0 { -1.0 } else { 1.0 },
                    if corner & 4 == 0 { -1.0 } else { 1.0 },
                );
                let point = transform.transform_point3(aabb.center + sign * aabb.half_extent);
                if !point.is_finite() {
                    return None;
                }
                radius = radius.max(point.as_dvec3().length() * scale * ROUNDING_GUARD);
            }
        }
    }
    Some(radius)
}

fn suffix_costs(thresholds: &mut [(f64, u64)]) -> u64 {
    thresholds.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut sum = 0;
    for (_, cost) in thresholds.iter_mut().rev() {
        sum += *cost;
        *cost = sum;
    }
    sum
}

fn thresholds(
    nodes: &[BvhNode],
    transform: Mat4,
    part: f64,
    scale: f64,
    focal: f64,
) -> Option<Vec<(f64, u64)>> {
    let mut result = Vec::new();
    for node in nodes {
        for i in 0..8 {
            let count = node.child_counts[i];
            if count == 0 {
                continue;
            }
            let sphere = node.lod_bounds[i];
            let error = node.aabbs[i].error;
            if !sphere.center.is_finite()
                || !sphere.radius.is_finite()
                || sphere.radius < 0.0
                || !error.is_finite()
                || error < 0.0
            {
                return None;
            }
            let extent = transform.transform_point3(sphere.center).length() as f64
                + sphere.radius as f64 * part;
            if !extent.is_finite() {
                return None;
            }
            // The production shader rejects errors below ONE pixel. Inflate the
            // full threshold for f32 matrix/norm/projection rounding, rather than
            // charging every instance for detail at twice its visible distance.
            // Camera-relative conversion/motion error remains the caller's duty.
            let limit = (extent * scale + error as f64 * part * scale * focal) * ROUNDING_GUARD;
            result.push((
                limit,
                if count == u8::MAX {
                    2
                } else {
                    // Early inputs remain live while occluded clusters append
                    // second-pass copies to the opposite end of this queue.
                    2 * u64::from(count)
                },
            ));
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn interval_mesh() -> MeshletMesh {
        use super::super::asset::{
            MeshletAabb, MeshletAabbErrorOffset, MeshletBoundingSphere, MeshletCullData,
        };
        let sphere = MeshletBoundingSphere {
            center: Vec3::new(1.0, 2.0, -1.0),
            radius: 2.0,
        };
        let mut node = BvhNode::default();
        node.child_counts[0] = 3;
        node.lod_bounds[0] = sphere;
        node.aabbs[0].error = 0.2;
        MeshletMesh {
            vertex_positions: Vec::new().into(),
            vertex_normals: Vec::new().into(),
            vertex_uvs: Vec::new().into(),
            vertex_tangents: Vec::new().into(),
            indices: Vec::new().into(),
            meshlets: Vec::new().into(),
            bvh: vec![node].into(),
            aabb: MeshletAabb::default(),
            bvh_depth: 1,
            meshlet_cull_data: [0.01, 0.05, 0.15]
                .map(|error| MeshletCullData {
                    aabb: MeshletAabbErrorOffset {
                        error,
                        ..Default::default()
                    },
                    lod_group_sphere: sphere,
                })
                .into(),
        }
    }

    #[test]
    fn root_bound_encloses_transformed_aabb_corners_at_every_instance_rotation() {
        use bevy_math::Quat;
        let mut mesh = interval_mesh();
        let mut node = mesh.bvh[0];
        node.aabbs[0].center = Vec3::new(2.0, 8.0, -1.0);
        node.aabbs[0].half_extent = Vec3::new(4.0, 6.0, 3.0);
        mesh.bvh = vec![node].into();
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 3.0),
            Quat::from_rotation_z(0.7),
            Vec3::X * 3.0,
        );
        let profile = mesh.workset_profile(transform, 1.15, 1000.0).unwrap();
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_y(2.7),
            Quat::from_rotation_x(1.4),
        ] {
            for corner in 0..8 {
                let sign = Vec3::new(
                    if corner & 1 == 0 { -1.0 } else { 1.0 },
                    if corner & 2 == 0 { -1.0 } else { 1.0 },
                    if corner & 4 == 0 { -1.0 } else { 1.0 },
                );
                let point = rotation
                    * transform
                        .transform_point3(node.aabbs[0].center + sign * node.aabbs[0].half_extent)
                    * 1.15;
                assert!(point.length() as f64 <= profile.root_radius());
            }
        }
        node.aabbs[0].half_extent.x = f32::NAN;
        mesh.bvh = vec![node].into();
        assert!(mesh.workset_profile(transform, 1.15, 1000.0).is_none());
    }

    #[test]
    fn interval_bound_covers_both_passes_across_scale_direction_motion_and_near_plane() {
        let mesh = interval_mesh();
        for near in [0.01_f64, 1000.0] {
            let p = mesh
                .workset_profile_range(Mat4::IDENTITY, 0.85, 1.15, 1000.0, near)
                .unwrap();
            for scale in [0.85, 1.0, 1.15] {
                for direction in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::ONE.normalize()] {
                    for distance in [0.0_f64, 5.0, 10.0, 40.0, 80.0, 150.0, 220.0, 300.0] {
                        let camera = direction.as_dvec3() * distance;
                        let sphere = mesh.bvh[0].lod_bounds[0];
                        let d = ((sphere.center.as_dvec3() * scale).distance(camera)
                            - sphere.radius as f64 * scale)
                            .max(near);
                        let mut writes = 2;
                        if 0.2_f32 as f64 * scale * 1000.0 / d >= 1.0 {
                            writes += 2 * mesh
                                .meshlet_cull_data
                                .iter()
                                .filter(|c| c.aabb.error as f64 * scale * 1000.0 / d < 1.0)
                                .count() as u64;
                        }
                        assert!(
                            p.slots_in_range((distance - 2.5).max(0.0), distance + 2.5) >= writes
                        );
                    }
                }
            }
        }
        let p = mesh
            .workset_profile_range(Mat4::IDENTITY, 0.85, 1.15, 1000.0, 0.01)
            .unwrap();
        assert!(p.slots_in_range(10.0, 10.0) < p.slots(10.0));
        assert_eq!(p.slots_in_range(f64::NAN, 1.0), p.all_lod_slots());
        assert_eq!(p.slots_in_range(10.0, 1.0), p.all_lod_slots());
        assert!(
            mesh.workset_profile_range(Mat4::IDENTITY, 2.0, 1.0, 1000.0, 0.01)
                .is_none()
        );
    }

    #[test]
    fn occluded_early_candidates_need_space_for_their_deferred_copies() {
        let mut n = BvhNode::default();
        n.child_counts[0] = 128;
        n.aabbs[0].error = 1.0;
        n.lod_bounds[0].radius = 1.0;
        let p = profile(&[n], Mat4::IDENTITY, 1.0, 1000.0).unwrap();
        // First-pass cluster threads retain 128 input slots while appending
        // 128 occluded candidates at the opposite end for the second pass.
        assert!(p.slots(10.0) >= 128 + 128);
    }

    fn node() -> BvhNode {
        let mut node = BvhNode::default();
        for i in 0..8 {
            node.child_counts[i] = if i == 0 { u8::MAX } else { i as u8 };
            node.aabbs[i].error = (i + 1) as f32 * 0.01;
            node.lod_bounds[i].center = Vec3::new(i as f32, 1.0, -2.0);
            node.lod_bounds[i].radius = 0.5;
        }
        node
    }

    #[test]
    fn distance_reduces_bound_and_invalid_queries_keep_all_lods() {
        let p = profile(&[node()], Mat4::IDENTITY, 1.0, 1000.0).expect("valid profile");
        assert_eq!(p.slots(0.0), 60);
        assert_eq!(p.slots(f64::NAN), 60);
        assert_eq!(p.slots(1e6), 2);
        assert!(p.slots(100.0) < p.slots(0.0));
        assert!(profile(&[node()], Mat4::IDENTITY, 0.0, 1000.0).is_none());
    }

    #[test]
    fn bound_dominates_shader_queue_writes_for_directions_scales_and_offsets() {
        let n = node();
        let transform = Mat4::from_translation(Vec3::new(2.0, -3.0, 4.0));
        let profile = profile(&[n], transform, 2.0, 1500.0).expect("valid profile");
        for scale in [0.25_f64, 1.0, 2.0] {
            for direction in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE.normalize()] {
                for distance in [0.0, 10.0, 25.0, 100.0, 250.0, 1000.0] {
                    let camera = direction.as_dvec3() * distance;
                    let mut actual = 2;
                    for i in 0..8 {
                        let sphere = n.lod_bounds[i];
                        let centre = transform.transform_point3(sphere.center).as_dvec3() * scale;
                        let d = (centre.distance(camera) - sphere.radius as f64 * scale).max(0.1);
                        if n.aabbs[i].error as f64 * scale / d * 1500.0 >= 1.0 {
                            actual += if n.child_counts[i] == u8::MAX {
                                2
                            } else {
                                2 * u64::from(n.child_counts[i])
                            };
                        }
                    }
                    assert!(profile.slots(distance) >= actual);
                }
            }
        }
    }

    #[test]
    fn admission_releases_detail_after_the_shader_one_pixel_threshold() {
        let mut n = BvhNode::default();
        n.child_counts[0] = 7;
        n.aabbs[0].error = 0.1;
        // A centred sphere with radius 1 at scale 2: threshold = 2 + .2*1000.
        n.lod_bounds[0].radius = 1.0;
        let p = profile(&[n], Mat4::IDENTITY, 2.0, 1000.0).unwrap();
        assert_eq!(p.slots(201.99), 16);
        assert_eq!(p.slots(202.01), 2);
    }

    #[test]
    fn malformed_profiles_fail_closed_but_terminal_errors_are_valid() {
        let mut n = node();
        n.aabbs[0].error = f32::MAX;
        assert!(profile(&[n], Mat4::IDENTITY, 1.0, 1000.0).is_some());
        n.lod_bounds[0].radius = -1.0;
        assert!(profile(&[n], Mat4::IDENTITY, 1.0, 1000.0).is_none());
        n.lod_bounds[0].radius = f32::NAN;
        assert!(profile(&[n], Mat4::IDENTITY, 1.0, 1000.0).is_none());
        assert!(
            profile(
                &[node()],
                Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0),
                1.0,
                1000.0
            )
            .is_none()
        );
    }
}
