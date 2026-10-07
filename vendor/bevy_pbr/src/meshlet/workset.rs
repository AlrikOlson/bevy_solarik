//! Conservative perspective queue admission, independent of occlusion and frustum culling.
use super::asset::{BvhNode, MeshletMesh};
use bevy_math::Mat4;

/// Sorted distance thresholds for a shared prototype's possible queue writes.
/// This is an upper bound, not a prediction of visible triangles or GPU time.
#[derive(Clone, Debug)]
pub struct MeshletWorkset {
    thresholds: Vec<(f64, u64)>,
    full: u64,
}

impl MeshletMesh {
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
    thresholds.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut sum = 0;
    for (_, cost) in thresholds.iter_mut().rev() {
        sum += *cost;
        *cost = sum;
    }
    Some(MeshletWorkset {
        thresholds,
        full: sum + 2,
    })
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
            // Half-pixel headroom; max(1,scale) also bounds old unscaled shaders.
            let limit = extent * scale + 2.0 * error as f64 * (part * scale).max(1.0) * focal;
            result.push((
                limit,
                if count == u8::MAX {
                    2
                } else {
                    u64::from(count)
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
        assert_eq!(p.slots(0.0), 32);
        assert_eq!(p.slots(f64::NAN), 32);
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
                                u64::from(n.child_counts[i])
                            };
                        }
                    }
                    assert!(profile.slots(distance) >= actual);
                }
            }
        }
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
