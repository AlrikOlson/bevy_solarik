//! Conservative empty region for geometry edits, in render-origin coordinates.
use bevy_math::{Affine3A, Vec3};
use bevy_mesh::{Mesh, VertexAttributeValues};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// CPU record of what this view actually rendered, not just what was extracted.
#[derive(Default)]
pub(crate) struct ViewHistory {
    scene_generation: AtomicU64,
    pending_full_reset: AtomicBool,
}
impl ViewHistory {
    pub(crate) fn invalidate(&self) {
        self.pending_full_reset.store(true, Ordering::Relaxed);
    }
    pub(crate) fn stable_radius(&self, generation: u64, published_radius: f32) -> f32 {
        let previous = self.scene_generation.load(Ordering::Relaxed);
        if previous == 0 || self.pending_full_reset.load(Ordering::Relaxed) {
            0.0
        } else if previous == generation {
            f32::INFINITY
        } else if previous.checked_add(1) == Some(generation) {
            published_radius
        } else {
            // More than one event was missed; no unbounded event log or
            // assumption that the last event subsumes all preceding changes.
            0.0
        }
    }
    pub(crate) fn rendered(&self, generation: u64) {
        self.scene_generation.store(generation, Ordering::Relaxed);
        self.pending_full_reset.store(false, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Bounds {
    min: Vec3,
    max: Vec3,
}
impl Bounds {
    pub(super) fn from_mesh(mesh: &Mesh) -> Option<Self> {
        let VertexAttributeValues::Float32x3(points) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)?
        else {
            return None;
        };
        let mut bounds = Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };
        for point in points {
            let point = Vec3::from_array(*point);
            if !point.is_finite() {
                return None;
            }
            bounds.min = bounds.min.min(point);
            bounds.max = bounds.max.max(point);
        }
        (!points.is_empty()).then_some(bounds)
    }

    /// Radius of an origin-centred sphere that cannot contain this instance.
    /// Transform all corners, including shear/nonuniform scale; shrink the
    /// lower bound for f32 rounding instead of overstating the stable region.
    pub(super) fn stable_radius(self, transform: Affine3A) -> f32 {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for x in [self.min.x, self.max.x] {
            for y in [self.min.y, self.max.y] {
                for z in [self.min.z, self.max.z] {
                    let point = transform.transform_point3(Vec3::new(x, y, z));
                    if !point.is_finite() {
                        return 0.0;
                    }
                    min = min.min(point);
                    max = max.max(point);
                }
            }
        }
        let nearest = min.max(Vec3::ZERO) + max.min(Vec3::ZERO);
        let radius = nearest.length();
        if !radius.is_finite() {
            return 0.0;
        }
        (radius * (1.0 - 1.0e-5) - 0.01).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::{Mat3, Quat};

    #[test]
    fn views_consume_events_only_after_rendering_and_keep_cuts_pending() {
        let view = ViewHistory::default();
        let other_view = ViewHistory::default();
        assert_eq!(
            view.stable_radius(1, 1000.0),
            0.0,
            "new view has no valid history"
        );
        view.rendered(1);
        assert!(
            view.stable_radius(1, 0.0).is_infinite(),
            "event already consumed"
        );
        assert_eq!(view.stable_radius(2, 1000.0), 1000.0, "one bounded change");
        assert_eq!(
            view.stable_radius(2, 1000.0),
            1000.0,
            "skipped node cannot consume"
        );
        assert_eq!(
            view.stable_radius(3, 2000.0),
            0.0,
            "empty/pipeline pause missed two events"
        );
        assert_eq!(
            other_view.stable_radius(3, 2000.0),
            0.0,
            "views are independent"
        );
        view.rendered(3);
        view.invalidate();
        assert_eq!(
            view.stable_radius(3, f32::INFINITY),
            0.0,
            "explicit cut is pending"
        );
        assert_eq!(
            view.stable_radius(3, f32::INFINITY),
            0.0,
            "extraction cannot consume cut"
        );
        view.rendered(3);
        assert!(view.stable_radius(3, 0.0).is_infinite());
        view.rendered(u64::MAX);
        assert_eq!(
            view.stable_radius(1, 1000.0),
            0.0,
            "generation wrap fails closed"
        );
    }

    #[test]
    fn bounds_cover_scale_rotation_shear_and_unknown_inputs() {
        let bounds = Bounds {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        assert_eq!(bounds.stable_radius(Affine3A::IDENTITY), 0.0);
        let far = Affine3A::from_translation(Vec3::X * 1000.0);
        assert!(bounds.stable_radius(far) > 998.0);
        assert!(bounds.stable_radius(far) < 999.0);
        let transform = Affine3A::from_scale_rotation_translation(
            Vec3::new(20.0, 2.0, 3.0),
            Quat::from_rotation_z(0.8),
            Vec3::X * 100.0,
        );
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    assert!(
                        bounds.stable_radius(transform)
                            < transform.transform_point3(Vec3::new(x, y, z)).length()
                    );
                }
            }
        }
        let shear = Affine3A::from_mat3_translation(
            Mat3::from_cols(Vec3::X, Vec3::new(2.0, 1.0, 0.0), Vec3::Z),
            Vec3::X * 100.0,
        );
        assert!(bounds.stable_radius(shear) < 97.0);
        assert_eq!(
            bounds.stable_radius(Affine3A::from_translation(Vec3::splat(f32::NAN))),
            0.0
        );
    }
}
