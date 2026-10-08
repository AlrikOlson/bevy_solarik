//! Fixed-cost conservative spatial unions of old and new occluder bounds.
use super::Bounds;
use bevy_math::{Affine3A, Vec3, Vec4};

#[derive(Default)]
pub(crate) struct Regions {
    bounds: [Option<Bounds>; 8],
    pub global: bool,
}
impl Regions {
    pub fn clear(&mut self, global: bool) {
        self.bounds.fill(None);
        self.global = global;
    }
    pub fn record(&mut self, bounds: Option<Bounds>, transform: Affine3A) {
        let Some(bounds) = bounds.and_then(|b| b.transformed(transform)) else {
            self.global = true;
            return;
        };
        let center = (bounds.min + bounds.max) * 0.5;
        let index = usize::from(center.x >= 0.0)
            | (usize::from(center.y >= 0.0) << 1)
            | (usize::from(center.z >= 0.0) << 2);
        self.bounds[index] = Some(self.bounds[index].map_or(bounds, |old| Bounds {
            min: old.min.min(bounds.min),
            max: old.max.max(bounds.max),
        }));
    }
    pub fn packed(&self) -> Vec<Vec4> {
        self.bounds
            .iter()
            .flat_map(|bounds| match bounds {
                Some(b) => [b.min.extend(1.0), b.max.extend(1.0)],
                None => [Vec4::ZERO; 2],
            })
            .collect()
    }
}
impl Bounds {
    pub(super) fn transformed(self, transform: Affine3A) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for x in [self.min.x, self.max.x] {
            for y in [self.min.y, self.max.y] {
                for z in [self.min.z, self.max.z] {
                    let p = transform.transform_point3(Vec3::new(x, y, z));
                    if !p.is_finite() {
                        return None;
                    }
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
        let margin = min.abs().max(max.abs()) * 1e-5 + Vec3::splat(0.01);
        Some(Self {
            min: min - margin,
            max: max + margin,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unions_preserve_both_ends_and_unknown_inputs_fail_closed() {
        let bounds = Bounds {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let mut regions = Regions::default();
        regions.record(Some(bounds), Affine3A::from_translation(Vec3::X * 1000.0));
        regions.record(Some(bounds), Affine3A::from_translation(Vec3::X * -1000.0));
        assert_eq!(regions.bounds.iter().flatten().count(), 2);
        assert!(regions.bounds.iter().flatten().any(|b| b.min.x < -1001.0));
        assert!(regions.bounds.iter().flatten().any(|b| b.max.x > 1001.0));
        regions.record(None, Affine3A::IDENTITY);
        assert!(regions.global);
        regions.clear(false);
        assert!(!regions.global);
        assert!(regions.packed().iter().all(|v| *v == Vec4::ZERO));
    }
}
