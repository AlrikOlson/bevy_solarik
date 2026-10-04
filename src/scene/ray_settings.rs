use bevy_ecs::resource::Resource;
use bevy_render::extract_resource::ExtractResource;

/// Scene-wide traversal range and optional coordinate-relative secondary-ray floor.
///
/// Increasing the range does not increase mesh coordinate precision. A nonzero
/// relative floor can suppress self-intersections in large-coordinate scenes,
/// but also omits nearby occluders. Reset view histories after changing settings.
#[derive(Resource, ExtractResource, Clone, Copy, Debug)]
pub struct SolarikRaySettings {
    max_distance: f32,
    relative_min_distance: f32,
}

impl SolarikRaySettings {
    /// Validate a maximum in [0.001, 1e16] render units and a relative floor
    /// in [0, 0.001]. The floor multiplies the largest absolute origin coordinate.
    pub fn new(max_distance: f32, relative_min_distance: f32) -> Result<Self, &'static str> {
        if !max_distance.is_finite() || !(0.001..=1.0e16).contains(&max_distance) {
            return Err("ray maximum must be finite and within [0.001, 1e16]");
        }
        if !relative_min_distance.is_finite() || !(0.0..=0.001).contains(&relative_min_distance) {
            return Err("relative ray minimum must be finite and within [0, 0.001]");
        }
        Ok(Self {
            max_distance,
            relative_min_distance,
        })
    }

    /// Maximum traversal distance in the consumer's render units.
    pub fn max_distance(self) -> f32 {
        self.max_distance
    }

    /// Multiplier applied to the largest absolute ray-origin coordinate.
    pub fn relative_min_distance(self) -> f32 {
        self.relative_min_distance
    }
}

impl Default for SolarikRaySettings {
    fn default() -> Self {
        Self {
            max_distance: 100_000.0,
            relative_min_distance: 0.0,
        }
    }
}
