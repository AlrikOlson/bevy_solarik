use bevy_ecs::resource::Resource;
use bevy_math::Vec3;
use bevy_render::extract_resource::ExtractResource;

/// Optional spherical-world geometry for the atmosphere. All distances are metres.
///
/// Coordinates and source directions share a planet-centred, body-fixed basis.
/// The application updates the observer after motion and removes this resource
/// to restore the original local-ground atmosphere. One observer is supported.
#[derive(Resource, ExtractResource, Debug, Clone, PartialEq)]
pub struct PlanetaryAtmosphere {
    pub radius: f32,
    pub height: f32,
    pub observer: Vec3,
    /// Fictional static weather coverage, 0..1; zero disables clouds.
    pub cloud_coverage: f32,
    /// Maximum cloud extinction per metre, 0..0.002.
    pub cloud_extinction: f32,
    /// Stable recipe seed, exactly representable on the GPU.
    pub seed: u16,
    /// Optional spherical source occluder, in the same body-fixed metres.
    pub occluder: Vec3,
    pub occluder_radius: f32,
}

impl PlanetaryAtmosphere {
    /// Validate finite geometry before generating lookup fields.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.radius.is_finite()
            || !(100_000.0..=100_000_000.0).contains(&self.radius)
            || !self.height.is_finite()
            || !(10_000.0..=200_000.0).contains(&self.height)
            || !self.observer.is_finite()
            || self.observer.length() <= self.radius
            || self.observer.length() > 1.0e9
            || !self.occluder.is_finite()
            || self.occluder.length() > 1.0e9
            || !self.occluder_radius.is_finite()
            || !(0.0..=1.0e8).contains(&self.occluder_radius)
            || !self.cloud_coverage.is_finite()
            || !(0.0..=1.0).contains(&self.cloud_coverage)
            || !self.cloud_extinction.is_finite()
            || !(0.0..=0.002).contains(&self.cloud_extinction)
        {
            return Err("invalid planetary atmosphere geometry or cloud controls");
        }
        Ok(())
    }
}
