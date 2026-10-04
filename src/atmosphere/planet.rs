use bevy_asset::Handle;
use bevy_ecs::resource::Resource;
use bevy_image::Image;
use bevy_math::Vec3;
use bevy_render::extract_resource::ExtractResource;

/// Thinnest cloud shell the view march is expected to resolve, metres.
const MIN_CLOUD_SHELL: f32 = 500.0;

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
    /// Cloud control, 0..1; zero disables clouds. Without a weather map it is
    /// the cover of the built-in noise field; with one it only enables the
    /// layer, because the map defines cover.
    pub cloud_coverage: f32,
    /// Maximum cloud extinction per metre, 0..0.2. Liquid water clouds are
    /// roughly 0.01 to 0.1 per metre.
    pub cloud_extinction: f32,
    /// Lowest cloud base above the surface, metres.
    pub cloud_base: f32,
    /// Highest cloud top above the surface, metres; inside the atmosphere.
    pub cloud_top: f32,
    /// Optional cube weather map, sampled with the body-fixed unit direction
    /// using the standard cube face convention. R: cover in 0..1. G: local
    /// cloud-top height as a fraction between `cloud_base` and `cloud_top`.
    /// B: extinction scale in 0..1. The application owns its contents. If it
    /// has mips, a footprint wider than a texel reads the level that averages
    /// it; averaging cover is exact under the independent-column blend.
    pub weather: Option<Handle<Image>>,
    /// Sub-texel detail for a weather map, 0..1: the cover change that one
    /// standard deviation of the first octave below the map's texel size
    /// causes at a cloud edge. Zero disables it.
    pub cloud_detail: f32,
    /// Stable recipe seed for the built-in noise field, exactly representable on the GPU.
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
            || !(0.0..=0.2).contains(&self.cloud_extinction)
        {
            return Err("invalid planetary atmosphere geometry or cloud controls");
        }
        if !self.cloud_base.is_finite()
            || !self.cloud_top.is_finite()
            || self.cloud_base < 0.0
            || self.cloud_top > self.height
            || self.cloud_top - self.cloud_base < MIN_CLOUD_SHELL
            || !self.cloud_detail.is_finite()
            || !(0.0..=1.0).contains(&self.cloud_detail)
        {
            return Err("cloud shell must lie inside the atmosphere and span at least 500 m");
        }
        Ok(())
    }
}
