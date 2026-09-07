//! Equivalent exit-aperture radiance of a collimator, without internal lens tracing.

use bevy_app::{App, Plugin};
use bevy_asset::{AssetId, Handle};
use bevy_ecs::{component::Component, resource::Resource};
use bevy_math::{Vec3, Vec4};
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::HashMap;
use bevy_render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy_shader::load_shader_library;

/// A circular virtual image at infinity. Axis points from aperture toward eye.
/// The soft outer 10% avoids a discontinuity; it never widens the stated angular radius.
#[derive(Debug, Clone, Copy)]
pub struct CollimatedEmission {
    axis: Vec3,
    tan_radius: f32,
}

impl CollimatedEmission {
    #[must_use]
    pub fn new(axis: Vec3, angular_radius_radians: f32) -> Option<Self> {
        if !axis.is_finite()
            || !angular_radius_radians.is_finite()
            || !(0.0..0.1).contains(&angular_radius_radians)
            || angular_radius_radians == 0.0
        {
            return None;
        }
        Some(Self {
            axis: axis.try_normalize()?,
            tan_radius: angular_radius_radians.tan(),
        })
    }

    /// xyz: world emission axis; w: tangent of angular radius. Zero disables.
    #[must_use]
    pub fn gpu(self) -> Vec4 {
        self.axis.extend(self.tan_radius)
    }

    #[must_use]
    pub fn weight(self, outgoing: Vec3) -> f32 {
        let forward = outgoing.dot(self.axis);
        if forward <= 0.0 {
            return 0.0;
        }
        let radius = outgoing.cross(self.axis).length() / forward;
        let t = ((radius / self.tan_radius - 0.9) / 0.1).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    /// Luminance × area × integral cos(theta) dOmega, in lumens.
    /// Bounds the softened disk by a uniform cone. Use f64 for tiny angles.
    #[must_use]
    pub fn flux_upper_bound(self, luminance: f64, area: f64) -> f64 {
        let t = f64::from(self.tan_radius);
        luminance * area * core::f64::consts::PI * t * t / (1.0 + t * t)
    }
}

/// Opt-in emission profiles. Callers own handle lifetime and remove stale entries.
/// `StandardMaterial` emission is peak radiance; ordinary materials are unchanged.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct CollimatedMaterials(pub HashMap<AssetId<StandardMaterial>, CollimatedEmission>);

/// Ray material for a mesh rendered with a custom raster material.
#[derive(Component, Clone)]
pub struct RaytracingMaterial3d(pub Handle<StandardMaterial>);

/// Registers the shader library in raster-only apps too.
pub struct CollimatedEmissionPlugin;
impl Plugin for CollimatedEmissionPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "collimated.wgsl");
        app.init_resource::<CollimatedMaterials>();
        app.add_plugins(ExtractResourcePlugin::<CollimatedMaterials>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    #[test]
    fn cone_is_directional_and_resolves_small_angles_without_cosine_cancellation() {
        let cone = CollimatedEmission::new(Vec3::Z, 4.0_f32.to_radians() / 120.0).unwrap();
        assert_eq!(cone.weight(Vec3::Z), 1.0);
        assert_eq!(cone.weight(-Vec3::Z), 0.0);
        assert_eq!(cone.weight(Vec3::X), 0.0);
        assert_eq!(cone.weight(Vec3::new(0.002, 0.0, 1.0).normalize()), 0.0);
        assert!(cone.weight(Vec3::new(0.0001, 0.0, 1.0).normalize()) > 0.99);
    }

    #[test]
    fn malformed_optics_are_rejected() {
        assert!(CollimatedEmission::new(Vec3::ZERO, 4.0).is_none());
        assert!(CollimatedEmission::new(Vec3::Z, f32::NAN).is_none());
        assert!(CollimatedEmission::new(Vec3::Z, 0.0).is_none());
        assert!(CollimatedEmission::new(Vec3::Z, 10800.0).is_none());
    }

    #[test]
    fn four_moa_aperture_has_microlumen_scale_power_not_a_lamp() {
        let cone = CollimatedEmission::new(Vec3::Z, 4.0_f32.to_radians() / 120.0).unwrap();
        let flux = cone.flux_upper_bound(1000.0, 0.048 * 0.034);
        assert!(flux > 1e-6 && flux < 2e-6, "{flux}");
    }

    #[test]
    fn transverse_eye_motion_moves_the_aperture_intersection_not_the_aim_direction() {
        let cone = CollimatedEmission::new(Vec3::Z, 4.0_f32.to_radians() / 120.0).unwrap();
        for eye in [Vec3::new(0.0, 0.0, 0.25), Vec3::new(0.012, 0.0, 0.25)] {
            let aperture_point = Vec3::new(eye.x, eye.y, 0.0);
            assert_eq!(cone.weight((eye - aperture_point).normalize()), 1.0);
        }
    }
}
