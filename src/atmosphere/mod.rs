//! Clear terrestrial atmosphere with one shared observer and linear photometric units.
//!
//! GPU radiance is RGB cd/m²; directional illuminance is RGB lux. The medium
//! follows Hillaire (2020), with Earth coefficients and ozone profiles from
//! Bruneton (2017). See `docs/atmosphere.md` for scope and error budgets.

mod gpu;
pub use gpu::{
    AtmosphereBackground, AtmosphereCamera, AtmosphereComposite, AtmosphereEnvironment,
    AtmospherePlugin,
};

use bevy_ecs::resource::Resource;
use bevy_math::{DVec3, Vec3, Vec4};
use bevy_render::{extract_resource::ExtractResource, render_resource::ShaderType};

/// Physical angular radius of the solar disk, in radians.
pub const SUN_ANGULAR_RADIUS: f32 = 0.004675;
/// Angular radius of the full moon used by this clear-sky model.
pub const MOON_ANGULAR_RADIUS: f32 = 0.00452;

/// Multipliers on Earth's molecular, aerosol and ozone density profiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereMedium {
    pub rayleigh: f32,
    pub mie: f32,
    pub ozone: f32,
    pub mie_anisotropy: f32,
    pub ground_albedo: Vec3,
    pub multiple_scattering: bool,
}

impl Default for AtmosphereMedium {
    fn default() -> Self {
        Self {
            rayleigh: 1.0,
            mie: 1.0,
            ozone: 1.0,
            mie_anisotropy: 0.8,
            ground_albedo: Vec3::splat(0.3),
            multiple_scattering: true,
        }
    }
}

/// View-ray integration quality. Medium LUT quality is fixed independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AtmosphereQuality {
    Balanced,
    #[default]
    High,
}

/// One coherent clear-sky state shared by presentation and lighting consumers.
///
/// Heights and aerial distances are metres above the local planet surface.
/// Sources are top-of-atmosphere illuminances, before atmospheric attenuation.
#[derive(Resource, ExtractResource, Debug, Clone, PartialEq)]
pub struct AtmosphereState {
    pub medium: AtmosphereMedium,
    pub sun_direction: Vec3,
    pub sun_illuminance: f32,
    pub moon_direction: Vec3,
    pub moon_illuminance: f32,
    pub observer_height: f32,
    pub aerial_distance: f32,
    pub stars: f32,
    pub quality: AtmosphereQuality,
}

impl Default for AtmosphereState {
    fn default() -> Self {
        Self {
            medium: AtmosphereMedium::default(),
            sun_direction: Vec3::Y,
            sun_illuminance: 110_000.0,
            moon_direction: Vec3::new(0.0, 0.66, -0.75).normalize(),
            moon_illuminance: 0.25,
            observer_height: 2.0,
            aerial_distance: 2000.0,
            stars: 1.0,
            quality: AtmosphereQuality::High,
        }
    }
}

impl AtmosphereState {
    /// Reject invalid controls before either renderer consumes them.
    pub fn validate(&self) -> Result<(), &'static str> {
        for (value, maximum) in [
            (self.medium.rayleigh, 8.0),
            (self.medium.mie, 16.0),
            (self.medium.ozone, 8.0),
            (self.stars, 4.0),
        ] {
            if !value.is_finite() || !(0.0..=maximum).contains(&value) {
                return Err(
                    "atmosphere density or star multiplier outside its finite physical range",
                );
            }
        }
        if !self.medium.mie_anisotropy.is_finite()
            || !(0.0..=0.95).contains(&self.medium.mie_anisotropy)
        {
            return Err("atmosphere Mie anisotropy must be finite in 0..0.95");
        }
        let albedo = self.medium.ground_albedo;
        if !albedo.is_finite() || albedo.min_element() < 0.0 || albedo.max_element() > 1.0 {
            return Err("atmosphere ground albedo must be finite in 0..1");
        }
        for direction in [self.sun_direction, self.moon_direction] {
            if !direction.is_finite() || (direction.length_squared() - 1.0).abs() > 0.001 {
                return Err("atmosphere source directions must be finite unit vectors");
            }
        }
        for lux in [self.sun_illuminance, self.moon_illuminance] {
            if !lux.is_finite() || !(0.0..=200_000.0).contains(&lux) {
                return Err("atmosphere source illuminance must be finite in 0..200000 lux");
            }
        }
        if !self.observer_height.is_finite() || !(1.0..=20_000.0).contains(&self.observer_height) {
            return Err("atmosphere observer height must be finite in 1..20000 metres");
        }
        if !self.aerial_distance.is_finite() || !(100.0..=20_000.0).contains(&self.aerial_distance)
        {
            return Err("atmosphere aerial distance must be finite in 100..20000 metres");
        }
        Ok(())
    }

    /// Direct transmittance. Only two celestial rays need CPU quadrature when
    /// the state changes; sky texels and lighting maps are generated on the GPU.
    #[must_use]
    pub fn transmittance_to_space(&self, direction: Vec3) -> Vec3 {
        if !direction.is_finite() || direction.length_squared() < 0.5 {
            return Vec3::ZERO;
        }
        let direction = direction.as_dvec3().normalize();
        let r = 6360.0 + f64::from(self.observer_height) * 0.001;
        let b = r * direction.y;
        let c = (r - 6360.0) * (r + 6360.0);
        if b < 0.0 && b * b >= c {
            return Vec3::ZERO;
        }
        let distance = -b + (b * b + (6460.0 - r) * (6460.0 + r)).max(0.0).sqrt();
        let dt = distance / 128.0;
        let mut optical = DVec3::ZERO;
        for i in 0..128 {
            let t = (f64::from(i) + 0.5) * dt;
            let h = (r * r + 2.0 * b * t + t * t).sqrt() - 6360.0;
            let rayleigh = DVec3::new(0.005802, 0.013558, 0.0331)
                * f64::from(self.medium.rayleigh)
                * (-h / 8.0).exp();
            let mie = DVec3::splat(0.00444 * f64::from(self.medium.mie) * (-h / 1.2).exp());
            let ozone = DVec3::new(0.00065, 0.001881, 0.000085)
                * f64::from(self.medium.ozone)
                * (1.0 - (h - 25.0).abs() / 15.0).max(0.0);
            optical += (rayleigh + mie + ozone) * dt;
        }
        Vec3::new(
            (-optical.x).exp() as f32,
            (-optical.y).exp() as f32,
            (-optical.z).exp() as f32,
        )
    }

    /// Normal-incidence RGB solar illuminance at the shared observer.
    #[must_use]
    pub fn sun_illuminance_rgb(&self) -> Vec3 {
        self.transmittance_to_space(self.sun_direction) * self.sun_illuminance
    }

    /// Normal-incidence RGB lunar illuminance at the shared observer.
    #[must_use]
    pub fn moon_illuminance_rgb(&self) -> Vec3 {
        self.transmittance_to_space(self.moon_direction) * self.moon_illuminance
    }

    pub(crate) fn uniform(&self) -> AtmosphereParams {
        AtmosphereParams {
            medium: Vec4::new(
                self.medium.rayleigh,
                self.medium.mie,
                self.medium.ozone,
                self.medium.mie_anisotropy,
            ),
            ground: self
                .medium
                .ground_albedo
                .extend(f32::from(self.medium.multiple_scattering)),
            sun: self.sun_direction.extend(self.sun_illuminance),
            moon: self.moon_direction.extend(self.moon_illuminance),
            observer: Vec4::new(
                self.observer_height,
                self.aerial_distance,
                self.stars,
                match self.quality {
                    AtmosphereQuality::Balanced => 32.0,
                    AtmosphereQuality::High => 64.0,
                },
            ),
        }
    }
}

#[derive(Debug, Clone, ShaderType)]
pub(crate) struct AtmosphereParams {
    medium: Vec4,
    ground: Vec4,
    sun: Vec4,
    moon: Vec4,
    observer: Vec4,
}
