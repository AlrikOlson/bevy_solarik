//! Optional physical lens controls for Solarik's HDR postprocess chain.
//! Requires Bevy's `PostProcessPlugin` (included in `DefaultPlugins`).
//! Optics metering/scattering precedes bloom, bokeh `DoF` and tone mapping.
//! This screen-space approximation cannot resolve hidden or multilayer surfaces.
use bevy_post_process::{
    bloom::{Bloom, BloomCompositeMode},
    dof::{DepthOfField, DepthOfFieldMode},
};

/// Opt-in lens controls; distances are metres and blur diameter is output pixels.
/// Adding these components never changes lighting, geometry or temporal history.
#[derive(Clone, Copy, Debug)]
pub struct CinematicLens {
    pub focus_distance_m: f32,
    pub sensor_height_m: f32,
    pub f_number: f32,
    pub max_blur_pixels: f32,
    pub bloom_fraction: f32,
}
impl Default for CinematicLens {
    fn default() -> Self {
        Self {
            focus_distance_m: 8.0,
            sensor_height_m: 0.024,
            f_number: 1.8,
            max_blur_pixels: 8.0,
            bloom_fraction: 0.04,
        }
    }
}
impl CinematicLens {
    /// Validate before inserting on an HDR perspective camera. Focus can then
    /// be animated directly through `DepthOfField` without resetting lighting.
    pub fn components(self) -> Result<(DepthOfField, Bloom), &'static str> {
        for (value, minimum, maximum) in [
            (self.focus_distance_m, 0.1, 1.0e7),
            (self.sensor_height_m, 0.001, 0.1),
            (self.f_number, 0.5, 64.0),
            (self.max_blur_pixels, 0.0, 32.0),
            (self.bloom_fraction, 0.0, 0.25),
        ] {
            if !value.is_finite() || !(minimum..=maximum).contains(&value) {
                return Err("invalid cinematic lens");
            }
        }
        Ok((
            DepthOfField {
                mode: DepthOfFieldMode::Bokeh,
                focal_distance: self.focus_distance_m,
                sensor_height: self.sensor_height_m,
                aperture_f_stops: self.f_number,
                max_circle_of_confusion_diameter: self.max_blur_pixels,
                max_depth: 1.0e7,
            },
            Bloom {
                intensity: self.bloom_fraction,
                low_frequency_boost: self.bloom_fraction,
                composite_mode: BloomCompositeMode::EnergyConserving,
                ..Bloom::NATURAL
            },
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_physical_lens_and_unbounded_blur() {
        for value in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            assert!(
                CinematicLens {
                    focus_distance_m: value,
                    ..Default::default()
                }
                .components()
                .is_err()
            );
            assert!(
                CinematicLens {
                    sensor_height_m: value,
                    ..Default::default()
                }
                .components()
                .is_err()
            );
            assert!(
                CinematicLens {
                    f_number: value,
                    ..Default::default()
                }
                .components()
                .is_err()
            );
        }
        for value in [f32::NAN, f32::INFINITY, -1.0, 33.0] {
            assert!(
                CinematicLens {
                    max_blur_pixels: value,
                    ..Default::default()
                }
                .components()
                .is_err()
            );
        }
        for value in [f32::NAN, f32::INFINITY, -1.0, 0.251] {
            assert!(
                CinematicLens {
                    bloom_fraction: value,
                    ..Default::default()
                }
                .components()
                .is_err()
            );
        }
    }
    #[test]
    fn glare_can_be_disabled_without_disabling_focus() {
        let (dof, bloom) = CinematicLens {
            bloom_fraction: 0.0,
            ..Default::default()
        }
        .components()
        .unwrap();
        assert_eq!(bloom.intensity, 0.0);
        assert_eq!(bloom.low_frequency_boost, 0.0);
        assert_eq!(dof.mode, DepthOfFieldMode::Bokeh);
        assert!(dof.focal_distance > 0.0 && dof.max_circle_of_confusion_diameter > 0.0);
    }
}
