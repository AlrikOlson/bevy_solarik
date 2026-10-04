//! Validation of consumer-configurable traversal limits.
use bevy_solarik::prelude::SolarikRaySettings;

#[test]
fn defaults_preserve_existing_range_and_no_relative_floor() {
    let settings = SolarikRaySettings::default();
    assert_eq!(settings.max_distance(), 100_000.0);
    assert_eq!(settings.relative_min_distance(), 0.0);
}

#[test]
fn invalid_limits_are_rejected_before_gpu_upload() {
    for maximum in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1.0e17] {
        assert!(SolarikRaySettings::new(maximum, 0.0).is_err());
    }
    for relative in [f32::NAN, f32::INFINITY, -1.0, 1.0] {
        assert!(SolarikRaySettings::new(1.0e12, relative).is_err());
    }
    let settings = SolarikRaySettings::new(1.0e12, 1.0 / 32768.0).unwrap();
    assert_eq!(settings.max_distance(), 1.0e12);
    assert_eq!(settings.relative_min_distance(), 1.0 / 32768.0);
}
