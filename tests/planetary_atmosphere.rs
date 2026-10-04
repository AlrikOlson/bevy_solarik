//! Public finite-shell configuration validation.
use bevy_math::Vec3;
use bevy_solarik::atmosphere::PlanetaryAtmosphere;

fn planet() -> PlanetaryAtmosphere {
    PlanetaryAtmosphere {
        radius: 6_378_136.5,
        height: 100_000.0,
        observer: Vec3::Z * 24_000_000.0,
        cloud_coverage: 0.55,
        cloud_extinction: 0.0005,
        seed: 137,
        occluder: Vec3::ZERO,
        occluder_radius: 0.0,
    }
}

#[test]
fn planetary_geometry_rejects_invalid_domains() {
    assert!(planet().validate().is_ok());
    for bad in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e10] {
        let mut p = planet();
        p.radius = bad;
        assert!(p.validate().is_err());
        let mut p = planet();
        p.height = bad;
        assert!(p.validate().is_err());
    }
    for origin in [Vec3::ZERO, Vec3::splat(f32::NAN), Vec3::Z * 1e10] {
        let mut p = planet();
        p.observer = origin;
        assert!(p.validate().is_err());
    }
    let mut p = planet();
    p.observer = Vec3::Y * (p.radius + 2.0);
    assert!(p.validate().is_ok());
    p.cloud_coverage = 1.01;
    assert!(p.validate().is_err());
    p.cloud_coverage = 0.0;
    p.cloud_extinction = -0.001;
    assert!(p.validate().is_err());
    // Liquid-cloud extinction of tens per kilometre is in range; the bound is 0.2 per metre.
    p.cloud_extinction = 0.045;
    assert!(p.validate().is_ok());
    p.cloud_extinction = 0.21;
    assert!(p.validate().is_err());
    p.cloud_extinction = 0.0;
    p.occluder_radius = f32::NAN;
    assert!(p.validate().is_err());
}
