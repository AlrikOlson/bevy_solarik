use super::*;
use crate::coarse_cache::{Cell, SourceCache};
use bytemuck::Zeroable;
fn fixture() -> (AppearanceCache, SourceCache) {
    let mut cell = Cell::zeroed();
    cell.grid = [0, 0, 0, 16];
    cell.centre_half = [1. / 32.; 4];
    cell.directions[0].counts = [1, 1, 1, 0];
    cell.directions[0].materials[0] = 1;
    cell.directions[0].normal = [0., 0., 1., 0.];
    cell.directions[0].diagonal = [0., 0., 1., 0.];
    let mut moments = vec![Moment::default(); DIRECTIONS];
    moments[0] = Moment {
        color: [0.1, 0.2, 0.3, 1.],
        normal: [0., 0., 1., 0.],
        diagonal: [0., 0., 1., 0.],
        surface: [0.0625, 0., 0.3, 0.],
        counts: [1, 0, 0, 0],
        ..Default::default()
    };
    (
        AppearanceCache {
            identity: [9; 32],
            materials: vec![Material {
                base_color: [1.; 4],
                surface: [0.5, 0., 0.3, 0.],
                optical: [1.45, 0., 0., 0.],
            }],
            moments,
        },
        SourceCache {
            identity: [3; 32],
            prototype: 43,
            resolution: 16,
            samples_side: 8,
            cells: vec![cell],
        },
    )
}
#[test]
fn source_coupled_appearance_roundtrip_preserves_materials_and_moments() {
    assert_eq!(size_of::<Material>(), 48);
    assert_eq!(size_of::<Moment>(), 96);
    let (cache, source) = fixture();
    let bytes = encode(&cache, &source).unwrap();
    let recovered = decode(&bytes, cache.identity, &source).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&cache.moments),
        bytemuck::cast_slice::<_, u8>(&recovered.moments)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&cache.materials),
        bytemuck::cast_slice::<_, u8>(&recovered.materials)
    );
}
#[test]
fn foreign_corrupt_truncated_or_changed_coverage_is_rejected() {
    let (cache, mut source) = fixture();
    let mut bytes = encode(&cache, &source).unwrap();
    assert!(decode(&bytes, [0; 32], &source).is_none());
    assert!(decode(&bytes[..bytes.len() - 1], cache.identity, &source).is_none());
    *bytes.last_mut().unwrap() ^= 1;
    assert!(decode(&bytes, cache.identity, &source).is_none());
    *bytes.last_mut().unwrap() ^= 1;
    source.prototype = 5;
    assert!(decode(&bytes, cache.identity, &source).is_none());
}
#[test]
fn invalid_appearance_never_loses_material_or_normal_evidence() {
    let (mut cache, source) = fixture();
    cache.moments[0].counts[0] = 0;
    assert!(!cache.valid(&source));
    cache.moments[0].counts[0] = 1;
    cache.moments[0].diagonal[2] = 0.;
    assert!(!cache.valid(&source));
    cache.moments[0].diagonal[2] = 1.;
    cache.moments[1].color[0] = 1e-6;
    assert!(!cache.valid(&source));
    cache.moments[1].color[0] = 0.;
    cache.materials[0].optical[0] = f32::NAN;
    assert!(!cache.valid(&source));
}
