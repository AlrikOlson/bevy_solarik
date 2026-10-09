use super::*;
use bytemuck::Zeroable;

#[test]
fn level_family_rejects_duplicates_disorder_and_unbounded_values() {
    assert_eq!(parse_levels("1,2,4,8,16,32").unwrap(), DEFAULT_LEVELS);
    assert_eq!(parse_levels("16, 32,64").unwrap(), [16, 32, 64]);
    for value in ["", "0", "3", "128", "1,1", "32,16", "1,", "-1", "NaN"] {
        assert!(parse_levels(value).is_none(), "{value}");
    }
}
#[test]
fn every_supported_grid_roundtrips_without_changing_source_observations() {
    for resolution in [1, 2, 4, 8, 16, 32, 64] {
        let mut cache = cache();
        cache.resolution = resolution;
        let step = 1. / resolution as f32;
        cache.cells[0].grid[3] = resolution as i32;
        cache.cells[0].centre_half = [-0.5 * step, 0.5 * step, 2.5 * step, 0.5 * step];
        let bytes = encode(&cache).unwrap();
        let loaded = decode(&bytes, cache.identity).unwrap();
        assert_eq!(loaded.resolution, resolution);
        assert_eq!(
            bytemuck::cast_slice::<Cell, u8>(&cache.cells),
            bytemuck::cast_slice::<Cell, u8>(&loaded.cells)
        );
    }
}

fn cache() -> SourceCache {
    let mut cell = Cell::zeroed();
    cell.grid = [-1, 0, 2, 16];
    cell.centre_half = [-0.5 / 16., 0.5 / 16., 2.5 / 16., 0.5 / 16.];
    for bin in &mut cell.directions {
        bin.counts = [64, 32, 48, 0];
        bin.depth = [0.2, 0.8, 10., 20.];
        bin.diagonal = [32., 0., 0., 0.];
        bin.normal = [32., 0., 0., 0.];
        bin.materials[2] = 32;
    }
    SourceCache {
        identity: [7; 32],
        prototype: 5,
        resolution: 16,
        samples_side: 8,
        cells: vec![cell],
    }
}
#[test]
fn cache_roundtrip_preserves_source_material_depth_and_normal_moments() {
    assert_eq!(size_of::<Directional>(), 112);
    assert_eq!(size_of::<Cell>(), 1600);
    let cache = cache();
    let bytes = encode(&cache).unwrap();
    let loaded = decode(&bytes, cache.identity).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<Cell, u8>(&cache.cells),
        bytemuck::cast_slice::<Cell, u8>(&loaded.cells)
    );
    assert_eq!(
        (loaded.prototype, loaded.resolution, loaded.samples_side),
        (5, 16, 8)
    );
}
#[test]
fn rejects_corruption_truncation_foreign_identity_and_inconsistent_cells() {
    let mut cache = cache();
    let bytes = encode(&cache).unwrap();
    assert!(decode(&bytes, [8; 32]).is_none());
    for offset in [0, 8, 40, 72, 76, 80, 84, 88, bytes.len() - 1] {
        let mut broken = bytes.clone();
        broken[offset] ^= 0x80;
        assert!(decode(&broken, cache.identity).is_none());
    }
    assert!(decode(&bytes[..bytes.len() - 1], cache.identity).is_none());
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(decode(&extra, cache.identity).is_none());
    cache.cells.push(cache.cells[0]);
    assert!(encode(&cache).is_none());
    cache.cells.pop();
    cache.cells[0].directions[0].counts[3] = 1;
    assert!(encode(&cache).is_none());
}
#[test]
fn rejects_inconsistent_first_and_second_normal_moments() {
    let mut cache = cache();
    cache.cells[0].directions[0].normal = [0., 32., 0., 0.];
    assert!(encode(&cache).is_none());
}
#[test]
fn sampled_empty_candidates_remain_and_invalid_moments_reject() {
    let mut cache = cache();
    cache.cells[0].directions = [Directional::default(); DIRECTIONS];
    assert_eq!(
        decode(&encode(&cache).unwrap(), cache.identity)
            .unwrap()
            .cells
            .len(),
        1
    );
    cache.cells[0].directions[0].diagonal[0] = f32::NAN;
    assert!(encode(&cache).is_none());
}
