use super::*;
use crate::coarse_cache::{Cell, SourceCache};

pub fn source() -> SourceCache {
    let mut cells = Vec::new();
    for x in [-1, 1] {
        let mut cell = Cell::zeroed();
        cell.grid = [x, 0, 0, 16];
        cell.centre_half = [(x as f32 + 0.5) / 16., 0.5 / 16., 0.5 / 16., 0.5 / 16.];
        for bin in &mut cell.directions {
            bin.counts = [256, 256, 512, 0];
            bin.depth = [0.25, 0.75, 64., 192.];
            bin.diagonal = [0., 0., 256., 0.];
            bin.normal = [0., 0., -256., 0.];
            bin.materials[7] = 256;
        }
        cells.push(cell);
    }
    cells[1].directions = [Directional::default(); DIRECTIONS];
    SourceCache {
        identity: [3; 32],
        prototype: 5,
        resolution: 16,
        samples_side: 16,
        cells,
    }
}
#[test]
fn preserves_maximum_counts_every_float_and_zero_hit_occupancy() {
    assert_eq!(size_of::<PackedDirectional>(), 76);
    assert_eq!(size_of::<Grid>(), 48);
    let source = source();
    let bytes = coarse_cache::encode(&source).unwrap();
    let packed = PackedSource::decode(&bytes, source.identity).unwrap();
    for cell in &source.cells {
        for direction in 0..DIRECTIONS {
            let row = packed
                .sample(cell.centre_half[..3].try_into().unwrap(), direction)
                .unwrap();
            assert_eq!(
                bytemuck::bytes_of(&row),
                bytemuck::bytes_of(&cell.directions[direction])
            );
        }
    }
    assert!(packed.sample([0., 0., 0.], 0).is_none());
    assert!(packed.sample([1.5 / 16., 0., 0.], 0).is_some());
    assert!(packed.bytes() < bytes.len());
}
#[test]
fn half_open_negative_coordinates_and_invalid_inputs_fail_closed() {
    let source = source();
    let mut bytes = coarse_cache::encode(&source).unwrap();
    let packed = PackedSource::decode(&bytes, source.identity).unwrap();
    assert!(packed.sample([-1. / 16., 0., 0.], 0).is_some());
    for point in [
        [-1.0001 / 16., 0., 0.],
        [2. / 16., 0., 0.],
        [0., f32::NAN, 0.],
        [0., f32::INFINITY, 0.],
    ] {
        assert!(packed.sample(point, 0).is_none());
    }
    assert!(packed.sample([-0.5 / 16., 0., 0.], DIRECTIONS).is_none());
    assert!(PackedSource::decode(&bytes, [4; 32]).is_none());
    *bytes.last_mut().unwrap() ^= 1;
    assert!(PackedSource::decode(&bytes, source.identity).is_none());
}
#[test]
fn rejects_overflow_and_noncanonical_padding() {
    let mut bin = source().cells[0].directions[0];
    bin.counts[3] = 1;
    assert!(PackedDirectional::new(&bin, 16).is_none());
    bin.counts[3] = 0;
    bin.normal[3] = -0.;
    assert!(PackedDirectional::new(&bin, 16).is_none());
}
