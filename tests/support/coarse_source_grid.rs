use super::input::Input;
use bevy_solarik::coarse_cache::{Cell, DIRECTIONS, MAX_CACHE_BYTES, directions};
use bytemuck::Zeroable;

/// Conservative triangle AABB candidates. Sampling never removes a candidate.
pub fn cells(input: &Input, resolution: u32) -> Vec<Cell> {
    assert!(matches!(resolution, 8 | 16 | 32 | 64));
    let addresses: Vec<[i32; 3]> = input
        .positions
        .iter()
        .map(|p| {
            core::array::from_fn(|axis| {
                let value = (p[axis] * resolution as f32).floor();
                assert!(value.is_finite() && value.abs() < 4096.);
                value as i32
            })
        })
        .collect();
    let low: [i32; 3] = core::array::from_fn(|a| addresses.iter().map(|p| p[a]).min().unwrap());
    let high: [i32; 3] = core::array::from_fn(|a| addresses.iter().map(|p| p[a]).max().unwrap());
    let size: [usize; 3] = core::array::from_fn(|a| (high[a] - low[a] + 1) as usize);
    let volume = size
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .unwrap();
    assert!(volume <= 16_777_216, "candidate grid exceeds 16 MiB");
    let index = |p: [i32; 3]| {
        ((p[0] - low[0]) as usize * size[1] + (p[1] - low[1]) as usize) * size[2]
            + (p[2] - low[2]) as usize
    };
    let mut occupied = vec![false; volume];
    let mut work = 0u64;
    for triangle in input.indices.chunks_exact(3) {
        let points = [
            addresses[triangle[0] as usize],
            addresses[triangle[1] as usize],
            addresses[triangle[2] as usize],
        ];
        let lo: [i32; 3] = core::array::from_fn(|a| points.iter().map(|p| p[a]).min().unwrap());
        let hi: [i32; 3] = core::array::from_fn(|a| points.iter().map(|p| p[a]).max().unwrap());
        work += (0..3).map(|a| (hi[a] - lo[a] + 1) as u64).product::<u64>();
        assert!(work <= 100_000_000, "candidate work budget exceeded");
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    occupied[index([x, y, z])] = true;
                }
            }
        }
    }
    let mut cells = Vec::new();
    let step = 1. / resolution as f32;
    for x in low[0]..=high[0] {
        for y in low[1]..=high[1] {
            for z in low[2]..=high[2] {
                if occupied[index([x, y, z])] {
                    let mut cell = Cell::zeroed();
                    cell.grid = [x, y, z, resolution as i32];
                    cell.centre_half = [
                        (x as f32 + 0.5) * step,
                        (y as f32 + 0.5) * step,
                        (z as f32 + 0.5) * step,
                        step * 0.5,
                    ];
                    cells.push(cell);
                    assert!(
                        cells.len() * size_of::<Cell>() + 88 <= MAX_CACHE_BYTES,
                        "cache size budget exceeded"
                    );
                }
            }
        }
    }
    cells
}

pub fn tasks(cells: &[Cell], side: u32) -> Vec<[f32; 8]> {
    let directions = directions();
    cells
        .iter()
        .flat_map(|c| {
            directions.iter().map(move |d| {
                [
                    c.centre_half[0],
                    c.centre_half[1],
                    c.centre_half[2],
                    c.centre_half[3],
                    d[0],
                    d[1],
                    d[2],
                    side as f32,
                ]
            })
        })
        .collect()
}

pub const BATCH_CELLS: usize = 4096;
pub const BATCH_TASKS: usize = BATCH_CELLS * DIRECTIONS;

#[test]
fn boundary_candidates_are_sorted_and_not_dropped() {
    let input = Input {
        positions: vec![
            [-0.125, 0., 0., 0.],
            [0.125, 0., 0., 0.],
            [0., 0.125, 0., 0.],
        ],
        indices: vec![0, 1, 2],
        ..Default::default()
    };
    let cells = cells(&input, 8);
    assert_eq!(cells.len(), 6);
    assert_eq!(cells[0].grid, [-1, 0, 0, 8]);
    assert_eq!(cells[5].grid, [1, 1, 0, 8]);
    assert_eq!(tasks(&cells, 8).len(), 6 * DIRECTIONS);
}
