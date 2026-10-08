use super::{MISSING, Probe, Request};
use bevy_solarik::{
    coarse_cache::{DIRECTIONS, Directional, SourceCache},
    coarse_scene::PackedSource,
};
pub fn build(source: &SourceCache, packed: &PackedSource) -> (Vec<Request>, Vec<Probe>) {
    let mut requests = Vec::new();
    let mut expected = Vec::new();
    for (cell_index, cell) in source.cells.iter().enumerate() {
        for direction in 0..DIRECTIONS {
            let point: [f32; 3] = cell.centre_half[..3].try_into().unwrap();
            let index = packed.grid.index(point).unwrap() as u32;
            requests.push(Request {
                point: [point[0], point[1], point[2], 0.],
                parameters: [direction as u32, 0, 0, 0],
            });
            expected.push(Probe {
                address: [1, cell_index as u32, direction as u32, index],
                measurements: cell.directions[direction],
            });
        }
    }
    let grid = &packed.grid;
    for x in 0..grid.dimensions[0] {
        for y in 0..grid.dimensions[1] {
            for z in 0..grid.dimensions[2] {
                let index = ((x * grid.dimensions[1] + y) * grid.dimensions[2] + z) as usize;
                if packed.lookup[index] == MISSING {
                    let point = [
                        (grid.origin_resolution[0] as f32 + x as f32 + 0.5)
                            / source.resolution as f32,
                        (grid.origin_resolution[1] as f32 + y as f32 + 0.5)
                            / source.resolution as f32,
                        (grid.origin_resolution[2] as f32 + z as f32 + 0.5)
                            / source.resolution as f32,
                        0.,
                    ];
                    requests.push(Request {
                        point,
                        parameters: [0; 4],
                    });
                    expected.push(miss());
                }
            }
        }
    }
    let first = source.cells[0].centre_half;
    let lower: [f32; 3] =
        core::array::from_fn(|a| grid.origin_resolution[a] as f32 / source.resolution as f32);
    let upper: [f32; 3] = core::array::from_fn(|a| {
        (grid.origin_resolution[a] as f32 + grid.dimensions[a] as f32) / source.resolution as f32
    });
    for axis in 0..3 {
        for coordinate in [lower[axis] - 0.00001, upper[axis], upper[axis] + 0.00001] {
            let mut point = first;
            point[axis] = coordinate;
            requests.push(Request {
                point,
                parameters: [0; 4],
            });
            expected.push(miss());
        }
    }
    for direction in [DIRECTIONS as u32, u32::MAX] {
        requests.push(Request {
            point: first,
            parameters: [direction, 0, 0, 0],
        });
        expected.push(miss());
    }
    // Every lower face belongs to its integer cell, including negative addresses.
    for (i, cell) in source.cells.iter().enumerate() {
        let point = [
            cell.grid[0] as f32 / source.resolution as f32,
            cell.grid[1] as f32 / source.resolution as f32,
            cell.grid[2] as f32 / source.resolution as f32,
            0.,
        ];
        requests.push(Request {
            point,
            parameters: [0; 4],
        });
        expected.push(Probe {
            address: [
                1,
                i as u32,
                0,
                packed.grid.index(point[..3].try_into().unwrap()).unwrap() as u32,
            ],
            measurements: cell.directions[0],
        });
    }
    (requests, expected)
}
fn miss() -> Probe {
    Probe {
        address: [0, MISSING, 0, 0],
        measurements: Directional::default(),
    }
}
