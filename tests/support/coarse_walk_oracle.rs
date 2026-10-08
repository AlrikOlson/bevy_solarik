use super::{Probe, Request};
use bevy_math::Vec3;
use bevy_solarik::coarse_cache::{self, Cell, SourceCache};
use bytemuck::Zeroable;

pub fn compare(source: &SourceCache, rays: &[Request], rows: &[Probe]) -> f64 {
    let mut maximum = 0f64;
    for (ray_index, (ray, row)) in rays.iter().zip(rows).enumerate() {
        assert_eq!(row.counts[2], 0, "ray{ray_index} overflow/invalid");
        let mut expected = Vec::new();
        // Independent oracle: intersect every original candidate AABB in f64,
        // sort the positive-length intervals; no CPU DDA or GPU address table.
        for (cell_index, cell) in source.cells.iter().enumerate() {
            if let Some((low, high)) = interval(cell, ray) {
                expected.push((low, high, cell_index as u32, cell.directions[0].counts[1]));
            }
        }
        expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
        assert_eq!(
            row.counts[0] as usize,
            expected.len(),
            "source{} r{} ray{ray_index}: {:?} {:?}; expected{expected:?}; actual{:?}",
            source.prototype,
            source.resolution,
            ray.origin,
            ray.direction,
            row.intervals[..row.counts[0] as usize]
                .iter()
                .map(|v| (v[0], endpoint(v, 4), endpoint(v, 6)))
                .collect::<Vec<_>>()
        );
        for (i, (low, high, cell, coverage)) in expected.iter().copied().enumerate() {
            let actual = row.intervals[i];
            assert_eq!(
                (actual[0], actual[1]),
                (cell, coverage),
                "ray{ray_index} interval{i}"
            );
            let a = endpoint(&actual, 4);
            let b = endpoint(&actual, 6);
            assert!(
                a.is_finite() && b.is_finite() && b > a,
                "ray{ray_index} invalid interval"
            );
            let error = (a - low).abs().max((b - high).abs());
            maximum = maximum.max(error);
            assert!(
                error < 1e-5,
                "ray{ray_index} interval{i} depth error {error}"
            );
            if i > 0 {
                assert!(
                    a >= endpoint(&row.intervals[i - 1], 6),
                    "unordered intervals"
                );
            }
        }
    }
    maximum
}
fn endpoint(record: &[u32; 8], offset: usize) -> f64 {
    f64::from(f32::from_bits(record[offset])) + f64::from(f32::from_bits(record[offset + 1]))
}
fn interval(cell: &Cell, ray: &Request) -> Option<(f64, f64)> {
    let mut low = f64::from(ray.origin[3]);
    let mut high = f64::from(ray.direction[3]);
    for axis in 0..3 {
        let minimum = cell.grid[axis] as f64 / cell.grid[3] as f64;
        let maximum = (cell.grid[axis] as f64 + 1.) / cell.grid[3] as f64;
        let origin = f64::from(ray.origin[axis]);
        let direction = f64::from(ray.direction[axis]);
        if direction == 0. {
            if origin < minimum || origin >= maximum {
                return None;
            }
        } else {
            let a = (minimum - origin) / direction;
            let b = (maximum - origin) / direction;
            low = low.max(a.min(b));
            high = high.min(a.max(b));
        }
    }
    (high > low).then_some((low, high))
}
pub fn ray(origin: [f32; 3], direction: Vec3, low: f32, high: f32) -> Request {
    let d = direction.normalize();
    Request {
        origin: [origin[0], origin[1], origin[2], low],
        direction: [d.x, d.y, d.z, high],
    }
}
pub fn real_rays(source: &SourceCache) -> Vec<Request> {
    let mut rays = Vec::new();
    for i in 0..16 {
        let cell = &source.cells[i * (source.cells.len() - 1) / 15];
        let point = Vec3::from_array(cell.centre_half[..3].try_into().unwrap());
        for d in coarse_cache::directions() {
            let d = Vec3::from_array(d);
            rays.push(ray((point - d * 2.).to_array(), d, 0., 4.));
        }
    }
    for i in 0..32 {
        let cell = &source.cells[i * (source.cells.len() - 1) / 31];
        let point = Vec3::from_array(cell.centre_half[..3].try_into().unwrap());
        let d = Vec3::new(
            (i * 17 % 23) as f32 - 11.,
            (i * 7 % 19) as f32 - 9.,
            (i * 5 % 13) as f32 - 6.,
        )
        .normalize();
        rays.push(ray((point - d * 2.).to_array(), d, 1.8, 2.3));
    }
    rays
}
pub fn analytic() -> (SourceCache, Vec<Request>) {
    let mut cells = Vec::new();
    for x in -1..=0 {
        for y in -1..=0 {
            for z in -1..=0 {
                let mut cell = Cell::zeroed();
                cell.grid = [x, y, z, 16];
                cell.centre_half = [
                    (x as f32 + 0.5) / 16.,
                    (y as f32 + 0.5) / 16.,
                    (z as f32 + 0.5) / 16.,
                    0.5 / 16.,
                ];
                cells.push(cell);
            }
        }
    }
    let rays = vec![
        ray([-0.2, -0.03125, -0.03125], Vec3::X, 0., 1.),
        ray([0.2, -0.03125, -0.03125], -Vec3::X, 0., 1.),
        ray([-0.03125, -0.2, -0.03125], Vec3::Y, 0., 1.),
        ray([-0.03125, -0.03125, 0.2], -Vec3::Z, 0., 1.),
        ray([-0.2, -0.2, -0.2], Vec3::ONE, 0., 1.),
        ray([0.2, 0.2, 0.2], -Vec3::ONE, 0., 1.),
        ray([-0.2, -0.2, 0.], Vec3::new(1., 1., 0.), 0., 1.),
        ray([-0.2, 0.0625, 0.], Vec3::X, 0., 1.),
        ray([-0.2, 0., 0.], Vec3::X, 0., 1.),
        ray([0., 0., 0.], Vec3::X, 0., 0.03125),
        ray([0., 0., 0.], -Vec3::X, 0., 0.03125),
        ray([-0.2, 0., 0.], Vec3::X, 0., 0.1),
        ray([-0.2, 0., 0.], Vec3::X, 0.16, 0.24),
        ray([-0.2, 0., 0.], Vec3::X, 0.2, 0.24),
        ray([-0.2, -0.03125, 0.], Vec3::new(1., 1e-37, 0.), 0., 1.),
        ray([-0.2, 0., 0.], Vec3::new(1., -1e-37, 0.), 0., 1.),
    ];
    (
        SourceCache {
            identity: [3; 32],
            prototype: 5,
            resolution: 16,
            samples_side: 8,
            cells,
        },
        rays,
    )
}
pub fn overflow() -> (SourceCache, Vec<Request>) {
    let (mut source, _) = analytic();
    source.cells = vec![source.cells[0], source.cells[0]];
    for (cell, x) in source.cells.iter_mut().zip([0, 1023]) {
        cell.grid = [x, 0, 0, 16];
        cell.centre_half = [(x as f32 + 0.5) / 16., 0.5 / 16., 0.5 / 16., 0.5 / 16.];
    }
    (source, vec![ray([-1., 0.03125, 0.03125], Vec3::X, 0., 66.)])
}
