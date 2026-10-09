use super::{Probe, gpu};
use bevy_math::Vec3;
use bevy_solarik::{coarse_cache::directions, coarse_scene::Grid, coarse_transport::Kernel};
pub fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
) -> usize {
    let analytic = gpu::pipeline(device, "analytic");
    let mut requests = Vec::new();
    for i in 0..65536 {
        requests.push([
            2.,
            3.,
            4.,
            (i as f32 + 0.5) / 65536.,
            0.3,
            0.4,
            (0.75_f32).sqrt(),
            0.,
        ]);
    }
    for d in directions() {
        requests.push([1., 0., 1., 0.5, d[0], d[1], d[2], 0.]);
    }
    requests.extend([
        [0., 0., 1., 0.5, 1., 0., 0., 0.],
        [-1., 0., 1., 0.5, 1., 0., 0., 0.],
        [1., 0., 1., 1., 1., 0., 0., 0.],
        [1., 1., 0., 0.5, 1., 0., 0., 0.],
    ]);
    let (rows, _) = gpu::run(
        device,
        queue,
        &analytic,
        &[(3, bytemuck::cast_slice(&requests))],
        requests.len(),
    );
    let mut hits = 0;
    for (request, row) in requests[..65536].iter().zip(&rows) {
        assert_eq!(row.counts[1], 1);
        let offset = -(1.0 - f64::from(request[3])).ln() / 2.;
        let expected = offset < 1.;
        assert_eq!(row.counts[0], u32::from(expected));
        if expected {
            hits += 1;
            assert!((f64::from(row.depth[0]) - 3. - offset).abs() < 2e-6);
        }
        assert!((f64::from(row.depth[1]) - (1. - (-2_f64).exp())).abs() < 2e-7);
        assert!((f64::from(row.depth[2]) - (0.5 - 1. / (2_f64.exp() - 1.))).abs() < 2e-7);
        let direction = Vec3::from_array(request[4..7].try_into().unwrap());
        let reconstructed: Vec3 = (0..3)
            .map(|a| Vec3::from_array(directions()[row.first[a] as usize]) * row.normal[a])
            .sum();
        assert!(reconstructed.normalize().distance(direction) < 2e-6);
    }
    assert!((hits as f64 / 65536. - (1. - (-2_f64).exp())).abs() < 1. / 65536.);
    for (i, row) in rows[65536..65550].iter().enumerate() {
        let mass: f32 = (0..3)
            .filter(|&a| row.first[a] == i as u32)
            .map(|a| row.normal[a])
            .sum();
        assert!((mass - 1.).abs() < 1e-6, "basis node {i}");
    }
    assert_eq!(rows[65550].counts[..2], [0, 1]);
    for row in &rows[65551..] {
        assert_eq!(row.counts[1], 0);
    }
    let grid = Grid {
        origin_resolution: [0, 0, 0, 8],
        dimensions: [1, 1, 1, 0],
        metadata: [1, 8, 0, 1],
    };
    let requests: [[f32; 8]; 4] = [
        [-1., 0.0625, 0.0625, 0., 1., 0., 0., 2.],
        [-1., 0.0625, 0.0625, 1.03125, 1., 0., 0., 2.],
        [-1., 0.0625, 0.0625, 0., 1., 0., 0., 1.03125],
        [-1., 0.0625, 0.0625, 0., 0., 0., 0., 2.],
    ];
    for (measured, address) in [(0x3fff, 0), (0, 0), (0x3fff, u32::MAX)] {
        let kernel = Kernel {
            rates: [4.; 14],
            measured,
            reserved: 0,
        };
        let (rows, _) = gpu::run(
            device,
            queue,
            pipeline,
            &[
                (0, bytemuck::bytes_of(&grid)),
                (1, bytemuck::bytes_of(&address)),
                (2, bytemuck::bytes_of(&kernel)),
                (3, bytemuck::cast_slice(&requests)),
            ],
            requests.len(),
        );
        for (i, row) in rows[..3].iter().enumerate() {
            let optical: f64 = [4., 3., 1.][i];
            let expected = if address == u32::MAX {
                0.
            } else {
                1. - (-optical).exp()
            };
            assert!(
                (f64::from(row.depth[0]) - expected).abs() < 2e-7,
                "interval {i}, mask {measured}, cell {address}: {:?} {:?}, expected {expected}",
                row.depth,
                row.counts
            );
            assert_eq!(
                row.counts[1],
                u32::from(measured == 0 && address != u32::MAX)
            );
            assert_eq!(row.counts[3], 0);
        }
        assert_eq!(
            rows[3].counts[3], 2,
            "invalid ray cannot become an accepted miss"
        );
        if address != u32::MAX {
            let full = 1. - f64::from(rows[0].depth[0]);
            let split = (1. - f64::from(rows[1].depth[0])) * (1. - f64::from(rows[2].depth[0]));
            assert!((full - split).abs() < 2e-7);
        }
    }
    requests.len() * 3 + 65554
}
pub fn images(input: &super::input::Input, side: usize) -> (Vec<[f32; 8]>, String) {
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for p in &input.positions {
        let v = Vec3::from_array(p[..3].try_into().unwrap());
        low = low.min(v);
        high = high.max(v);
    }
    let centre = (low + high) * 0.5;
    let half = (high - low) * 0.5;
    let directions = [
        [0., 0., 1.],
        [0., 0., -1.],
        [1., 0., 0.],
        [-1., 0., 0.],
        [0.3, 0.1, 0.9],
        [-0.3, -0.1, -0.9],
        [0.8, 0.2, -0.6],
        [-0.8, -0.2, 0.6],
        [0.2, -0.7, 0.5],
        [-0.2, 0.7, -0.5],
    ];
    let mut rays = Vec::new();
    let mut views = Vec::new();
    for (view, d) in directions.into_iter().enumerate() {
        let d = Vec3::from_array(d).normalize();
        let right = d.cross(Vec3::Y).normalize();
        let up = right.cross(d);
        let extent = right.abs().dot(half).max(up.abs().dot(half)) * 1.05;
        views.push(format!(
            r#"{{"index":{view},"direction":{:?},"half_extent":{extent},"held_out":{}}}"#,
            d.to_array(),
            view >= 4
        ));
        for y in 0..side {
            for x in 0..side {
                let uv = [
                    (x as f32 + 0.5) / side as f32 * 2. - 1.,
                    1. - (y as f32 + 0.5) / side as f32 * 2.,
                ];
                let origin = centre + right * uv[0] * extent + up * uv[1] * extent - d * 3.;
                rays.push([origin.x, origin.y, origin.z, 0., d.x, d.y, d.z, 6.]);
            }
        }
    }
    (rays, format!("[{}]", views.join(",")))
}
pub fn save(path: &std::path::Path, rows: &[Probe], reference: bool) {
    let values: Vec<[f32; 4]> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.counts[3], 0, "unresolved overflow in diagnostic");
            if reference {
                [if r.counts[1] > 0 { 1. } else { 0. }, r.depth[0], 0., 0.]
            } else {
                assert!(r.depth[..2].iter().all(|v| v.is_finite()));
                assert!((0. ..=1.).contains(&r.depth[0]));
                [r.depth[0], r.depth[1], r.counts[1] as f32, 0.]
            }
        })
        .collect();
    super::input::write(path, bytemuck::cast_slice(&values));
}
