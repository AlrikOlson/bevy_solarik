use super::{Grid, gpu, pipeline};
use bevy_math::Vec3;
pub fn check(device: &wgpu::Device, queue: &wgpu::Queue, trace: &wgpu::ComputePipeline) -> usize {
    let requests: Vec<[f32; 8]> = (0..8192)
        .map(|i| {
            let z = 1. - 2. * (i as f32 + 0.5) / 8192.;
            let phi = i as f32 * 2.3999631;
            let radius = (1. - z * z).sqrt();
            [
                0.,
                0.,
                0.,
                0.,
                radius * phi.cos(),
                radius * phi.sin(),
                z,
                i as f32 / 8191.,
            ]
        })
        .collect();
    let (quantized, _) = gpu::run(
        device,
        queue,
        &pipeline(device, "quantization"),
        &[(3, bytemuck::cast_slice(&requests))],
        requests.len(),
    );
    for (i, (r, q)) in requests.iter().zip(&quantized).enumerate() {
        assert_eq!(q.first[1], i as u32 % 4);
        assert_ne!(q.first[0] & 4095, 0);
        assert!((q.depth[0] - r[7]).abs() <= 0.5 / 4094. + 1e-7);
        let original = Vec3::from_array(r[4..7].try_into().unwrap());
        let decoded = Vec3::from_array(q.normal[..3].try_into().unwrap());
        assert!((decoded.length() - 1.).abs() < 2e-6);
        assert!(
            original.dot(decoded) > 0.0085_f32.cos(),
            "normal quantization {i}"
        );
    }
    let grid = Grid {
        origin_resolution: [0, 0, 0, 8],
        dimensions: [1, 1, 1, 0],
        metadata: [1, 8, 0, 1],
    };
    let mut samples = vec![0u32; 384];
    // Quantized normal +Z, source depth .5, original material0.
    samples[4 * 64..6 * 64].fill(0x8040_0800);
    let mut rays = Vec::new();
    let mut expected = Vec::new();
    for d in [
        [0., 0., 1.],
        [0., 0., -1.],
        [0.4, 0.2, 0.9],
        [-0.4, -0.2, -0.9],
        [0.6, -0.4, 0.9],
        [-0.6, 0.4, -0.9],
    ] {
        let d = Vec3::from_array(d).normalize();
        for y in 0..8 {
            for x in 0..8 {
                let point = Vec3::new((x as f32 + 0.37) / 64., (y as f32 + 0.63) / 64., 0.0625);
                let origin = point - d;
                for (low, high, hit) in [(0_f32, 2_f32, true), (0., 0.99, false), (1.01, 2., false)]
                {
                    rays.push([origin.x, origin.y, origin.z, low, d.x, d.y, d.z, high]);
                    expected.push((hit, d.z < 0.));
                }
            }
        }
    }
    for (two_sided, empty, missing) in [
        (1u32, false, false),
        (0, false, false),
        (1, true, false),
        (1, false, true),
    ] {
        let samples = if empty {
            vec![0u32; 384]
        } else {
            samples.clone()
        };
        let lookup = [if missing { u32::MAX } else { 0 }];
        let settings = [1u32, 8, two_sided, 0, u32::from(empty)];
        let (rows, _) = gpu::run(
            device,
            queue,
            trace,
            &[
                (0, bytemuck::bytes_of(&grid)),
                (1, bytemuck::cast_slice(&lookup)),
                (2, bytemuck::cast_slice(&samples)),
                (3, bytemuck::cast_slice(&rays)),
                (5, bytemuck::cast_slice(&settings)),
            ],
            rays.len(),
        );
        for (i, (row, &(hit, front))) in rows.iter().zip(&expected).enumerate() {
            assert_eq!(row.counts[3], 0, "analytic traversal overflow {i}");
            let hit = hit && (two_sided != 0 || front) && !empty && !missing;
            assert_eq!(
                row.depth[0] > 0.,
                hit,
                "source plane ray {i}, two-sided {two_sided}, empty {empty}, missing {missing}: {:?}",
                row.depth
            );
            if hit {
                assert!((row.depth[1] - 1.).abs() < 1e-4, "source plane depth");
                assert_eq!(row.first[0], 1);
            }
            assert_eq!(
                row.counts[1] != 0,
                empty && !missing,
                "unknown cells must survive"
            );
        }
    }
    requests.len() + rays.len() * 4 + boundary_plane(device, queue, trace)
}

fn boundary_plane(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    trace: &wgpu::ComputePipeline,
) -> usize {
    let grid = Grid {
        origin_resolution: [0, 0, 0, 8],
        dimensions: [1, 1, 2, 0],
        metadata: [2, 8, 0, 1],
    };
    let normal = Vec3::new(-0.5, -0.25, 1.).normalize();
    let oct = normal / normal.abs().element_sum();
    let qx = ((oct.x * 0.5 + 0.5) * 511.).round() as u32;
    let qy = ((oct.y * 0.5 + 0.5) * 511.).round() as u32;
    let mut samples = vec![0u32; 2 * 384];
    for z in 0..2 {
        for y in 0..8 {
            for x in 0..8 {
                let depth = 0.97
                    + 0.5 * ((x as f32 + 0.5) / 8. - 0.5)
                    + 0.25 * ((y as f32 + 0.5) / 8. - 0.5)
                    - z as f32;
                if (0. ..1.).contains(&depth) {
                    let word = ((depth * 4094.).round() as u32 + 1) | (qx << 14) | (qy << 23);
                    for side in 0..2 {
                        samples[z * 384 + (4 + side) * 64 + y * 8 + x] = word;
                    }
                }
            }
        }
    }
    let mut rays = Vec::new();
    for direction in [
        Vec3::Z,
        -Vec3::Z,
        Vec3::new(0.4, 0.2, 1.).normalize(),
        Vec3::new(-0.4, -0.2, -1.).normalize(),
    ] {
        for y in 0..64 {
            for x in 0..64 {
                let u = (x as f32 + 0.37) / 64.;
                let v = (y as f32 + 0.61) / 64.;
                let z = 0.97 + 0.5 * (u - 0.5) + 0.25 * (v - 0.5);
                let point = Vec3::new(u, v, z) / 8.;
                let origin = point - direction;
                rays.push([
                    origin.x,
                    origin.y,
                    origin.z,
                    0.,
                    direction.x,
                    direction.y,
                    direction.z,
                    2.,
                ]);
            }
        }
    }
    let lookup = [0u32, 1];
    let settings = [1u32, 8, 1, 0, 0, 0];
    let (rows, _) = gpu::run(
        device,
        queue,
        trace,
        &[
            (0, bytemuck::bytes_of(&grid)),
            (1, bytemuck::cast_slice(&lookup)),
            (2, bytemuck::cast_slice(&samples)),
            (3, bytemuck::cast_slice(&rays)),
            (5, bytemuck::cast_slice(&settings)),
        ],
        rays.len(),
    );
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.counts[3], 0, "boundary plane overflow {i}");
        assert_eq!(row.depth[0], 1., "source plane hole {i}");
        assert!(
            (row.depth[1] - 1.).abs() < 1e-4,
            "boundary plane depth {i}: {}",
            row.depth[1]
        );
    }
    rays.len()
}
