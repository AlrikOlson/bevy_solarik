//! Immutable spatial source-hit bake; kept independent of runtime traversal.
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[expect(
    dead_code,
    reason = "reuse the identical camera rays; transport controls have their own target"
)]
#[path = "support/coarse_transport_controls.rs"]
mod cameras;
#[path = "support/coarse_spatial_cook_gpu.rs"]
mod cook_gpu;
#[path = "support/coarse_transport_gpu.rs"]
mod gpu;
#[path = "support/source_coverage_input.rs"]
mod input;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{coarse_cache, coarse_spatial::SpatialCache};
use std::{path::Path, time::Instant};
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Probe {
    depth: [f32; 4],
    counts: [u32; 4],
    first: [u32; 4],
    normal: [f32; 4],
}
fn identity(name: &str) -> [u8; 32] {
    let key = std::env::var(name).unwrap();
    assert_eq!(key.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap())
}
fn analytic(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline) -> usize {
    let tasks = [
        [0., 0., 0., 0.5, 0., 0., 1., 8.],
        [0., 0., 0., 0.5, 0., 0., -1., 8.],
    ];
    for (cutoff, two_sided) in [(-1_f32, 1), (-1., 0), (0.5, 1)] {
        let source = input::Input {
            width: 2,
            height: 1,
            alpha: vec![255, 255, 255, 0, 255, 255, 255, 255],
            positions: vec![
                [-0.5, -0.5, 0., 0.],
                [0.5, -0.5, 0., 0.],
                [0.5, 0.5, 0., 0.],
                [-0.5, 0.5, 0., 0.],
            ],
            uvs: vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            indices: vec![0, 1, 2, 0, 2, 3],
            parts: vec![[0, 6, cutoff.to_bits(), two_sided]],
            rays: vec![[0.; 8]],
            ..Default::default()
        };
        let scene = acceleration::build(device, queue, &source);
        let rows = cook_gpu::run(device, queue, pipeline, &scene, &tasks);
        for (face, row) in rows.iter().enumerate() {
            let count = row.iter().filter(|&&w| w != 0).count();
            let expected = if two_sided == 0 && face == 0 {
                0
            } else if cutoff < 0. {
                64
            } else {
                32
            };
            assert_eq!(count, expected, "source cutoff/sidedness face {face}");
            for &word in row.iter().filter(|&&w| w != 0) {
                assert_eq!((word >> 12) & 3, 0);
                assert!((((word & 4095) - 1) as f64 / 4094. - 0.5).abs() <= 0.5 / 4094. + 1e-12);
                assert_eq!(word >> 14, 0x20100);
            }
        }
    }
    // A nearer X-dominant plane must not hide the eligible Z-normal plane.
    let source = input::Input {
        width: 1,
        height: 1,
        alpha: vec![255; 4],
        positions: vec![
            [-0.5, -0.5, 0., 0.],
            [0.5, -0.5, 0., 0.],
            [0.5, 0.5, 0., 0.],
            [-0.5, 0.5, 0., 0.],
            [-0.5, -0.5, -1.125, 0.],
            [0.5, -0.5, 0.875, 0.],
            [0.5, 0.5, 0.875, 0.],
            [-0.5, 0.5, -1.125, 0.],
        ],
        uvs: vec![[0.; 2]; 8],
        indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        parts: vec![[0, 12, (-1_f32).to_bits(), 1]],
        rays: vec![[0.; 8]],
        ..Default::default()
    };
    let scene = acceleration::build(device, queue, &source);
    let rows = cook_gpu::run(device, queue, pipeline, &scene, &tasks);
    assert!(rows.iter().flatten().all(|&word| word == 0x8040_0800));
    8
}
#[test]
#[ignore = "serialized immutable source spatial map bake"]
fn source_spatial_maps_partition_original_surfaces_without_inventing_coverage() {
    let geometry = std::env::var("SOLARIK_SOURCE_COVERAGE_INPUT").unwrap();
    let coverage = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let identity = identity("SOLARIK_SPATIAL_IDENTITY");
    let source_identity = self::identity("SOLARIK_COARSE_IDENTITY");
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let pipeline = cook_gpu::pipeline(device);
        let analytic = analytic(device, queue, &pipeline);
        let mut reports = Vec::new();
        for id in [5, 43] {
            let started = Instant::now();
            let mut original = input::load(&Path::new(&geometry).join(format!("source-{id}.bin")));
            assert_eq!(original.prototype, id);
            let (rays, views) = cameras::images(&original, 256);
            input::write(
                &Path::new(&output).join(format!("rays-{id}.bin")),
                bytemuck::cast_slice(&rays),
            );
            original.rays = vec![[0.; 8]];
            let scene = acceleration::build(device, queue, &original);
            let prepare_seconds = started.elapsed().as_secs_f64();
            for resolution in [16, 32] {
                let started = Instant::now();
                let bytes = std::fs::read(
                    Path::new(&coverage).join(format!("source-{id}-r{resolution}.sfcg")),
                )
                .unwrap();
                let source = coarse_cache::decode(&bytes, source_identity).unwrap();
                let mut cache = SpatialCache {
                    identity,
                    parts: original.parts.len() as u32,
                    two_sided: original
                        .parts
                        .iter()
                        .enumerate()
                        .fold(0, |mask, (i, p)| mask | (p[3] << i)),
                    samples: Vec::new(),
                };
                for cells in source.cells.chunks(4096) {
                    let tasks: Vec<[f32; 8]> = cells
                        .iter()
                        .flat_map(|c| {
                            coarse_cache::directions()[..6]
                                .iter()
                                .map(|d| {
                                    [
                                        c.centre_half[0],
                                        c.centre_half[1],
                                        c.centre_half[2],
                                        c.centre_half[3],
                                        d[0],
                                        d[1],
                                        d[2],
                                        8.,
                                    ]
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    let rows = cook_gpu::run(device, queue, &pipeline, &scene, &tasks);
                    cache.samples.extend(rows.into_iter().flatten());
                }
                assert!(
                    cache.valid(&source),
                    "source {id} r{resolution} exceeds original source axis coverage"
                );
                let mut material_hits = vec![0usize; cache.parts as usize];
                for &word in &cache.samples {
                    if word != 0 {
                        material_hits[((word >> 12) & 3) as usize] += 1;
                    }
                }
                assert!(
                    material_hits.iter().all(|&n| n > 0),
                    "lost original material"
                );
                let encoded = cache.encode(&source).unwrap();
                let decoded = SpatialCache::decode(&encoded, identity, &source).unwrap();
                assert_eq!(cache.samples, decoded.samples);
                assert!(SpatialCache::decode(&encoded, [255; 32], &source).is_none());
                assert!(
                    SpatialCache::decode(&encoded[..encoded.len() - 1], identity, &source)
                        .is_none()
                );
                let mut corrupt = encoded.clone();
                *corrupt.last_mut().unwrap() ^= 1;
                assert!(SpatialCache::decode(&corrupt, identity, &source).is_none());
                let unknown = cache.unknown_cells().iter().filter(|&&v| v != 0).count();
                let hits = cache.samples.iter().filter(|&&v| v != 0).count();
                input::write(
                    &Path::new(&output).join(format!("spatial-{id}-r{resolution}.sfsp")),
                    &encoded,
                );
                reports.push(format!(r#"{{"prototype":{id},"resolution":{resolution},"cells":{},"bytes":{},"hits":{hits},"unknown_cells":{unknown},"prepare_seconds":{prepare_seconds},"bake_host_seconds":{},"views":{views}}}"#,source.cells.len(),encoded.len(),started.elapsed().as_secs_f64()));
            }
        }
        input::write(
            &Path::new(&output).join("bake.json"),
            format!(
                r#"{{"levels":[{}],"analytic_controls":{analytic},"axis_coverage_subset":true}}"#,
                reports.join(",")
            )
            .as_bytes(),
        );
    });
}
