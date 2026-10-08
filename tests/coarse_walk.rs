//! Independent f64 interval reference for the shared bounded GPU coarse walker.
#[path = "support/coarse_scene_gpu.rs"]
mod gpu;
#[path = "support/coarse_walk_oracle.rs"]
mod oracle;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{
    coarse_cache,
    coarse_scene::{self, PackedSource},
};
use std::{io::Write, path::Path, time::Instant};
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Request {
    origin: [f32; 4],
    direction: [f32; 4],
}
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Probe {
    counts: [u32; 4],
    intervals: [[u32; 8]; 512],
}

#[test]
#[ignore = "serialized actual Vulkan sparse coarse interval walk"]
fn sparse_coarse_intervals_match_independent_f64_cell_intersections() {
    let source = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let key = std::env::var("SOLARIK_COARSE_IDENTITY").unwrap();
    assert_eq!(key.len(), 64);
    let identity =
        core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap());
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            ..Default::default()
        })
        .await;
        let shared = coarse_scene::WALK_SHADER
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let entry = format!("{shared}\n{}", include_str!("coarse_walk.wgsl"));
        let pipeline = gpu::pipeline(resources.0.wgpu_device(), &entry);
        let (analytic, rays) = oracle::analytic();
        let bytes = coarse_cache::encode(&analytic).unwrap();
        let packed = PackedSource::decode(&bytes, analytic.identity).unwrap();
        let (rows, _) = gpu::run(&resources.0, &resources.1, &pipeline, &packed, &rays);
        let analytic_error = oracle::compare(&analytic, &rays, &rows);
        let controls = rays.len();
        let invalid = [
            Request {
                origin: [0., 0., 0., 0.],
                direction: [0., 0., 0., 1.],
            },
            Request {
                origin: [0., 0., 0., 1.],
                direction: [1., 0., 0., 0.5],
            },
        ];
        let (invalid_rows, _) = gpu::run(&resources.0, &resources.1, &pipeline, &packed, &invalid);
        for row in invalid_rows {
            assert_eq!(row.counts[0], 0);
            assert_eq!(row.counts[2], 2);
        }
        let (overflow, rays) = oracle::overflow();
        let packed =
            PackedSource::decode(&coarse_cache::encode(&overflow).unwrap(), overflow.identity)
                .unwrap();
        let (rows, _) = gpu::run(&resources.0, &resources.1, &pipeline, &packed, &rays);
        assert_eq!(rows[0].counts[1], 512);
        assert_eq!(
            rows[0].counts[2], 1,
            "exhaustion must not become a complete miss"
        );
        let mut reports = Vec::new();
        for id in [5, 43] {
            for resolution in [16, 32] {
                let started = Instant::now();
                let path = Path::new(&source).join(format!("source-{id}-r{resolution}.sfcg"));
                assert!(path.metadata().unwrap().len() <= coarse_cache::MAX_CACHE_BYTES as u64);
                let bytes = std::fs::read(path).unwrap();
                let original = coarse_cache::decode(&bytes, identity).unwrap();
                let packed = PackedSource::decode(&bytes, identity).unwrap();
                let rays = oracle::real_rays(&original);
                let (rows, total_gpu_bytes) =
                    gpu::run(&resources.0, &resources.1, &pipeline, &packed, &rays);
                let maximum_error = oracle::compare(&original, &rays, &rows);
                let intervals: u32 = rows.iter().map(|r| r.counts[0]).sum();
                assert!(intervals > 0);
                reports.push(format!(
                "{{\"prototype\":{id},\"resolution\":{resolution},\"rays\":{},\"intervals\":{intervals},\"maximum_depth_error\":{maximum_error},\"gpu_bytes\":{total_gpu_bytes},\"seconds\":{}}}",
                rays.len(),started.elapsed().as_secs_f64()));
            }
        }
        let report = format!(
            "{{\"adapter\":{:?},\"sources\":[{}],\"analytic_controls\":{controls},\"analytic_error\":{analytic_error},\"overflow_detected\":true}}\n",
            format!("{:?}", **resources.2),
            reports.join(",")
        );
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(Path::new(&output).join("walk.json"))
            .unwrap();
        file.write_all(report.as_bytes()).unwrap();
    });
}
