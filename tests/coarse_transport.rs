//! Isolated native source/coarse transmission images, not full scene acceptance.
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[path = "support/coarse_transport_controls.rs"]
mod controls;
#[path = "support/coarse_transport_gpu.rs"]
mod gpu;
#[path = "support/source_coverage_input.rs"]
mod input;
#[path = "support/source_coverage_gpu.rs"]
mod reference;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{coarse_cache, coarse_scene::PackedSource, coarse_transport};
use std::{path::Path, time::Instant};
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Probe {
    depth: [f32; 4],
    counts: [u32; 4],
    first: [u32; 4],
    normal: [f32; 4],
}
#[test]
#[ignore = "serial native Vulkan original/coarse transmission diagnostic"]
fn native_original_and_coarse_transport_images() {
    let geometry = std::env::var("SOLARIK_SOURCE_COVERAGE_INPUT").unwrap();
    let coverage = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let key = std::env::var("SOLARIK_COARSE_IDENTITY").unwrap();
    let identity =
        core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap());
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let pipeline = gpu::pipeline(device, "main");
        let analytic = controls::check(device, queue, &pipeline);
        let reference = reference::pipeline(device, include_str!("source_coverage.wgsl"));
        let mut reports = Vec::new();
        for id in [5, 43] {
            let started = Instant::now();
            let mut original = input::load(&Path::new(&geometry).join(format!("source-{id}.bin")));
            assert_eq!(original.prototype, id);
            let (rays, views) = controls::images(&original, 256);
            original.rays = rays;
            let scene = acceleration::build(device, queue, &original);
            let prepared = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let exact = reference::trace(device, queue, &reference, &scene, original.rays.len());
            let reference_seconds = started.elapsed().as_secs_f64();
            controls::save(
                &Path::new(&output).join(format!("reference-{id}.bin")),
                &exact,
                true,
            );
            drop(exact);
            for resolution in [16, 32] {
                let started = Instant::now();
                let raw = std::fs::read(
                    Path::new(&coverage).join(format!("source-{id}-r{resolution}.sfcg")),
                )
                .unwrap();
                let source = coarse_cache::decode(&raw, identity).unwrap();
                let packed = PackedSource::decode(&raw, identity).unwrap();
                let kernels = coarse_transport::prepare(&source).expect("source fitting");
                let fit_seconds = started.elapsed().as_secs_f64();
                let started = Instant::now();
                let (rows, gpu_bytes) = gpu::run(
                    device,
                    queue,
                    &pipeline,
                    &[
                        (0, bytemuck::bytes_of(&packed.grid)),
                        (1, bytemuck::cast_slice(&packed.lookup)),
                        (2, bytemuck::cast_slice(&kernels)),
                        (3, bytemuck::cast_slice(&original.rays)),
                    ],
                    original.rays.len(),
                );
                let trace_seconds = started.elapsed().as_secs_f64();
                controls::save(
                    &Path::new(&output).join(format!("coarse-{id}-r{resolution}.bin")),
                    &rows,
                    false,
                );
                let unresolved = rows.iter().filter(|r| r.counts[1] != 0).count();
                let interior: usize = kernels
                    .iter()
                    .map(|k| k.measured.count_ones() as usize)
                    .sum();
                let kernel_bytes = kernels.len() * size_of::<coarse_transport::Kernel>()
                    + packed.lookup.len() * 4
                    + size_of_val(&packed.grid);
                reports.push(format!(r#"{{"prototype":{id},"resolution":{resolution},"rays":{},"cells":{},"interior_bins":{interior},"total_bins":{},"unresolved_rays":{unresolved},"kernel_bytes":{kernel_bytes},"explicit_fixture_bytes":{},"fit_seconds":{fit_seconds},"trace_host_seconds":{trace_seconds},"source_prepare_seconds":{prepared},"reference_trace_host_seconds":{reference_seconds},"views":{views}}}"#,
                    original.rays.len(),kernels.len(),kernels.len()*14,gpu_bytes+scene.explicit_bytes));
            }
        }
        input::write(&Path::new(&output).join("gpu.json"),format!(r#"{{"adapter":{:?},"side":256,"views":10,"analytic_controls":{analytic},"sources":[{}],"source_transport_accepted":false}}"#,
            format!("{:?}",**resources.2),reports.join(",")).as_bytes());
    });
}
