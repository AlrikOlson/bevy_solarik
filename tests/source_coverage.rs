//! Offline actual-source coverage/depth reference; no production rendering consumer.
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[path = "support/source_coverage_analytic.rs"]
mod analytic;
#[path = "support/source_coverage_gpu.rs"]
mod gpu;
#[path = "support/source_coverage_input.rs"]
mod input;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use std::{path::Path, time::Instant};

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Probe {
    depth: [f32; 4],
    counts: [u32; 4],
    first: [u32; 4],
    normal: [f32; 4],
}

#[test]
#[ignore = "serialized actual Vulkan alpha/source reference; needs exported source fixtures"]
fn source_coverage_matches_analytic_and_independent_cpu_rays() {
    let source = std::env::var("SOLARIK_SOURCE_COVERAGE_INPUT").expect("source fixture directory");
    let output = std::env::var("SOLARIK_SOURCE_COVERAGE_OUTPUT").expect("fresh output directory");
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let pipeline = gpu::pipeline(device);
        controls(device, queue, &pipeline);
        let mut reports = Vec::new();
        for id in [5, 43] {
            reports.push(probe_asset(
                device,
                queue,
                &pipeline,
                Path::new(&source),
                Path::new(&output),
                id,
            ));
        }
        let report = format!(
            "{{\"adapter\":{:?},\"sources\":[{}],\"analytic_controls\":8,\"overflow_detected\":true}}\n",
            format!("{:?}", **resources.2),
            reports.join(",")
        );
        input::write(&Path::new(&output).join("gpu.json"), report.as_bytes());
    });
}

fn controls(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline) {
    let input = analytic::scene();
    let scene = acceleration::build(device, queue, &input);
    let rows = gpu::trace(device, queue, pipeline, &scene, input.rays.len());
    analytic::check(&input, &rows, 1e-5);
    analytic::validate(&input, &rows);
    let input = analytic::overflow();
    let scene = acceleration::build(device, queue, &input);
    let rows = gpu::trace(device, queue, pipeline, &scene, 1);
    assert_eq!(rows[0].counts[0], 4097);
    assert_eq!(rows[0].counts[3], 1, "overflow must remain explicit");
}

fn probe_asset(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    source: &Path,
    output: &Path,
    id: u32,
) -> String {
    let started = Instant::now();
    let input = input::load(&source.join(format!("source-{id}.bin")));
    assert_eq!(input.prototype, id);
    let scene = acceleration::build(device, queue, &input);
    let rows = gpu::trace(device, queue, pipeline, &scene, input.rays.len());
    analytic::check(&input, &rows, 1e-4);
    analytic::validate(&input, &rows);
    assert!(
        rows.iter().any(|r| r.counts[2] > 0),
        "source has no leaf hits"
    );
    input::write(
        &output.join(format!("source-{id}.readback")),
        bytemuck::cast_slice(&rows),
    );
    format!(
        "{{\"prototype\":{id},\"rays\":{},\"triangles\":{},\"input_bytes\":{},\"output_and_readback_bytes\":{},\"seconds\":{},\"controls\":{}}}",
        rows.len(),
        input.indices.len() / 3,
        scene.explicit_bytes,
        rows.len() * 128,
        started.elapsed().as_secs_f64(),
        input.expected.len()
    )
}
