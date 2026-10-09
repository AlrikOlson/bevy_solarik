//! Small native spatial-cache images; no original triangle AS on warm iteration.
#[path = "support/coarse_spatial_controls.rs"]
mod controls;
#[expect(
    dead_code,
    reason = "reuse buffer dispatch; its old transport pipeline is a separate target"
)]
#[path = "support/coarse_transport_gpu.rs"]
mod gpu;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{
    coarse_cache,
    coarse_scene::{self, Grid, PackedSource},
    coarse_spatial::{self, SpatialCache},
};
use std::{io::Write, path::Path, time::Instant};
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Probe {
    depth: [f32; 4],
    counts: [u32; 4],
    first: [u32; 4],
    normal: [f32; 4],
}
fn pipeline(device: &wgpu::Device, entry: &str) -> wgpu::ComputePipeline {
    let source = [
        coarse_scene::SHADER,
        coarse_scene::WALK_SHADER,
        coarse_spatial::PACK_SHADER,
        coarse_spatial::SHADER,
        include_str!("coarse_spatial.wgsl"),
    ]
    .map(|s| {
        s.lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .join("\n");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spatial source hit traversal"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
fn write(path: &Path, bytes: &[u8]) {
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}
fn identity(name: &str) -> [u8; 32] {
    let key = std::env::var(name).unwrap();
    assert_eq!(key.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap())
}
#[test]
#[ignore = "serialized actual Vulkan spatial visibility experiment"]
fn native_spatial_source_visibility() {
    let coverage = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let spatial = std::env::var("SOLARIK_SPATIAL_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let source_identity = identity("SOLARIK_COARSE_IDENTITY");
    let identity = identity("SOLARIK_SPATIAL_IDENTITY");
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let pipeline = pipeline(device, "main");
        let analytic = controls::check(device, queue, &pipeline);
        let mut reports = Vec::new();
        for id in [5, 43] {
            let rays = std::fs::read(Path::new(&spatial).join(format!("rays-{id}.bin"))).unwrap();
            assert_eq!(rays.len(), 655360 * 32);
            for resolution in [16, 32] {
                let raw = std::fs::read(
                    Path::new(&coverage).join(format!("source-{id}-r{resolution}.sfcg")),
                )
                .unwrap();
                let source = coarse_cache::decode(&raw, source_identity).unwrap();
                let packed = PackedSource::decode(&raw, source_identity).unwrap();
                let spatial = std::fs::read(
                    Path::new(&spatial).join(format!("spatial-{id}-r{resolution}.sfsp")),
                )
                .unwrap();
                let cache = SpatialCache::decode(&spatial, identity, &source).unwrap();
                let mut settings = vec![cache.parts, 8, cache.two_sided, 0];
                settings.extend(cache.unknown_cells());
                let started = Instant::now();
                let (rows, bytes) = gpu::run(
                    device,
                    queue,
                    &pipeline,
                    &[
                        (0, bytemuck::bytes_of(&packed.grid)),
                        (1, bytemuck::cast_slice(&packed.lookup)),
                        (2, bytemuck::cast_slice(&cache.samples)),
                        (3, &rays),
                        (5, bytemuck::cast_slice(&settings)),
                    ],
                    655360,
                );
                let elapsed = started.elapsed().as_secs_f64();
                let mut unresolved = 0;
                let mut hits = 0;
                let mut visited = 0u64;
                let values: Vec<[f32; 4]> = rows
                    .iter()
                    .map(|r| {
                        assert_eq!(r.counts[3], 0, "spatial traversal unresolved overflow");
                        assert!(r.depth[..2].iter().all(|v| v.is_finite()));
                        assert!(matches!(r.depth[0], 0. | 1.));
                        if r.depth[0] > 0. {
                            hits += 1;
                            assert!(r.first[0] > 0 && r.first[0] <= cache.parts);
                        }
                        unresolved += usize::from(r.counts[1] != 0);
                        visited += u64::from(r.counts[2]);
                        [r.depth[0], r.depth[1], r.counts[1] as f32, 0.]
                    })
                    .collect();
                write(
                    &Path::new(&output).join(format!("coarse-{id}-r{resolution}.bin")),
                    bytemuck::cast_slice(&values),
                );
                reports.push(format!(r#"{{"prototype":{id},"resolution":{resolution},"rays":655360,"hits":{hits},"unresolved_rays":{unresolved},"columns_visited":{visited},"explicit_fixture_bytes":{bytes},"trace_host_seconds":{elapsed}}}"#));
            }
        }
        write(&Path::new(&output).join("gpu.json"),format!(r#"{{"adapter":{:?},"sources":[{}],"side":256,"views":10,"analytic_controls":{analytic},"source_transport_accepted":false}}"#,format!("{:?}",**resources.2),reports.join(",")).as_bytes());
    });
}
