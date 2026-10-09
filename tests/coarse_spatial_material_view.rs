//! Cached source material views; warm runs need no original triangles or texture upload.
#[expect(
    dead_code,
    reason = "the reused visibility runner and decoder have their own fixture"
)]
#[path = "support/coarse_spatial_material_view_gpu.rs"]
mod gpu;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{
    coarse_appearance, coarse_cache,
    coarse_scene::{self, PackedSource},
    coarse_spatial::{self, SpatialCache},
    coarse_spatial_material::{MaterialCache, Surface},
    coarse_spatial_scene,
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
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Sample {
    surface: Surface,
    meta: [u32; 4],
}
fn identity(name: &str) -> [u8; 32] {
    let key = std::env::var(name).unwrap();
    assert_eq!(key.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap())
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
#[test]
#[ignore = "serialized small native original-material comparison"]
fn native_spatial_material_views() {
    let geometry = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let spatial_root = std::env::var("SOLARIK_SPATIAL_INPUT").unwrap();
    let materials = std::env::var("SOLARIK_MATERIAL_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let baseline = std::env::var("SOLARIK_SPATIAL_BASELINE").unwrap();
    let geometry_identity = identity("SOLARIK_COARSE_IDENTITY");
    let spatial_identity = identity("SOLARIK_SPATIAL_IDENTITY");
    let material_identity = identity("SOLARIK_MATERIAL_IDENTITY");
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            ..Default::default()
        })
        .await;
        let device = &resources.0;
        let queue = &resources.1;
        let shader = [
            coarse_scene::SHADER,
            coarse_scene::WALK_SHADER,
            coarse_spatial::PACK_SHADER,
            coarse_spatial::SHADER,
            coarse_spatial_scene::SHADER,
            coarse_appearance::SHADER,
            include_str!("coarse_spatial.wgsl"),
            include_str!("coarse_spatial_material_view.wgsl"),
        ]
        .map(|s| {
            s.lines()
                .filter(|l| !l.starts_with('#'))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .join("\n");
        let raw = device.wgpu_device();
        let module = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("material_view"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut reports = Vec::new();
        for id in [5, 43] {
            let rays =
                std::fs::read(Path::new(&spatial_root).join(format!("rays-{id}.bin"))).unwrap();
            for resolution in [16, 32] {
                let raw = std::fs::read(
                    Path::new(&geometry).join(format!("source-{id}-r{resolution}.sfcg")),
                )
                .unwrap();
                let source = coarse_cache::decode(&raw, geometry_identity).unwrap();
                let packed = PackedSource::decode(&raw, geometry_identity).unwrap();
                let raw = std::fs::read(
                    Path::new(&spatial_root).join(format!("spatial-{id}-r{resolution}.sfsp")),
                )
                .unwrap();
                let spatial = SpatialCache::decode(&raw, spatial_identity, &source).unwrap();
                let raw = std::fs::read(
                    Path::new(&materials).join(format!("material-{id}-r{resolution}.sfsh")),
                )
                .unwrap();
                let cache =
                    MaterialCache::decode(&raw, material_identity, &spatial, &source).unwrap();
                let started = Instant::now();
                let resident = cache.upload(device, &spatial, &source).unwrap();
                let mut settings = vec![spatial.parts, 8, spatial.two_sided, 0];
                settings.extend(spatial.unknown_cells());
                let values = gpu::run_material(
                    device,
                    queue,
                    &pipeline,
                    &[
                        (0, bytemuck::bytes_of(&packed.grid)),
                        (1, bytemuck::cast_slice(&packed.lookup)),
                        (2, bytemuck::cast_slice(&spatial.samples)),
                        (3, &rays),
                        (5, bytemuck::cast_slice(&settings)),
                    ],
                    &resident[1],
                    655360,
                );
                let elapsed = started.elapsed().as_secs_f64();
                let samples: &[Sample] = bytemuck::cast_slice(&values);
                let old = std::fs::read(
                    Path::new(&baseline).join(format!("coarse-{id}-r{resolution}.bin")),
                )
                .unwrap();
                assert_eq!(old.len(), samples.len() * 16);
                for (s, bytes) in samples.iter().zip(old.chunks_exact(16)) {
                    let previous: [f32; 4] = bytemuck::pod_read_unaligned(bytes);
                    assert_eq!(s.meta[1], 0);
                    assert_eq!(s.meta[0] as f32, previous[0]);
                    assert_eq!(f32::from_bits(s.meta[3]), previous[1]);
                    if s.meta[0] != 0 {
                        assert!(s.surface.valid());
                        assert!((1..=spatial.parts).contains(&s.meta[2]));
                    }
                }
                write(
                    &Path::new(&output).join(format!("material-view-{id}-r{resolution}.bin")),
                    bytemuck::cast_slice(samples),
                );
                reports.push(format!(r#"{{"prototype":{id},"resolution":{resolution},"rays":{},"attribute_gpu_bytes":{},"trace_upload_readback_seconds":{elapsed}}}"#,samples.len(),resident.iter().map(|b|b.size()).sum::<u64>()));
            }
        }
        write(
            &Path::new(&output).join("gpu.json"),
            format!(
                r#"{{"levels":[{}],"spatial_visibility_unchanged":true}}"#,
                reports.join(",")
            )
            .as_bytes(),
        );
    });
}
