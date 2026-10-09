//! Source material samples for the exact immutable spatial maps.
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[path = "support/coarse_spatial_material_gpu.rs"]
mod appearance_gpu;
#[path = "support/coarse_appearance_input.rs"]
mod appearance_input;
#[path = "support/coarse_appearance_oracle.rs"]
mod appearance_oracle;
#[path = "support/coarse_appearance_controls.rs"]
mod controls;
#[path = "support/source_coverage_input.rs"]
mod input;
#[path = "support/coarse_spatial_material_controls.rs"]
mod spatial_controls;
#[path = "support/coarse_appearance_tangents.rs"]
mod tangents;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{
    coarse_cache,
    coarse_spatial::SpatialCache,
    coarse_spatial_material::{MaterialCache, Surface},
};
use std::{path::Path, time::Instant};
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
#[test]
#[ignore = "serialized native original-material spatial bake"]
fn spatial_source_materials_match_original_hit_selection() {
    let geometry = std::env::var("SOLARIK_SOURCE_COVERAGE_INPUT").unwrap();
    let appearance = std::env::var("SOLARIK_APPEARANCE_INPUT").unwrap();
    let coverage = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let spatial_root = std::env::var("SOLARIK_SPATIAL_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let references = std::env::var("SOLARIK_ORIGINAL_IMAGES").unwrap();
    let identity = identity("SOLARIK_MATERIAL_IDENTITY");
    let coverage_identity = self::identity("SOLARIK_COARSE_IDENTITY");
    let spatial_identity = self::identity("SOLARIK_SPATIAL_IDENTITY");
    let levels = coarse_cache::parse_levels(
        &std::env::var("SOLARIK_COARSE_LEVELS").unwrap_or_else(|_| "16,32".into()),
    )
    .expect("bounded spatial material levels");
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let cook = appearance_gpu::spatial_pipeline(device, "main");
        let reference = appearance_gpu::spatial_pipeline(device, "reference");
        let analytic = controls::check(
            device,
            queue,
            &appearance_gpu::pipeline(device, "main"),
            &appearance_gpu::pipeline(device, "sample_material"),
        );
        let spatial_controls = spatial_controls::check(device, queue, &cook);
        let probe = appearance_gpu::pipeline(device, "sample_material");
        let mut reports = Vec::new();
        let mut reference_reports = Vec::new();
        for id in [5, 43] {
            let started = Instant::now();
            let mut geometry = input::load(&Path::new(&geometry).join(format!("source-{id}.bin")));
            let material = appearance_input::load(
                &Path::new(&appearance).join(format!("appearance-{id}.bin")),
                &geometry,
            );
            geometry.rays = vec![[0.; 8]; 32768];
            let scene = acceleration::build(device, queue, &geometry);
            let uploaded = appearance_gpu::upload(device, queue, &material);
            let (rows, _) = appearance_gpu::run::<appearance_oracle::Sample>(
                device,
                queue,
                &probe,
                &scene,
                &uploaded,
                material.requests.len(),
                Some(&material.requests),
            );
            let (texture_error, math_error) = appearance_oracle::compare(&material, &rows);
            let prepare_seconds = started.elapsed().as_secs_f64();
            for &resolution in &levels {
                let started = Instant::now();
                let raw = std::fs::read(
                    Path::new(&coverage).join(format!("source-{id}-r{resolution}.sfcg")),
                )
                .unwrap();
                let source = coarse_cache::decode(&raw, coverage_identity).unwrap();
                let raw = std::fs::read(
                    Path::new(&spatial_root).join(format!("spatial-{id}-r{resolution}.sfsp")),
                )
                .unwrap();
                let mut spatial = SpatialCache::decode(&raw, spatial_identity, &source).unwrap();
                let mut cache = MaterialCache {
                    identity,
                    materials: material.materials.clone(),
                    surfaces: Vec::new(),
                };
                let mut offset = 0;
                let mut gpu_bytes = 0;
                for cells in source.cells.chunks(512) {
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
                    queue.write_buffer(&scene.buffers[4], 0, bytemuck::cast_slice(&tasks));
                    let (rows, bytes) = appearance_gpu::run::<Sample>(
                        device,
                        queue,
                        &cook,
                        &scene,
                        &uploaded,
                        tasks.len() * 64,
                        None,
                    );
                    gpu_bytes = gpu_bytes.max(bytes);
                    for row in rows {
                        assert_eq!(
                            row.meta[1], 0,
                            "invalid original frame or traversal overflow"
                        );
                        assert_eq!(
                            row.meta[0], spatial.samples[offset],
                            "changed source hit {id}/{resolution}/{offset}"
                        );
                        if row.meta[0] != 0 {
                            assert_eq!(row.meta[2], ((row.meta[0] >> 12) & 3) + 1);
                            assert!(
                                row.surface.valid(),
                                "invalid source material {id}/{resolution}/{offset}"
                            );
                            cache.surfaces.push(row.surface);
                        }
                        offset += 1;
                    }
                }
                assert_eq!(offset, spatial.samples.len());
                assert!(cache.valid(&spatial, &source));
                let encoded = cache.encode(&spatial, &source).unwrap();
                let decoded = MaterialCache::decode(&encoded, identity, &spatial, &source).unwrap();
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&cache.surfaces),
                    bytemuck::cast_slice::<_, u8>(&decoded.surfaces)
                );
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&cache.materials),
                    bytemuck::cast_slice::<_, u8>(&decoded.materials)
                );
                assert!(MaterialCache::decode(&encoded, [255; 32], &spatial, &source).is_none());
                assert!(
                    MaterialCache::decode(
                        &encoded[..encoded.len() - 1],
                        identity,
                        &spatial,
                        &source
                    )
                    .is_none()
                );
                let mut corrupt = encoded.clone();
                *corrupt.last_mut().unwrap() ^= 1;
                assert!(MaterialCache::decode(&corrupt, identity, &spatial, &source).is_none());
                spatial.identity[0] ^= 1;
                assert!(MaterialCache::decode(&encoded, identity, &spatial, &source).is_none());
                spatial.identity[0] ^= 1;
                input::write(
                    &Path::new(&output).join(format!("material-{id}-r{resolution}.sfsh")),
                    &encoded,
                );
                reports.push(format!(r#"{{"prototype":{id},"resolution":{resolution},"samples":{offset},"hits":{},"bytes":{},"gpu_bytes":{gpu_bytes},"texture_error":{texture_error},"math_error":{math_error},"prepare_seconds":{prepare_seconds},"bake_seconds":{}}}"#,cache.surfaces.len(),encoded.len(),started.elapsed().as_secs_f64()));
            }
            let started = Instant::now();
            let bytes =
                std::fs::read(Path::new(&spatial_root).join(format!("rays-{id}.bin"))).unwrap();
            let rays: Vec<[f32; 8]> = bytes
                .chunks_exact(32)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            assert_eq!(rays.len(), 655360);
            let old =
                std::fs::read(Path::new(&references).join(format!("reference-{id}.bin"))).unwrap();
            let expected: Vec<[f32; 4]> = old
                .chunks_exact(16)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            assert_eq!(expected.len(), rays.len());
            let mut values = Vec::new();
            for batch in rays.chunks(32768) {
                queue.write_buffer(&scene.buffers[4], 0, bytemuck::cast_slice(batch));
                let (rows, _) = appearance_gpu::run::<Sample>(
                    device,
                    queue,
                    &reference,
                    &scene,
                    &uploaded,
                    batch.len(),
                    None,
                );
                for row in rows {
                    assert_eq!(row.meta[1], 0);
                    let reference = expected[values.len()];
                    assert_eq!(f32::from(u8::from(row.meta[0] != 0)), reference[0]);
                    if row.meta[0] != 0 {
                        assert!(row.surface.valid());
                        assert!((f32::from_bits(row.meta[3]) - reference[1]).abs() < 1e-5);
                    }
                    values.push(row);
                }
            }
            input::write(
                &Path::new(&output).join(format!("reference-material-{id}.bin")),
                bytemuck::cast_slice(&values),
            );
            reference_reports.push(format!(
                r#"{{"prototype":{id},"rays":{},"seconds":{}}}"#,
                values.len(),
                started.elapsed().as_secs_f64()
            ));
        }
        input::write(&Path::new(&output).join("bake.json"),format!(r#"{{"levels":[{}],"references":[{}],"analytic_controls":{analytic},"spatial_controls":{spatial_controls},"all_geometry_words_exact":true}}"#,reports.join(","),reference_reports.join(",")).as_bytes());
    });
}
