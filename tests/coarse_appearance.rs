//! Immutable source-coupled appearance cook, using original Bevy tangent generation.
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[path = "support/coarse_appearance_gpu.rs"]
mod appearance_gpu;
#[path = "support/coarse_appearance_input.rs"]
mod appearance_input;
#[path = "support/coarse_appearance_oracle.rs"]
mod appearance_oracle;
#[path = "support/coarse_appearance_controls.rs"]
mod controls;
#[path = "support/coarse_source_grid.rs"]
mod grid;
#[path = "support/source_coverage_input.rs"]
mod input;
#[path = "support/coarse_appearance_tangents.rs"]
mod tangents;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::{
    coarse_appearance::{self, AppearanceCache, Moment},
    coarse_cache,
};
use std::{path::Path, time::Instant};
fn identity(name: &str) -> [u8; 32] {
    let key = std::env::var(name).unwrap();
    assert_eq!(key.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap())
}
#[test]
#[ignore = "serialized actual Vulkan appearance cook"]
fn original_materials_cook_without_changing_geometric_coverage() {
    let geometry = std::env::var("SOLARIK_SOURCE_COVERAGE_INPUT").unwrap();
    let appearance = std::env::var("SOLARIK_APPEARANCE_INPUT").unwrap();
    let coverage = std::env::var("SOLARIK_COARSE_INPUT").unwrap();
    let output = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let identity = identity("SOLARIK_APPEARANCE_IDENTITY");
    let coverage_identity = self::identity("SOLARIK_COARSE_IDENTITY");
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let cook = appearance_gpu::pipeline(device, "main");
        let probe = appearance_gpu::pipeline(device, "sample_material");
        let analytic = controls::check(device, queue, &cook, &probe);
        let mut reports = Vec::new();
        for id in [5, 43] {
            let started = Instant::now();
            let mut geometry = input::load(&Path::new(&geometry).join(format!("source-{id}.bin")));
            let material = appearance_input::load(
                &Path::new(&appearance).join(format!("appearance-{id}.bin")),
                &geometry,
            );
            geometry.rays = vec![[0.; 8]; grid::BATCH_TASKS];
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
            let prepared_seconds = started.elapsed().as_secs_f64();
            for resolution in [16, 32] {
                let started = Instant::now();
                let raw = std::fs::read(
                    Path::new(&coverage).join(format!("source-{id}-r{resolution}.sfcg")),
                )
                .unwrap();
                let source = coarse_cache::decode(&raw, coverage_identity).unwrap();
                let mut cache = AppearanceCache {
                    identity,
                    materials: material.materials.clone(),
                    moments: Vec::new(),
                };
                let mut gpu_bytes = 0;
                for cells in source.cells.chunks(grid::BATCH_CELLS) {
                    let tasks = grid::tasks(cells, source.samples_side);
                    queue.write_buffer(&scene.buffers[4], 0, bytemuck::cast_slice(&tasks));
                    let (rows, bytes) = appearance_gpu::run::<[Moment; 8]>(
                        device,
                        queue,
                        &cook,
                        &scene,
                        &uploaded,
                        tasks.len(),
                        None,
                    );
                    gpu_bytes = gpu_bytes.max(bytes);
                    for row in rows {
                        cache
                            .moments
                            .extend_from_slice(&row[..cache.materials.len()]);
                    }
                }
                assert!(
                    cache.valid(&source),
                    "source{id} r{resolution} material counts/moments differ"
                );
                let encoded = coarse_appearance::encode(&cache, &source).unwrap();
                let decoded = coarse_appearance::decode(&encoded, identity, &source).unwrap();
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&decoded.moments),
                    bytemuck::cast_slice::<_, u8>(&cache.moments)
                );
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&decoded.materials),
                    bytemuck::cast_slice::<_, u8>(&cache.materials)
                );
                let mut material_counts = vec![0u64; cache.materials.len()];
                for row in cache.moments.chunks_exact(cache.materials.len()) {
                    for (count, m) in material_counts.iter_mut().zip(row) {
                        *count += u64::from(m.counts[0]);
                    }
                }
                assert!(material_counts.iter().all(|&n| n > 0));
                input::write(
                    &Path::new(&output).join(format!("appearance-{id}-r{resolution}.sfca")),
                    &encoded,
                );
                reports.push(format!(r#"{{"prototype":{id},"resolution":{resolution},"cells":{},"moments":{},"bytes":{},"material_counts":{material_counts:?},"gpu_bytes":{gpu_bytes},"texture_error":{texture_error},"math_error":{math_error},"prepare_seconds":{prepared_seconds},"measure_seconds":{}}}"#,
                    source.cells.len(),cache.moments.len(),encoded.len(),started.elapsed().as_secs_f64()));
            }
        }
        let report = format!(
            r#"{{"adapter":{:?},"levels":[{}],"analytic_controls":{analytic},"source_coverage_counts_exact":true}}"#,
            format!("{:?}", **resources.2),
            reports.join(",")
        );
        input::write(&Path::new(&output).join("gpu.json"), report.as_bytes());
    });
}
