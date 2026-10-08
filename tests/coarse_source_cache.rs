//! Bounded original-source cook. No game launch or scientific population work.
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[path = "support/coarse_source_analytic.rs"]
mod analytic;
#[path = "support/source_coverage_gpu.rs"]
mod gpu;
#[path = "support/coarse_source_grid.rs"]
mod grid;
#[path = "support/source_coverage_input.rs"]
mod input;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use bevy_solarik::coarse_cache::{self, DIRECTIONS, SourceCache};
use std::{path::Path, time::Instant};
type Probe = coarse_cache::Directional;

#[test]
#[ignore = "serialized actual Vulkan source cook; needs sealed fixture and fresh output"]
fn cook_original_source_cells() {
    let source = std::env::var("SOLARIK_SOURCE_COVERAGE_INPUT").expect("fixture directory");
    let output = std::env::var("SOLARIK_SOURCE_COVERAGE_OUTPUT").expect("fresh output directory");
    let key = std::env::var("SOLARIK_COARSE_IDENTITY").expect("sealed cook identity");
    assert_eq!(key.len(), 64);
    let identity =
        core::array::from_fn(|i| u8::from_str_radix(&key[i * 2..i * 2 + 2], 16).unwrap());
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let pipeline = gpu::pipeline(device, include_str!("coarse_source_cache.wgsl"));
        let controls = analytic::controls(device, queue, &pipeline);
        let mut reports = Vec::new();
        for id in [5, 43] {
            let started = Instant::now();
            let mut input = input::load(&Path::new(&source).join(format!("source-{id}.bin")));
            assert_eq!(input.prototype, id);
            input.rays = vec![[0.; 8]; grid::BATCH_TASKS];
            let scene = acceleration::build(device, queue, &input);
            let explicit_bytes =
                scene.explicit_bytes + (grid::BATCH_TASKS * size_of::<Probe>() * 2) as u64;
            assert!(
                explicit_bytes < 512 << 20,
                "explicit GPU resource budget exceeded"
            );
            let prepared_seconds = started.elapsed().as_secs_f64();
            for resolution in [16, 32] {
                let started = Instant::now();
                let mut cells = grid::cells(&input, resolution);
                let grid_seconds = started.elapsed().as_secs_f64();
                for batch in cells.chunks_mut(grid::BATCH_CELLS) {
                    let tasks = grid::tasks(batch, 8);
                    queue.write_buffer(&scene.buffers[4], 0, bytemuck::cast_slice(&tasks));
                    let rows = gpu::trace(device, queue, &pipeline, &scene, tasks.len());
                    for (cell, rows) in batch.iter_mut().zip(rows.chunks_exact(DIRECTIONS)) {
                        for (direction, row) in rows.iter().enumerate() {
                            assert!(
                                row.validate(8),
                                "source{id} grid{:?} direction{direction}: {row:?}",
                                cell.grid
                            );
                        }
                        cell.directions.copy_from_slice(rows);
                    }
                }
                let measured_seconds = started.elapsed().as_secs_f64();
                let mut materials = [0u64; 8];
                let mut support = 0u64;
                let mut covered = 0u64;
                let mut events = 0u64;
                let mut sampled_empty_cells = 0usize;
                for cell in &cells {
                    sampled_empty_cells +=
                        usize::from(cell.directions.iter().all(|d| d.counts[1] == 0));
                    for bin in &cell.directions {
                        support += u64::from(bin.counts[0]);
                        covered += u64::from(bin.counts[1]);
                        events += u64::from(bin.counts[2]);
                        for (total, count) in materials.iter_mut().zip(bin.materials) {
                            *total += u64::from(count);
                        }
                    }
                }
                for (part, count) in input.parts.iter().zip(materials) {
                    assert!(
                        count > 0,
                        "source{id} unmeasured original material {part:?}"
                    );
                }
                let cache = SourceCache {
                    identity,
                    prototype: id,
                    resolution,
                    samples_side: 8,
                    cells,
                };
                let bytes = coarse_cache::encode(&cache).expect("valid cache");
                let roundtrip = coarse_cache::decode(&bytes, identity).expect("cache roundtrip");
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&roundtrip.cells),
                    bytemuck::cast_slice::<_, u8>(&cache.cells)
                );
                input::write(
                    &Path::new(&output).join(format!("source-{id}-r{resolution}.sfcg")),
                    &bytes,
                );
                reports.push(format!(
                    "{{\"prototype\":{id},\"resolution\":{resolution},\"cells\":{},\"sampled_empty_cells\":{sampled_empty_cells},\"support\":{support},\"covered\":{covered},\"events\":{events},\"materials\":{materials:?},\"bytes\":{},\"explicit_gpu_bytes\":{explicit_bytes},\"prepare_seconds\":{prepared_seconds},\"grid_seconds\":{grid_seconds},\"measure_seconds\":{measured_seconds}}}",
                    cache.cells.len(),bytes.len()));
            }
        }
        let report = format!(
            "{{\"adapter\":{:?},\"levels\":[{}],\"analytic_controls\":{controls}}}\n",
            format!("{:?}", **resources.2),
            reports.join(",")
        );
        input::write(&Path::new(&output).join("gpu.json"), report.as_bytes());
    });
}
