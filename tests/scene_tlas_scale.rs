//! Separate native TLAS encoding, finishing, submission and completion costs.
#![expect(
    clippy::print_stdout,
    reason = "The explicit GPU diagnostic emits measured TLAS stage timings."
)]
#[path = "../src/scene/tlas.rs"]
#[expect(
    dead_code,
    reason = "This GPU fixture uses only the production TLAS allocation and build subset."
)]
mod scene_tlas;
use bevy_render::{
    renderer::{RenderDevice, RenderQueue, initialize_headless_renderer},
    settings::WgpuSettings,
};
use std::time::Instant;
use wgpu::util::DeviceExt;

fn triangle(device: &RenderDevice, queue: &RenderQueue) -> wgpu::Blas {
    let raw = device.wgpu_device();
    let vertices: [f32; 9] = [-0.4, -0.4, 0.0, 0.4, -0.4, 0.0, 0.0, 0.4, 0.0];
    let buffer = raw.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::BLAS_INPUT,
    });
    let size = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = raw.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    let mut encoder = raw.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        &[wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &buffer,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        &[],
    );
    queue.submit([encoder.finish()]);
    blas
}
fn instance(blas: &wgpu::Blas, slot: u32, moved: bool) -> wgpu::TlasInstance {
    let x = if slot < 2 {
        slot as f32 * 2.0
    } else {
        10.0 + (slot % 1000) as f32 * 2.0
    };
    let y = (slot / 1000) as f32 * 2.0 + if moved { 2.0 } else { 0.0 };
    wgpu::TlasInstance::new(
        blas,
        [1., 0., 0., x, 0., 1., 0., y, 0., 0., 1., 0.],
        slot,
        255,
    )
}
fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}

#[test]
#[ignore = "Native full-scene TLAS cost diagnostic; run explicitly in release"]
fn scale_same_blas_build_cost() {
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let (device, queue) = (resources.0, resources.1);
        let raw = device.wgpu_device();
        let blas = triangle(&device, &queue);
        let other_blas = triangle(&device, &queue);
        let geometry = |slot: u32| {
            if slot.is_multiple_of(3) {
                &blas
            } else {
                &other_blas
            }
        };
        let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(r#"
enable wgpu_ray_query;
@group(0) @binding(0) var scene:acceleration_structure;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var query:ray_query;
    rayQueryInitialize(&query, scene, RayDesc(0u, 255u, 0.001, 10.0,
        vec3(f32(id.x) * 2.0, 0.0, -2.0), vec3(0.0, 0.0, 1.0)));
    while rayQueryProceed(&query) {}
    let hit = rayQueryGetCommittedIntersection(&query);
    output[id.x] = select(0u, hit.instance_custom_data + 1u, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
"#.into()),
        });
        let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = raw.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = raw.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut cache = scene_tlas::SceneTlas::default();
        for count in [1_000, 10_000, 100_000, 789_000] {
            let (tlas, _) = cache.prepare(&device, count);
            for slot in 0..count as u32 {
                scene_tlas::publish_slot(tlas, slot, Some(instance(geometry(slot), slot, false)));
            }
            let group = raw.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tlas.as_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            for iteration in 0..3 {
                let moved = iteration == 1;
                for slot in (0..count as u32).step_by(count / 64) {
                    scene_tlas::publish_slot(
                        tlas,
                        slot,
                        Some(instance(geometry(slot), slot, moved)),
                    );
                }
                let mut encoder = raw.create_command_encoder(&Default::default());
                let start = Instant::now();
                encoder.build_acceleration_structures(&[], [&*tlas]);
                let encode_ms = elapsed(start);
                let start = Instant::now();
                let commands = encoder.finish();
                let finish_ms = elapsed(start);
                let start = Instant::now();
                queue.submit([commands]);
                let submit_ms = elapsed(start);
                let start = Instant::now();
                raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let completion_ms = elapsed(start);
                println!(
                    "TLAS count={count} iteration={iteration} encode_ms={encode_ms:.3} finish_ms={finish_ms:.3} submit_ms={submit_ms:.3} completion_ms={completion_ms:.3}"
                );
                let mut encoder = raw.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups(2, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 8);
                queue.submit([encoder.finish()]);
                readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
                raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                {
                    let bytes = readback.get_mapped_range(..);
                    let expected = if moved { [0, 2] } else { [1, 2] };
                    assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), expected);
                }
                readback.unmap();
            }
        }
    });
}
