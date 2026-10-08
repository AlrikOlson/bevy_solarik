//! Real BLAS compaction alongside scheduled asset uploads, followed by ray readback.
use super::{BlasManager, compact_raytracing_blas};
use bevy_asset::AssetId;
use bevy_ecs::system::Res;
use bevy_ecs::{
    schedule::Schedule,
    system::{IntoSystem, System},
    world::World,
};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use wgpu::util::DeviceExt;

fn upload(device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    let raw = device.wgpu_device();
    let source = raw.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("concurrent asset upload"),
        contents: &[7u8; 4096],
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let destination = raw.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = raw.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&source, 0, &destination, 0, 4096);
    queue.submit([encoder.finish()]);
}

#[test]
#[ignore = "requires Vulkan ray query adapter"]
fn compaction_upload_schedule_keeps_compacted_geometry_visible() {
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let (device, queue) = (resources.0, resources.1);
        let raw = device.wgpu_device();
        let vertices: [f32; 9] = [-0.4, -0.4, 0.0, 0.4, -0.4, 0.0, 0.0, 0.4, 0.0];
        let vertex_buffer = raw.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        let mut manager = BlasManager::default();
        let ids: Vec<_> = (1..=8)
            .map(|n| AssetId::from(bevy_asset::uuid::Uuid::from_u128(n)))
            .collect();
        let mut originals = Vec::new();
        for &id in &ids {
            let blas = raw.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("scheduled compactable triangle"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
                        | wgpu::AccelerationStructureFlags::ALLOW_COMPACTION,
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
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &size,
                            vertex_buffer: &vertex_buffer,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: None,
                            first_index: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                }],
                &[],
            );
            queue.submit([encoder.finish()]);
            originals.push(blas.clone());
            manager.blas.insert(id, blas);
            manager.compaction_queue.push_back((id, 3, false));
        }
        let mut world = World::new();
        world.insert_resource(device.clone());
        world.insert_resource(queue.clone());
        world.insert_resource(manager);
        let compaction = IntoSystem::into_system(compact_raytracing_blas);
        // Fail immediately on a scheduling regression instead of hanging in wgpu.
        assert!(compaction.is_exclusive());
        let mut schedule = Schedule::default();
        schedule.add_systems((compaction, upload, upload, upload, upload));
        for _ in 0..4 {
            schedule.run(&mut world);
            queue.submit([]);
            raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if world.resource::<BlasManager>().compaction_queue.is_empty() {
                break;
            }
        }
        let manager = world.resource::<BlasManager>();
        assert!(
            manager.compaction_queue.is_empty(),
            "compaction must finish, not be disabled"
        );
        assert_eq!(manager.generation, ids.len() as u64);
        let mut tlas = raw.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            max_instances: ids.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        for (n, id) in ids.iter().enumerate() {
            let compacted = manager.get(id).unwrap();
            assert_ne!(compacted, &originals[n]);
            let transform = [
                1.0,
                0.0,
                0.0,
                n as f32 * 2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ];
            *tlas.get_mut_single(n).unwrap() = Some(wgpu::TlasInstance::new(
                compacted,
                transform,
                n as u32 + 23,
                255,
            ));
        }
        let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
enable wgpu_ray_query;
@group(0) @binding(0) var scene:acceleration_structure;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    var query:ray_query;
    rayQueryInitialize(&query,scene,RayDesc(0u,255u,0.001,10.0,
        vec3(f32(id.x)*2.0,0.0,-2.0),vec3(0.0,0.0,1.0)));
    while rayQueryProceed(&query) {}
    let hit=rayQueryGetCommittedIntersection(&query);
    output[id.x]=select(0u,hit.instance_custom_data+1u,hit.kind!=RAY_QUERY_INTERSECTION_NONE);
}
"#
                .into(),
            ),
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
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = raw.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let mut encoder = raw.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(&[], [&tlas]);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(8, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&readback.get_mapped_range(..)),
            &[24, 25, 26, 27, 28, 29, 30, 31]
        );
        readback.unmap();
    });
}
