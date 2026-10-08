//! Actual ray visibility through the production persistent allocation helper.
#[path = "../src/scene/tlas.rs"]
mod scene_tlas;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use wgpu::util::DeviceExt;

#[test]
fn persistent_tlas_preserves_sparse_custom_slots_through_readiness_empty_return_and_reuse() {
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

        let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(r#"
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
        let mut original = None;
        let mut retained_group = None;
        let mut previous_dense: Vec<(f32, u32)> = Vec::new();
        for (required, full, dense, removals, positions, expected) in [
            (2, true, true, vec![], vec![(0.0, 23), (2.0, 5)], [24, 6]),
            // Removing an early physical descriptor displaces a clean survivor.
            (1, false, true, vec![], vec![(2.0, 5)], [0, 6]),
            (0, false, true, vec![], vec![], [0, 0]),
            (2, false, true, vec![], vec![(2.0, 5), (0.0, 23)], [24, 6]),
            (2, false, true, vec![], vec![(0.0, 5), (2.0, 23)], [6, 24]),
            (2, false, true, vec![], vec![(2.0, 23), (0.0, 5)], [6, 24]),
            // Full readiness publication follows the final active order.
            (
                3,
                true,
                true,
                vec![],
                vec![(0.0, 23), (9.0, 4099), (2.0, 5)],
                [24, 6],
            ),
            (2, true, true, vec![], vec![(0.0, 23), (2.0, 5)], [24, 6]),
            // A high custom ID does not require a high physical address.
            (1, false, true, vec![], vec![(0.0, 4099)], [4100, 0]),
            (1, false, true, vec![], vec![(2.0, 4099)], [0, 4100]),
            (1, false, true, vec![], vec![(2.0, 4099)], [0, 4100]),
            (24, true, false, vec![], vec![(0.0, 23), (2.0, 5)], [24, 6]),
            (24, false, false, vec![23], vec![], [0, 6]),
            (24, false, false, vec![5], vec![], [0, 0]),
            (24, false, false, vec![], vec![(0.0, 23), (2.0, 5)], [24, 6]),
            (24, false, false, vec![], vec![(2.0, 23), (0.0, 5)], [6, 24]),
            (24, false, false, vec![], vec![(0.0, 23), (2.0, 5)], [24, 6]),
            (4100, true, false, vec![], vec![(0.0, 4099)], [4100, 0]),
            (4100, false, false, vec![4099], vec![(2.0, 5)], [0, 6]),
            (6, true, false, vec![], vec![(2.0, 5)], [0, 6]),
        ] {
            assert_eq!(cache.get().is_some(), original.is_some());
            let (tlas, reused) = if full {
                cache.prepare(&device, required)
            } else {
                assert!(
                    cache.retained(required + 8192).is_none(),
                    "growth fails before mutation"
                );
                (cache.retained(required).unwrap(), true)
            };
            if let Some(previous) = &original {
                if reused {
                    assert_eq!(tlas, previous);
                } else {
                    assert_ne!(tlas, previous);
                }
            } else {
                assert!(!reused);
            }
            original = Some(tlas.clone());
            for slot in removals {
                scene_tlas::publish_slot(tlas, slot, None);
            }
            let publish: Vec<_> = if dense {
                let old: Vec<_> = previous_dense.iter().map(|&(_, slot)| slot).collect();
                let changed = scene_tlas::changed_dense_slots(
                    &old,
                    positions.iter().map(|&(_, slot)| slot),
                    |slot| {
                        full || previous_dense.iter().find(|p| p.1 == slot)
                            != positions.iter().find(|p| p.1 == slot)
                    },
                );
                for index in required..previous_dense.len() {
                    scene_tlas::publish_slot(tlas, index as u32, None);
                }
                let publish = changed
                    .into_iter()
                    .map(|(address, slot)| {
                        let x = positions.iter().find(|p| p.1 == slot).unwrap().0;
                        (address, x, slot)
                    })
                    .collect();
                previous_dense = positions.clone();
                publish
            } else {
                positions.iter().map(|&(x, slot)| (slot, x, slot)).collect()
            };
            for (address, x, stable_slot) in publish {
                let transform = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
                scene_tlas::publish_slot(
                    tlas,
                    address,
                    Some(wgpu::TlasInstance::new(&blas, transform, stable_slot, 255)),
                );
            }
            if dense {
                assert!(tlas.get()[required..].iter().all(Option::is_none));
                assert_eq!(tlas.get().len(), 4096, "custom IDs do not grow descriptors");
            }
            if full {
                retained_group = Some(raw.create_bind_group(&wgpu::BindGroupDescriptor {
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
                }));
            }
            let group = retained_group.as_ref().unwrap();
            let mut encoder = raw.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(&[], [&*tlas]);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(2, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 8);
            queue.submit([encoder.finish()]);
            readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
            raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            {
                let bytes = readback.get_mapped_range(..);
                assert_eq!(
                    bytemuck::cast_slice::<u8, u32>(&bytes),
                    expected,
                    "required={required}"
                );
            }
            readback.unmap();
            if required == 4100 {
                original = Some(tlas.clone());
            }
        }
    });
}
