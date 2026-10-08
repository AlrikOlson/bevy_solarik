//! Real ray queries plus production cache support rejection after scene edits.
#[expect(
    dead_code,
    unused_imports,
    reason = "shared production history module; this ray fixture covers its bounded subset"
)]
#[path = "../src/scene/history.rs"]
mod history;
#[path = "../src/scene/tlas.rs"]
mod scene_tlas;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use wgpu::util::DeviceExt;

fn function(source: &str, name: &str) -> String {
    let start = source
        .find(&format!("fn {name}("))
        .expect("production function");
    let body = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    for (offset, ch) in source[body..].char_indices() {
        if ch == '{' {
            depth += 1;
        }
        if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return source[start..=body + offset].to_owned();
            }
        }
    }
    panic!("unterminated production function");
}

#[test]
#[ignore = "requires Vulkan ray queries; run separately from builds and captures"]
fn cache_support_rejects_moving_distant_blockers_and_preserves_bounded_unaffected_cells() {
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
        let sampling = include_str!("../src/scene/sampling.wgsl");
        let support_struct = &sampling[sampling.find("struct ShadowTransmission").unwrap()
            ..sampling.find("fn trace_light_transmission(").unwrap()];
        let source = format!(
            "{}\n{support_struct}\n{}\n{}",
            include_str!("cache_locality_fixture.wgsl"),
            function(sampling, "trace_shadow_transmission_with_support"),
            function(
                include_str!("../src/realtime/world_cache_compact.wgsl"),
                "cache_support_is_valid"
            )
        );
        let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production bounded cache validity with actual TLAS"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |usage| {
            raw.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage,
                mapped_at_creation: false,
            })
        };
        let cached = buffer(wgpu::BufferUsages::STORAGE);
        let output = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut cache = scene_tlas::SceneTlas::default();
        let view = history::ViewHistory::default();
        let source_mesh = bevy_mesh::Mesh::new(
            wgpu::PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_attribute(
            bevy_mesh::Mesh::ATTRIBUTE_POSITION,
            vertices
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>(),
        );
        let bounds = history::Bounds::from_mesh(&source_mesh).expect("source bounds");
        assert!(
            bounds.stable_radius(bevy_math::Affine3A::from_translation(
                bevy_math::Vec3::X * 1000.0
            )) > 999.0
        );
        // Actual old/new bounds lie outside each declared conservative stable radius.
        // Includes unrelated far admission, >50m blocker, removal, motion, empty,
        // readiness/slot return and global source/light/material epoch fallback.
        type Placements = Vec<(f32, f32, u32)>;
        type Case = (Placements, f32, bool, [f32; 2], [f32; 2]);
        let cases: Vec<Case> = vec![
            (vec![(0.0, 2.0, 23)], 0.0, true, [0.0, 0.0], [0.0, 1.0]),
            (
                vec![(0.0, 2.0, 23), (1000.0, 2.0, 5)],
                999.0,
                false,
                [1.0, 0.0],
                [0.0, 1.0],
            ),
            (
                vec![(0.0, 2.0, 23), (1000.0, 2.0, 5), (2.0, 2000.0, 47)],
                1999.0,
                false,
                [1.0, 0.0],
                [0.0, 0.0],
            ),
            (
                vec![(0.0, 2.0, 23), (1000.0, 2.0, 5)],
                1999.0,
                false,
                [1.0, 0.0],
                [0.0, 1.0],
            ),
            (
                vec![(0.0, 3.0, 23), (1000.0, 2.0, 5)],
                1.99,
                false,
                [0.0, 0.0],
                [0.0, 1.0],
            ),
            (vec![], 2.99, false, [0.0, 0.0], [1.0, 1.0]),
            (vec![(0.0, 2.0, 99)], 1.99, false, [0.0, 0.0], [0.0, 1.0]),
            (
                vec![(0.0, 2.0, 99)],
                f32::INFINITY,
                true,
                [0.0, 0.0],
                [0.0, 1.0],
            ),
            // Empty publication while the node cannot run, then a different
            // far scene: neither may silently retain the departed near blocker.
            (vec![], 1.99, false, [0.0, 0.0], [1.0, 1.0]),
            (
                vec![(1000.0, 2.0, 101)],
                999.0,
                false,
                [0.0, 0.0],
                [1.0, 1.0],
            ),
        ];
        for (case, (positions, radius, full, validity, energy)) in cases.into_iter().enumerate() {
            let generation = case as u64 + 1;
            if full {
                view.invalidate();
            }
            if case == 8 {
                cache.prepare(&device, 0);
                continue; // mirrors the production early return; no history consumption
            }
            let radius = view.stable_radius(generation, radius);
            let full = radius == 0.0;
            let config = raw.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[radius, f32::from(full), 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let (tlas, _) = cache.prepare(&device, positions.len());
            for (slot, &(x, z, id)) in positions.iter().enumerate() {
                let transform = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z];
                *tlas.get_mut_single(slot).unwrap() =
                    Some(wgpu::TlasInstance::new(&blas, transform, id, 255));
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
                        resource: cached.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: config.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = raw.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(&[], [&*tlas]);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(2, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
            queue.submit([encoder.finish()]);
            view.rendered(generation);
            readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
            raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            {
                let bytes = readback.get_mapped_range(..);
                let result: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
                for i in 0..2 {
                    assert_eq!(result[i][0], validity[i], "case {case}, cell {i}: reuse");
                    assert_eq!(
                        result[i][1], energy[i],
                        "case {case}, cell {i}: retained energy"
                    );
                    assert_eq!(
                        result[i][1], result[i][2],
                        "case {case}, cell {i}: stale visibility"
                    );
                }
                if case == 2 {
                    assert!(result[1][3] > 1999.0, "far blockers still affect sky");
                }
            }
            readback.unmap();
        }
    });
}
