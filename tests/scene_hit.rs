//! Actual Vulkan conformance for the renderer-owned hit boundary.
#[expect(
    dead_code,
    reason = "This boundary fixture shares the source AS builder without its other outputs"
)]
#[path = "support/source_coverage_acceleration.rs"]
mod acceleration;
#[expect(
    dead_code,
    reason = "The shared input type is constructed analytically, without file export or import"
)]
#[path = "support/source_coverage_input.rs"]
mod input;
use bevy_render::{renderer::initialize_headless_renderer, settings::WgpuSettings};
use wgpu::util::DeviceExt;

fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let open = start + source[start..].find('{').unwrap();
    let mut depth = 1;
    for (offset, c) in source[open + 1..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            return source[start..open + offset + 2].to_string();
        }
    }
    panic!("unterminated function {name}");
}
#[test]
fn tagged_hits_preserve_triangle_queries_and_reject_foreign_payloads_on_vulkan() {
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let device = resources.0.wgpu_device();
        let queue = &resources.1;
        let mut input = input::Input {
            prototype: 1,
            width: 1,
            height: 1,
            alpha: vec![255; 4],
            positions: Vec::new(),
            uvs: vec![[0.; 2]; 12],
            indices: Vec::new(),
            parts: Vec::new(),
            rays: vec![[0.; 8]],
            expected: Vec::new(),
        };
        for z in 0..3 {
            let base = input.positions.len() as u32;
            input
                .parts
                .push([input.indices.len() as u32, 6, (-1f32).to_bits(), 1]);
            input.positions.extend([
                [-1., -1., z as f32, 0.],
                [1., -1., z as f32, 0.],
                [1., 1., z as f32, 0.],
                [-1., 1., z as f32, 0.],
            ]);
            input
                .indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let scene = acceleration::build(device, queue, &input);
        let mut rays: Vec<[f32; 12]> = Vec::new();
        for x in [-0.25, 0.25, 2.] {
            for (z, dz) in [(-2., 1.), (4., -1.)] {
                for low in [0., 0.001, 2.25, 5.] {
                    for high in [0., 1.5, 2.5, 5., 10.] {
                        for flags in [0u32, 4u32] {
                            for glass in [false, true] {
                                rays.push([
                                    x,
                                    0.125,
                                    z,
                                    low,
                                    0.,
                                    0.,
                                    dz,
                                    high,
                                    f32::from_bits(flags),
                                    f32::from_bits(u32::from(glass)),
                                    0.,
                                    0.,
                                ]);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(rays.len(), 480);
        let source = include_str!("../src/scene/raytracing_scene_bindings.wgsl");
        let shared = include_str!("../src/scene/scene_hit.wgsl")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let functions = [
            "trace_ray",
            "trace_glass_ray",
            "scene_hit_from_triangle",
            "trace_ray_impl",
            "trace_triangle_ray_impl",
            "scene_hit_material",
            "scene_hit_alpha",
            "resolve_ray_hit_filtered",
        ]
        .map(|name| function(source, name))
        .join("\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production scene hit boundary"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "enable wgpu_ray_query;\n{shared}\n{functions}\n{}",
                    include_str!("scene_hit.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let request = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&rays),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (rays.len() * 32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: request.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = readback.get_mapped_range(..);
        let rows: &[[u32; 8]] = bytemuck::cast_slice(&bytes);
        let mut hits = 0;
        for (i, (row, ray)) in rows.iter().zip(&rays).enumerate() {
            assert_eq!([row[0], row[6], row[7]], [0; 3], "case{i} {ray:?} {row:?}");
            if row[1] == 1 {
                hits += 1;
                assert!(row[2] < 3 && row[3] < 2 && row[5] <= 1);
                let t = f32::from_bits(row[4]);
                assert!(t >= ray[3] && t <= ray[7].min(6.));
                // Every accepted event lies on an original plane, not its bounding box.
                assert!((ray[2] + ray[6] * t - row[2] as f32).abs() < 1e-6);
            } else {
                assert_eq!(row[1], 0);
            }
        }
        assert!(
            hits > 100 && hits < 400,
            "both hits and misses exercised: {hits}"
        );
    });
}
