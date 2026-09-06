//! Production bounded surface paths: complementary BSDF/NEE through thin panes.
use wgpu::util::DeviceExt;

fn function(source: &str, name: &str) -> String {
    let start = source
        .find(&format!("fn {name}("))
        .expect("production function");
    let body = start + source[start..].find('{').expect("body");
    let mut depth = 0;
    for (offset, ch) in source[body..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..=body + offset].to_owned();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function");
}

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn surface_shadow_gpu() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let sampling = include_str!("../src/scene/sampling.wgsl");
        let surface = include_str!("../src/realtime/surface_path.wgsl");
        let glass = include_str!("../src/scene/thin_glass.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let mut source = format!("{glass}\n{}", include_str!("surface_shadow_fixture.wgsl"));
        source.push_str(
            &function(surface, "shade_surface_path")
                .replace("arrayLength(&light_sources)", "arrayLength(&config)"),
        );
        for name in ["power_heuristic", "balance_heuristic"] {
            source.push_str(&function(sampling, name));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production emitter MIS"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 65536 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Coverage, pane count, fixture mode (glass, diffuse coverage, reflection).
        for [alpha, count, mode] in [
            [1.0f32, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.35, 1.0, 0.0],
            [1.0, 4.0, 0.0],
            [1.0, 32.0, 0.0],
            [1.0, 33.0, 0.0],
            [0.35, 2.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ] {
            let color = if count >= 32.0 {
                [1.0f32; 3]
            } else {
                [0.2, 0.5, 0.8]
            };
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[
                    [color[0], color[1], color[2], alpha],
                    [count, mode, 0.0, 0.0],
                ]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1024, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                sender.send(r).expect("callback");
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU");
            receiver.recv().expect("callback").expect("map");
            let data = readback.slice(..).get_mapped_range();
            let samples: &[[f32; 4]] = bytemuck::cast_slice(&data);
            let tint = color.map(f64::from);
            for (channel, tint) in tint.iter().enumerate() {
                let measured = samples.iter().map(|p| f64::from(p[channel])).sum::<f64>() / 65536.0;
                let single = if mode == 1.0 {
                    1.0 - f64::from(alpha)
                } else {
                    1.0 - f64::from(alpha) + f64::from(alpha) * (12.0 / 13.0) * tint
                };
                let expected = if count > 32.0 {
                    0.0
                } else if mode == 2.0 {
                    1.0 / 13.0
                } else {
                    single.powi(count as i32)
                };
                assert!(
                    (measured - expected).abs() < 0.006,
                    "alpha={alpha} count={count} mode={mode} channel={channel}: {measured} expected {expected}"
                );
            }
            drop(data);
            readback.unmap();
        }
    });
}
