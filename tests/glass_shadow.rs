//! Deterministic production shadow traversal through synthetic pane intersections.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn glass_shadow_gpu() {
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
        let start = sampling
            .find("fn trace_shadow_transmission(")
            .expect("shadow transport");
        let end = sampling[start..]
            .find("// End shadow transport")
            .expect("end shadow transport")
            + start;
        let glass = include_str!("../src/scene/thin_glass.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "{glass}\n{}\n{}",
            &sampling[start..end],
            include_str!("glass_shadow_fixture.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production shadow transport"),
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
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // alpha, count, distance, diffuse coverage, opaque blocker, incidence cosine.
        for [alpha, count, distance, diffuse, opaque, cosine] in [
            [1.0f32, 0.0, 10.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 10.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 10.0, 0.0, 0.0, 1.0],
            [0.35, 1.0, 10.0, 0.0, 0.0, 1.0],
            [1.0, 4.0, 10.0, 0.0, 0.0, 1.0],
            [1.0, 32.0, 40.0, 0.0, 0.0, 1.0],
            [1.0, 33.0, 40.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5, 0.0, 0.0, 1.0],
            [1.0, 1.0, 10.0, 0.0, 1.0, 1.0],
            [0.35, 2.0, 10.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 10.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 10.0, 0.0, 0.0, 0.5],
        ] {
            let tint = [0.2f32, 0.5, 0.8];
            let input = [
                [tint[0], tint[1], tint[2], alpha],
                [0.5, count, distance, diffuse],
                [opaque, cosine, 0.0, 0.0],
            ];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
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
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 16);
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
            let result: &[f32] = bytemuck::cast_slice(&data);
            let f = 0.04 + 0.96 * (1.0 - cosine).powi(5);
            let reflection = 2.0 * f / (1.0 + f);
            let blocked = opaque > 0.0 || count > 32.0;
            let panes = if distance < 1.0 { 0 } else { count as i32 };
            for (i, value) in result.iter().enumerate() {
                let single = if diffuse > 0.0 {
                    1.0 - alpha
                } else if i == 3 {
                    1.0 - alpha * reflection
                } else {
                    1.0 - alpha + alpha * (1.0 - reflection) * tint[i]
                };
                let expected = if blocked { 0.0 } else { single.powi(panes) };
                assert!(
                    (*value - expected).abs() < 2e-5,
                    "case {input:?}: {result:?}, channel{i} expected{expected}"
                );
            }
            drop(data);
            readback.unmap();
        }
    });
}
