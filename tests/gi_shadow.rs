//! Production GI reuse retains endpoint radiance and shades the current connection.
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
fn gi_shadow_gpu() {
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
        let gi = include_str!("../src/realtime/restir_gi.wgsl");
        let sampling = include_str!("../src/scene/sampling.wgsl");
        let mut source = include_str!("gi_shadow_fixture.wgsl").to_owned();
        for name in ["empty_reservoir", "merge_reservoirs", "jacobian", "isinf"] {
            source.push_str(&function(gi, name));
        }
        for name in ["balance_heuristic", "isnan", "shade_gi_connection"] {
            source.push_str(&function(sampling, name));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production GI connection reuse"),
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
            size: 3 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Tint includes black and single-channel energy; repeat temporal/spatial reuse.
        for tint in [
            [1.0f32, 1.0, 1.0, 0.0],
            [0.2, 0.5, 0.8, 1.0],
            [0.0, 0.0, 0.0, 8.0],
            [1.0, 0.0, 0.0, 64.0],
            [0.25, 0.5, 0.75, 256.0],
        ] {
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[tint]),
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
                pass.dispatch_workgroups(1, 1, 1);
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
            for (channel, expected) in tint[..3].iter().enumerate() {
                for sample in &samples[..1] {
                    assert!(
                        (sample[channel] - expected * 2.0).abs() < 2e-4,
                        "tint {tint:?}: {samples:?}"
                    );
                }
            }
            assert!(
                (samples[2][0] - 2.0).abs() < 2e-4,
                "UCW changed: {samples:?}"
            );
            for value in &samples[1][..3] {
                assert!((*value - 1.0).abs() < 2e-4, "endpoint changed: {samples:?}");
            }
            drop(data);
            readback.unmap();
        }
    });
}
