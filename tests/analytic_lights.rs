//! Production analytic light ray intersections against independent geometric controls.
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
fn analytic_lights_gpu() {
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
        let types_start = sampling
            .find("struct ResolvedLightSample {")
            .expect("types");
        let types_end = sampling
            .find("struct LightContributionNoPdf {")
            .expect("types end");
        let mut source = sampling[types_start..types_end].to_owned();
        for name in ["analytic_light_radiance", "local_light_attenuation"] {
            source.push_str(
                &function(sampling, name)
                    .replace("arrayLength(&local_lights)", "1u")
                    .replace("arrayLength(&directional_lights)", "1u"),
            );
        }
        source.push_str(include_str!("analytic_lights_fixture.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production analytic lights"),
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
            size: 16 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for scale in [0.1f32, 1.0, 10.0] {
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[[scale, 0.0, 0.0, 0.0]]),
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
            // Center hit, miss, blocker, behind, inside/backface, spot on/off,
            // range cutoff, sun on/off/blocked, disabled ownership, additivity.
            let window = (1.0f32 - (4.5f32 / 100.0).powi(4)).powi(2);
            let expected = [
                2.0 * window,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0 * window,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
                0.0,
                2.0 * window + 3.0,
                0.0,
                2.0 * (1.0f32 - 0.9f32.powi(4)).powi(2),
                0.0,
            ];
            for (i, (sample, expected)) in samples.iter().zip(expected).enumerate() {
                for channel in &sample[..3] {
                    assert!(
                        (channel - expected).abs() < 2e-5,
                        "case {i} scale {scale}: {sample:?}, expected {expected}"
                    );
                }
            }
            drop(data);
            readback.unmap();
        }
    });
}
