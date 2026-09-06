//! GPU sample-count diagnostic preserves integer counts through sRGB encoding.
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
fn sample_count_gpu() {
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
        let source = function(
            include_str!("../src/pathtracer/pathtracer.wgsl"),
            "encode_sample_count",
        ) + r#"
@group(0) @binding(0) var<storage> config: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output: array<vec4<f32>>;
@compute @workgroup_size(16)
fn probe(@builtin(local_invocation_index) i:u32) { output[i]=vec4(encode_sample_count(config[0].x),1.0); }
"#;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production sample count"),
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
        for count in [
            0.0f32, 1.0, 128.0, 2048.0, 8192.0, 65535.0, 65536.0, 16777215.0, 16777216.0,
        ] {
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[[count, 0.0, 0.0, 0.0]]),
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
            for sample in samples {
                let mut decoded = 0u32;
                for (channel, value) in sample[..3].iter().enumerate() {
                    let srgb = if *value <= 0.0031308 {
                        *value * 12.92
                    } else {
                        1.055 * value.powf(1.0 / 2.4) - 0.055
                    };
                    decoded |= ((srgb * 255.0).round() as u32) << (channel * 8);
                }
                assert_eq!(decoded, (count as u32).min(0xffffff), "count {count}");
            }
            drop(data);
            readback.unmap();
        }
    });
}
