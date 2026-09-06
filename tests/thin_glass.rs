//! Numerical contracts executed against the production thin-glass WGSL.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a Vulkan GPU; run separately from builds and scene captures"]
fn thin_glass_gpu() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Vulkan adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let production = include_str!("../src/scene/thin_glass.wgsl");
        let production = production
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "{production}\n{}",
            r#"
@group(0) @binding(0) var<storage> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let u = (f32(id.x) + 0.5) / 65536.0;
    let wo = inputs[1].xyz;
    let n = inputs[2].xyz;
    let s = sample_thin_glass(wo, n, inputs[0].xyz, inputs[0].w, inputs[1].w, u);
    output[id.x * 3u] = vec4(s.wi, f32(s.reflected));
    output[id.x * 3u + 1u] = vec4(s.throughput, 0.0);
    let origin = offset_thin_glass_ray(vec3(0.0), n, s.wi, 0.001);
    output[id.x * 3u + 2u] = vec4(origin, 0.0);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production thin glass"),
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
            size: 65536 * 3 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // (color, coverage, reflectance control, cosine, expected covered reflection).
        // Default material F0=0.04 gives two-interface reflection 1/13 at normal incidence.
        let cases = [
            ([1.0, 1.0, 1.0], 1.0, 0.5, 1.0, 1.0 / 13.0),
            ([0.2, 0.5, 0.8], 0.0, 0.5, 1.0, 1.0 / 13.0),
            ([0.2, 0.5, 0.8], 0.35, 0.5, 1.0, 1.0 / 13.0),
            ([0.2, 0.5, 0.8], 1.0, 0.5, 1.0, 1.0 / 13.0),
            ([0.0, 0.0, 0.0], 1.0, 0.5, 1.0, 1.0 / 13.0),
            ([1.0, 1.0, 1.0], 1.0, 0.5, 0.0, 1.0),
            ([0.2, 0.5, 0.8], 0.35, 0.5, 0.0, 1.0),
            ([1.0, 1.0, 1.0], 1.0, 0.0, 1.0, 0.0),
            ([1.0, 1.0, 1.0], 1.0, 4.0, 1.0, 1.0),
            ([1.0, 1.0, 1.0], 2.0, 0.5, 1.0, 1.0 / 13.0),
            ([1.0, 1.0, 1.0], -1.0, 0.5, 1.0, 1.0 / 13.0),
            // cos=0.5 gives Schlick F=0.07 and pane R=14/107.
            ([0.2, 0.5, 0.8], 1.0, 0.5, 0.5, 14.0 / 107.0),
        ];
        for (color, alpha, reflectance, cosine, pane_r) in cases {
            for side in [-1.0f32, 1.0] {
                let wo = [(1.0f32 - cosine * cosine).sqrt(), 0.0, side * cosine];
                let input = [
                    [color[0], color[1], color[2], alpha],
                    [wo[0], wo[1], wo[2], reflectance],
                    [0.0, 0.0, 1.0, 0.0],
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
                    pass.dispatch_workgroups(1024, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
                queue.submit([encoder.finish()]);
                let (sender, receiver) = std::sync::mpsc::channel();
                readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    sender.send(r).expect("map result");
                });
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU completed");
                receiver.recv().expect("map callback").expect("map");
                let data = readback.slice(..).get_mapped_range();
                let samples: &[[f32; 4]] = bytemuck::cast_slice(&data);
                let mut reflections = 0;
                let mut energy = [0.0f64; 3];
                for sample in samples.chunks_exact(3) {
                    assert!(
                        sample.iter().flatten().all(|x| x.is_finite()),
                        "finite at endpoints"
                    );
                    let reflected = sample[0][3] == 1.0;
                    reflections += usize::from(reflected);
                    let expected = [-wo[0], -wo[1], if reflected { wo[2] } else { -wo[2] }];
                    for (actual, expected) in sample[0][..3].iter().zip(expected) {
                        assert!(
                            (actual - expected).abs() < 0.00001,
                            "direction on side {side}"
                        );
                    }
                    assert!(
                        sample[2][2] * sample[0][2] >= 0.0,
                        "origin on outgoing side"
                    );
                    assert!((sample[2][2].abs() - 0.001).abs() < 0.000001);
                    for (sum, value) in energy.iter_mut().zip(sample[1]) {
                        *sum += f64::from(value);
                    }
                }
                let coverage = alpha.clamp(0.0, 1.0);
                let expected_r = coverage * pane_r;
                assert!(
                    (reflections as f32 / 65536.0 - expected_r).abs() < 0.00002,
                    "Fresnel probability"
                );
                for (sum, tint) in energy.into_iter().zip(color) {
                    let expected = expected_r + 1.0 - coverage + coverage * (1.0 - pane_r) * tint;
                    assert!(
                        (sum / 65536.0 - f64::from(expected)).abs() < 0.00003,
                        "energy: {sum} vs {expected}"
                    );
                }
                drop(data);
                readback.unmap();
            }
        }
    });
}
