//! Production finite ring integration and HDR compositor validation.
const SAMPLES: usize = 64;
fn run_probe() -> Vec<[f32; 4]> {
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
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            })
            .await
            .expect("device");
        let production = include_str!("../src/ring_math.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("enable "))
            .collect::<Vec<_>>()
            .join("\n");
        let transport = include_str!("../src/ring_transport.wgsl")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let production = format!("{production}\n{transport}");
        let source = format!("{production}\n{}", include_str!("rings_fixture.wgsl"));
        let composite = include_str!("../src/rings.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let full = format!(
            "struct View {{view_from_clip:mat4x4<f32>,world_from_view:mat4x4<f32>,exposure:f32}}\n{production}\n{composite}"
        );
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production volume compositor syntax and bindings"),
            source: wgpu::ShaderSource::Wgsl(full.into()),
        });
        let _compositor = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &composite_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production finite Gaussian ray integration"),
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
            size: (SAMPLES * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut ring = vec![0.0f32; 28 + 8];
        ring[3] = 5.0;
        ring[5] = 1.0;
        ring[7] = 1e8;
        ring[8] = 1.0;
        ring[11] = 2e8;
        ring[13] = 1.0;
        ring[15] = 0.0005;
        ring[16] = 1000.0;
        ring[17] = 1000.0;
        ring[18] = 1000.0;
        ring[19] = 1.0;
        ring[20] = 5e7;
        ring[21] = 4.5e7;
        ring[22] = 5e7;
        ring[23] = 1e8;
        ring[24] = 1e8;
        ring[26] = 1.0;
        for i in 0..2 {
            ring[28 + i * 4] = 0.5;
            ring[29 + i * 4] = 0.5;
            ring[30 + i * 4] = -0.5;
        }
        use wgpu::util::DeviceExt;
        let ring_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&ring),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: ring_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((SAMPLES / 64) as u32, 1, 1);
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
        bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
    })
}

#[test]
#[ignore = "requires Vulkan; serialize with builds and captures"]
fn physical_ring_gpu_readback() {
    let values = run_probe();
    for (i, v) in values.iter().enumerate() {
        let x = 0.05 + (i % 8) as f64 * 0.7;
        let a = (i / 8) as f64 * 0.4;
        let b = (7 - i / 8) as f64 * 0.3;
        let count = 32768;
        let expected = (0..count)
            .map(|k| {
                let t = (k as f64 + 0.5) / count as f64;
                (-x * t - a - (b - a) * t).exp() * x / count as f64
            })
            .sum::<f64>();
        assert!(
            (v[0] as f64 - expected).abs() < 2e-6,
            "source {i}: {v:?} vs {expected}"
        );
        let expected_column = if i % 4 == 0 {
            0.5
        } else if i % 4 == 1 {
            0.5 / 0.6
        } else if i % 4 == 2 {
            2.5e6
        } else {
            0.0
        };
        assert!(
            (v[1] as f64 - expected_column).abs() < 2e-5 * expected_column.max(1.0),
            "column {i}: {v:?} vs {expected_column}"
        );
        assert!((0.0..=1.0).contains(&v[2]));
        let fraction = [
            0.0,
            1.0,
            0.5,
            1.0 - (0.5_f64.acos() - 0.5 * 0.75_f64.sqrt()) / core::f64::consts::PI,
        ][i % 4];
        assert!(
            (f64::from(v[2]) - fraction).abs() < 3e-5,
            "ellipsoid limb {i}: {} vs {fraction}",
            v[2]
        );
        assert!((v[3] as f64 - (-expected_column).exp()).abs() < 2e-6);
    }
}
