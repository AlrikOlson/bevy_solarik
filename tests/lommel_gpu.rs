//! Numerical GPU readback of the production Lommel-Seeliger optical model.
const SAMPLES: usize = 4096 * 3;
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
            .request_device(&Default::default())
            .await
            .expect("device");
        let production = include_str!("../src/lommel_math.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("enable "))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!("{production}\n{}", include_str!("lommel_fixture.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production BRDF, view-facing normals"),
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
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
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
fn lommel_gpu_readback() {
    let values = run_probe();
    for (i, v) in values.iter().enumerate() {
        let nl = (i % 4096) as f64 / 4096.0 + 0.0001;
        let nv = [0.001, 0.3, 1.0][i / 4096];
        let expected = bevy_solarik::lommel::lommel_seeliger(0.2, nl, nv) * nl;
        assert!(
            (f64::from(v[0]) - expected).abs() < 2e-7,
            "{i}: {v:?} vs {expected}"
        );
        assert!((v[1] - v[2]).abs() < 2e-5, "reciprocity {i}: {v:?}");
        assert!((v[3] - 0.2 / core::f32::consts::PI).abs() < 1e-7);
    }
}
