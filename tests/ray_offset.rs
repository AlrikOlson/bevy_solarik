//! Float32 visibility-origin offsets remain representable from centimetres to planetary distances.
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
            .request_device(&Default::default())
            .await
            .expect("device");
        let source_text = include_str!("../src/scene/raytracing_scene_bindings.wgsl");
        let production = source_text
            .split("fn offset_surface_ray(")
            .nth(1)
            .unwrap()
            .split("// RAY_T_MAX retains")
            .next()
            .unwrap();
        let production = format!("const RAY_T_MIN:f32=0.001;\nfn offset_surface_ray({production}");
        let source = format!("{production}\n{}", include_str!("ray_offset_fixture.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production visibility-origin offset"),
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
fn offset_gpu_remains_representable() {
    for (i, v) in run_probe().iter().enumerate() {
        let magnitude = [0.0_f32, 0.1, 1.0, 100.0, 1e4, 1e6, 1e7, -1e7][i % 8];
        let requested = 0.001_f32.max(magnitude.abs() * 8.0 * f32::EPSILON);
        assert!(v[..3].iter().all(|x| x.is_finite()));
        assert!(
            v[3] >= requested * 0.85 && v[3] <= requested * 1.2,
            "{magnitude}: {v:?} vs {requested}"
        );
        if magnitude.abs() < 100.0 {
            assert!(v[3] < 0.00101);
        }
    }
}
