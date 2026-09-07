//! Execute the shared angular-radiance law on GPU; integrate emitted power.
use bevy_math::Vec3;
use bevy_solarik::scene::collimated::CollimatedEmission;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn collimated_gpu() {
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
        let production = include_str!("../src/scene/collimated.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "{production}\n{}",
            r#"
@group(0) @binding(0) var<storage> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
    output[id.x] = collimated_weight(inputs[0], inputs[id.x + 1u].xyz);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production collimated radiance"),
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
        let cone = CollimatedEmission::new(Vec3::Z, 4.0_f32.to_radians() / 120.0).unwrap();
        let radius = cone.gpu().w;
        let mut input = vec![cone.gpu().to_array()];
        // Uniform area in tangent-direction disk; include beyond the cone.
        for i in 0..4096 {
            let r = radius * 2.0 * ((i as f32 + 0.5) / 4096.0).sqrt();
            input.push([r, 0.0, 1.0, 0.0]);
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4096 * 4,
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
            pass.dispatch_workgroups(64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("map callback");
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU complete");
        rx.recv().expect("callback").expect("map");
        let data = readback.slice(..).get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        for (value, direction) in values.iter().zip(&input[1..]) {
            let expected = cone.weight(Vec3::new(direction[0], direction[1], direction[2]));
            assert!((value - expected).abs() < 1e-5, "{value} vs {expected}");
        }
        let integral_ratio = values.iter().map(|&v| f64::from(v)).sum::<f64>() * 4.0 / 4096.0;
        // Analytic integral of the radial smoothstep on [0.9, 1] is 0.903.
        assert!((integral_ratio - 0.903).abs() < 0.001, "{integral_ratio}");
        assert!(
            values[1024..].iter().all(|&x| x == 0.0),
            "no out-of-cone energy"
        );
        drop(data);
        readback.unmap();
    });
}
