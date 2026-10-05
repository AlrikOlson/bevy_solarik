//! Numerical GPU readback of the production Lommel-Seeliger optical model.
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
        let production = include_str!("../src/thin_volume_math.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("enable "))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!("{production}\n{}", include_str!("thin_volume_fixture.wgsl"));
        let composite = include_str!("../src/thin_volume.wgsl")
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
fn finite_gaussian_gpu_matches_independent_numeric_quadrature() {
    let values = run_probe();
    for (i, v) in values.iter().enumerate() {
        let q = [(i % 4) as f64 * 1.5, 0.2, -8.0];
        let ray = if i / 16 == 0 {
            [0.0, 0.0, 0.5]
        } else {
            [0.3, 0.4, 0.5]
        };
        let end = [0.0, 8.0, 16.0, 40.0][(i / 4) % 4];
        let n = 32768;
        let step = end / n as f64;
        let expected = (0..n)
            .map(|j| {
                let t = (j as f64 + 0.5) * step;
                let r2 = (0..3).map(|k| (q[k] + t * ray[k]).powi(2)).sum::<f64>();
                (-0.5 * r2).exp() * step
            })
            .sum::<f64>();
        assert!(
            (f64::from(v[0]) - expected).abs() < 2e-6,
            "{i}: GPU {} CPU {expected}",
            v[0]
        );
        assert!(v[0] >= 0.0 && v[0].is_finite());
    }
}
#[test]
fn normalized_kernel_flux_and_invalid_domains() {
    use bevy_math::DVec3;
    use bevy_solarik::thin_volume::GaussianKernel;
    let k = GaussianKernel::axial(
        DVec3::ZERO,
        DVec3::new(1.0, 2.0, 3.0),
        2.0,
        7.0,
        DVec3::splat(11.0),
    )
    .unwrap();
    let volume = (2.0 * core::f64::consts::PI).powf(1.5) * 4.0 * 7.0;
    assert!((f64::from(k.emission.x) * volume - 11.0).abs() < 1e-6);
    assert!((k.x.truncate().length() - 0.5).abs() < 1e-7);
    assert!(k.x.dot(k.z).abs() < 1e-7);
    assert!(GaussianKernel::axial(DVec3::ZERO, DVec3::Y, 0.0, 1.0, DVec3::ONE).is_none());
    assert!(GaussianKernel::axial(DVec3::ZERO, DVec3::ZERO, 1.0, 1.0, DVec3::ONE).is_none());
}
