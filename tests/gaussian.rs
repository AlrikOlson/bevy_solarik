//! Numerical GPU readback of the production Gaussian optical model.
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
        let production = include_str!("../src/gaussian_math.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("enable "))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!("{production}\n{}", include_str!("gaussian_fixture.wgsl"));
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
fn parameter_domain_and_fresnel_references() {
    use bevy_solarik::gaussian::{GaussianDielectric, dielectric_fresnel};
    for bad in [f32::NAN, -1.0, 0.0, 0.01, 1.01] {
        assert!(GaussianDielectric::new(1.333, bad, Default::default()).is_none());
    }
    assert!(GaussianDielectric::new(1.333, 0.2, Default::default()).is_some());
    // Packed deferred reflectance is UNORM8: eta above 7/3 cannot be encoded.
    for bad_ior in [0.99, 2.34, 3.0, f32::INFINITY, f32::NAN] {
        assert!(GaussianDielectric::new(bad_ior, 0.2, Default::default()).is_none());
    }
    assert!((dielectric_fresnel(1.0, 1.333) - 0.020373187841971414).abs() < 1e-12);
    assert_eq!(dielectric_fresnel(0.0, 1.333), 1.0);
    assert_eq!(dielectric_fresnel(0.0, 1.0), 0.0);
    // At Brewster incidence the p polarization vanishes exactly.
    let n = 1.333_f64;
    let brewster = 1.0 / (1.0 + n * n).sqrt();
    let expected = 0.5 * ((1.0 - n * n) / (1.0 + n * n)).powi(2);
    assert!((dielectric_fresnel(brewster, n) - expected).abs() < 1e-14);
}
#[test]
#[ignore = "requires Vulkan; serialize with every build and capture"]
fn gaussian_gpu_readback() {
    let values = run_probe();
    for (wind, samples) in [1.0_f64, 5.0, 12.0]
        .into_iter()
        .zip(values.chunks_exact(4096))
    {
        let a2 = 0.003 + 0.00508 * wind;
        let (mut integral, mut moment) = (0.0, 0.0);
        for (i, v) in samples.iter().enumerate() {
            let c = (i as f64 + 0.5) / 4096.0;
            let expected = bevy_solarik::gaussian::dielectric_fresnel(c, 1.333);
            assert!(
                (f64::from(v[0]) - expected).abs() < 2e-6,
                "Fresnel {i}: {v:?}"
            );
            let density =
                (-(1.0 - c * c) / (a2 * c * c)).exp() / (core::f64::consts::PI * a2 * c.powi(4));
            assert!((f64::from(v[1]) - density).abs() < 3e-4, "NDF {i}: {v:?}");
            assert!((v[2] - v[3]).abs() < 2e-5, "reciprocity {i}: {v:?}");
            let area = f64::from(v[1]) * c * core::f64::consts::TAU / 4096.0;
            integral += area;
            moment += area * (1.0 - c * c) / (c * c);
        }
        assert!(
            (integral - 1.0).abs() < 0.001,
            "normalization {wind}: {integral}"
        );
        assert!((moment - a2).abs() < 2e-5, "MSS {wind}: {moment} vs {a2}");
    }
}
