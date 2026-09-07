//! Numerical contracts executed against the production foliage diffuse BSDF WGSL.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a GPU (SOLARIK_TEST_BACKEND=metal for Metal); run separately from builds and captures"]
fn foliage_gpu() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: if std::env::var("SOLARIK_TEST_BACKEND").as_deref() == Ok("metal") {
                wgpu::Backends::METAL
            } else {
                wgpu::Backends::VULKAN
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");

        let production = include_str!("../src/scene/brdf.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("enable "))
            .collect::<Vec<_>>()
            .join("\n");
        // DFG is specular-only; this probe exercises diffuse transport.
        let production = production.replace(
            "return textureSampleLevel(brdf_dfg_lut, brdf_dfg_lut_sampler, vec2<f32>(NdotV, perceptual_roughness), 0.0).rg;",
            "return vec2(0.0);"
        );
        let math = include_str!("../src/realtime/foliage_math.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "{production}\n{math}\n{}",
            include_str!("foliage_fixture.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production foliage BRDF"),
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

        for transmission in [0.0f32, 0.25, 0.5, 1.0] {
            let input = [[transmission, 0.0, 0.0, 0.0]];
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

            let mut back = 0;
            let mut energy = [0.0f64; 3];
            for sample in samples.chunks_exact(3) {
                assert!(sample.iter().flatten().all(|x| x.is_finite()));
                assert_eq!(sample[0][3], 1.0, "depth ownership and PDF conversion");
                back += usize::from(sample[0][2] < 0.0);
                assert!(
                    (sample[1][3] - sample[2][3]).abs() < 1e-6,
                    "sample and MIS PDF"
                );
                for (sum, value) in energy.iter_mut().zip(sample[1]) {
                    *sum += f64::from(value);
                }
                let expected = [0.2, 0.5, 0.8];
                for (actual, color) in sample[2][..3].iter().zip(expected) {
                    assert!(
                        (actual - color / core::f32::consts::PI).abs() < 1e-6,
                        "front + back energy at normal incidence"
                    );
                }
            }
            assert!(
                (back as f32 / 65536.0 - transmission).abs() < 0.01,
                "hemisphere probability"
            );
            // Schlick entry F=0; cosine-averaged exit transmission is 20/21.
            for (sum, color) in energy.into_iter().zip([0.2, 0.5, 0.8]) {
                assert!((sum / 65536.0 - color * 20.0 / 21.0).abs() < 0.005);
            }
            drop(data);
            readback.unmap();
        }
    });
}
