//! Production emitter MIS PDFs integrated over a square area light.
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
fn emitter_mis_gpu() {
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
        for name in [
            "power_heuristic",
            "balance_heuristic",
            "area_to_solid_angle_pdf",
            "random_emissive_light_pdf",
            "random_emissive_light_solid_angle_pdf",
            "calculate_resolved_light_contribution",
        ] {
            source.push_str(&function(sampling, name));
        }
        source.push_str(
            &include_str!("../src/scene/collimated.wgsl")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        source.push_str(include_str!("emitter_mis_fixture.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production emitter MIS"),
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
            size: 65536 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Half-width, distance, selection PMF; scaled scenes have equal solid angle.
        for [half_width, distance, pmf, cone_radius] in [
            [1.0f32, 1.0, 1.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [4.0, 4.0, 1.0, 0.0],
            [0.1, 0.1, 0.25, 0.0],
            [10.0, 10.0, 0.25, 0.0],
            // Resolve a narrow directional emitter densely, with unchanged MIS PDFs.
            [0.0007, 1.0, 0.25, 0.0005817765],
        ] {
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[[half_width, distance, pmf, cone_radius]]),
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
                pass.dispatch_workgroups(1024, 1, 1);
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
            let mut radiance = 0.0f64;
            for (i, sample) in samples.iter().enumerate() {
                let x = ((i % 256) as f64 + 0.5) / 256.0 * 2.0 - 1.0;
                let y = ((i / 256) as f64 + 0.5) / 256.0 * 2.0 - 1.0;
                let a = f64::from(half_width);
                let d = f64::from(distance);
                let r2 = a * a * (x * x + y * y) + d * d;
                let expected_pdf = f64::from(pmf) * r2.sqrt().powi(3) / (4.0 * a * a * d);
                for pdf in &sample[1..3] {
                    assert!(
                        (f64::from(*pdf) / expected_pdf - 1.0).abs() < 2e-5,
                        "PDF: {sample:?} expected {expected_pdf}"
                    );
                }
                assert!(
                    (sample[3] - 1.0).abs() < 2e-5,
                    "complementary MIS: {sample:?}"
                );
                radiance += f64::from(sample[0]);
            }
            radiance /= 65536.0;
            let ratio = f64::from(half_width) / f64::from(half_width.hypot(distance));
            let expected = if cone_radius > 0.0 {
                // Soft disk angular integral / PI; tiny-angle error is < 1 ppm.
                0.903 * f64::from(cone_radius).powi(2)
            } else {
                4.0 / core::f64::consts::PI * ratio * ratio.atan()
            };
            assert!(
                (radiance - expected).abs() < 2e-5 * expected.max(0.000001),
                "a={half_width} d={distance} pmf={pmf}: {radiance} expected {expected}"
            );
            drop(data);
            readback.unmap();
        }
    });
}
