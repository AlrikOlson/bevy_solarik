//! Production glossy-to-foliage transport with deterministic scene I/O.
use wgpu::util::DeviceExt;

fn shader_body(source: &str, rr: bool) -> String {
    let mut excluded = false;
    source
        .lines()
        .filter(|line| {
            if line.starts_with("#ifdef") {
                excluded = !rr;
                return false;
            }
            if line.starts_with("#endif") {
                excluded = false;
                return false;
            }
            !excluded && !line.starts_with('#') && !line.starts_with("enable ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn glossy_foliage_gpu() {
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
        let glossy = include_str!("../src/realtime/specular_gi.wgsl");
        let start = glossy
            .find("fn trace_glossy_path(")
            .expect("production path");
        let end = glossy
            .find("// https://en.wikipedia.org/wiki/Householder_transformation")
            .expect("PSR helpers");
        let samples = 32768usize;
        for rr in [false, true] {
            let glossy = shader_body(&glossy[start..end], rr);
            let surface = shader_body(include_str!("../src/realtime/surface_path.wgsl"), rr)
                .replace("arrayLength(&light_sources)", "arrayLength(&config)");
            let brdf = shader_body(include_str!("../src/scene/brdf.wgsl"), rr).replace(
                "return textureSampleLevel(brdf_dfg_lut, brdf_dfg_lut_sampler, vec2<f32>(NdotV, perceptual_roughness), 0.0).rg;",
                "return vec2(0.0);");
            let glass = shader_body(include_str!("../src/scene/thin_glass.wgsl"), rr);
            let source = format!(
                "{glossy}\n{surface}\n{brdf}\n{glass}\n{}",
                include_str!("glossy_foliage_fixture.wgsl")
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("production reflected foliage"),
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
                size: (samples * 32) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: output.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // transmission, primary roughness, leaf emission, scene mode; expected diffuse energy multiplier.
            let cases: [(&str, [f32; 4], f32); 8] = [
                ("opaque control", [0.0, 0.001, 0.0, 0.0], 0.0),
                (
                    "half transmission",
                    [0.5, 0.001, 0.0, 0.0],
                    0.5 * 20.0 / 21.0,
                ),
                ("full transmission", [1.0, 0.001, 0.0, 0.0], 20.0 / 21.0),
                ("backlit NEE", [0.5, 0.001, 0.0, 1.0], 0.5),
                ("emission owned once", [1.0, 0.001, 3.0, 2.0], 0.0),
                ("emission owned by DI", [1.0, 0.1, 3.0, 2.0], 0.0),
                (
                    "rough path bypasses cache",
                    [1.0, 0.6, 0.0, 3.0],
                    20.0 / 21.0,
                ),
                (
                    "glass before foliage",
                    [1.0, 0.001, 0.0, 4.0],
                    0.5 * 20.0 / 21.0,
                ),
            ];
            for (name, config, energy) in cases {
                let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&config),
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
                    pass.dispatch_workgroups((samples / 64) as u32, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
                queue.submit([encoder.finish()]);
                let (sender, receiver) = std::sync::mpsc::channel();
                readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    sender.send(r).expect("map result");
                });
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU complete");
                receiver.recv().expect("map callback").expect("map");
                let data = readback.slice(..).get_mapped_range();
                let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
                let mut mean = [0.0f64; 3];
                for sample in values.chunks_exact(2) {
                    assert!(
                        sample.iter().flatten().all(|v| v.is_finite()),
                        "{name}: finite"
                    );
                    assert!(sample[0][3] <= 3.0, "{name}: bounded traversal");
                    assert_eq!(sample[1][0], 0.0, "{name}: outgoing ray offset");
                    assert_eq!(
                        sample[1][1], 0.0,
                        "{name}: foliage never enters one-sided cache"
                    );
                    let psr = if rr && config[1] <= 0.002 && config[3] != 4.0 {
                        1.0
                    } else {
                        0.0
                    };
                    assert_eq!(sample[1][2], psr, "{name}: RR surface ownership");
                    for (sum, value) in mean.iter_mut().zip(sample[0]) {
                        *sum += f64::from(value);
                    }
                }
                let emission = if config[1] <= 0.0225 { config[2] } else { 0.0 };
                for (sum, color) in mean.into_iter().zip([0.2, 0.5, 0.8]) {
                    let actual = sum / samples as f64;
                    let expected = f64::from(color * energy + emission);
                    assert!(
                        (actual - expected).abs() < 0.008,
                        "{name}, RR={rr}: {actual}, expected {expected}"
                    );
                }
                drop(data);
                readback.unmap();
            }
        }
    });
}
