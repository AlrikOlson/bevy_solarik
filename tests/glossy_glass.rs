//! Execute the production glossy path against deterministic synthetic intersections.
//! The fixture replaces scene I/O, not transport or thin-pane sampling.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn glossy_glass_gpu() {
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
        let source = include_str!("../src/realtime/specular_gi.wgsl");
        let start = source
            .find("fn trace_glossy_path(")
            .expect("production path");
        let end = source
            .find("// https://en.wikipedia.org/wiki/Householder_transformation")
            .expect("PSR helpers");
        for rr_guides in [false, true] {
            let mut excluded = false;
            let transport = source[start..end]
                .lines()
                .filter(|line| {
                    if line.starts_with("#ifdef") {
                        excluded = !rr_guides;
                        return false;
                    }
                    if line.starts_with("#endif") {
                        excluded = false;
                        return false;
                    }
                    !excluded
                })
                .collect::<Vec<_>>()
                .join("\n");
            let glass = include_str!("../src/scene/thin_glass.wgsl")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect::<Vec<_>>()
                .join("\n");
            let source = format!(
                "{glass}\n{transport}\n{}",
                include_str!("glossy_glass_fixture.wgsl")
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("production glossy transport"),
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
                size: 32,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // name, tint/coverage, reflectance/random/count/primary roughness, expected radiance.
            let cases: [(&str, [f32; 4], [f32; 4], [f32; 3]); 10] = [
                (
                    "opaque PSR control",
                    [1.0; 4],
                    [0.5, 0.5, 0.0, 0.001],
                    [1.0; 3],
                ),
                (
                    "tint",
                    [0.2, 0.5, 0.8, 1.0],
                    [0.5, 0.5, 1.0, 0.001],
                    [0.2, 0.5, 0.8],
                ),
                (
                    "clear",
                    [1.0, 1.0, 1.0, 1.0],
                    [0.5, 0.5, 1.0, 0.001],
                    [1.0; 3],
                ),
                (
                    "hole",
                    [0.2, 0.5, 0.8, 0.0],
                    [0.5, 0.5, 1.0, 0.001],
                    [1.0; 3],
                ),
                (
                    "reflection",
                    [0.2, 0.5, 0.8, 1.0],
                    [0.5, 0.0, 1.0, 0.001],
                    [1.0, 0.0, 0.0],
                ),
                (
                    "four panes retain opaque budget",
                    [0.5, 0.5, 0.5, 1.0],
                    [0.5, 0.5, 4.0, 0.001],
                    [0.0625; 3],
                ),
                ("chain cap", [1.0; 4], [0.5, 0.5, 33.0, 0.001], [0.0; 3]),
                (
                    "transmission retains ReSTIR emission ownership",
                    [1.0; 4],
                    [0.5, 0.5, 1.0, 0.1],
                    [0.0; 3],
                ),
                (
                    "delta reflection clears ReSTIR emission ownership",
                    [1.0; 4],
                    [0.5, 0.0, 1.0, 0.1],
                    [1.0, 0.0, 0.0],
                ),
                (
                    "black transmission",
                    [0.0, 0.0, 0.0, 1.0],
                    [0.5, 0.5, 1.0, 0.001],
                    [0.0; 3],
                ),
            ];
            let cases = cases.into_iter().chain([
                (
                    "analytic smooth primary",
                    [1.0; 4],
                    [0.5, 0.5, 0.0, 0.01],
                    [2.0; 3],
                ),
                (
                    "analytic NEE primary",
                    [1.0; 4],
                    [0.5, 0.5, 0.0, 0.1],
                    [0.0; 3],
                ),
                (
                    "analytic glass reflection",
                    [1.0; 4],
                    [0.5, 0.0, 1.0, 0.1],
                    [2.0; 3],
                ),
                (
                    "analytic straight NEE",
                    [1.0; 4],
                    [0.5, 0.5, 1.0, 0.1],
                    [0.0; 3],
                ),
                (
                    "diffuse accepted",
                    [1.0, 0.0, 0.0, 0.5],
                    [0.5, 0.25, 1.0, 0.001],
                    [0.0; 3],
                ),
                (
                    "diffuse skipped",
                    [1.0, 0.0, 0.0, 0.5],
                    [0.5, 0.75, 1.0, 0.001],
                    [1.0; 3],
                ),
                (
                    "diffuse opaque",
                    [1.0; 4],
                    [0.5, 0.75, 1.0, 0.001],
                    [0.0; 3],
                ),
                (
                    "diffuse hole",
                    [1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 1.0, 0.001],
                    [1.0; 3],
                ),
            ]);
            for (name, pane, config, expected) in cases {
                let mode = [
                    if name.starts_with("diffuse") {
                        1.0
                    } else {
                        0.0
                    },
                    if name.starts_with("analytic") {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ];
                let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[pane, config, mode]),
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
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
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
                let result: &[f32] = bytemuck::cast_slice(&data);
                for (actual, expected) in result[..3].iter().zip(expected) {
                    assert!(
                        (actual - expected).abs() < 0.00001,
                        "{name}: {result:?}, expected channel {expected}"
                    );
                }
                assert!(result[3] <= 33.0, "bounded traversal: {name}");
                let expected_replacements = if rr_guides && config[2] == 0.0 && config[3] <= 0.002 {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(
                    result[4], expected_replacements,
                    "PSR ownership: {name}, RR {rr_guides}"
                );
                drop(data);
                readback.unmap();
            }
        }
    });
}
