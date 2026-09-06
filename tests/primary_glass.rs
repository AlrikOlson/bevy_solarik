//! Numerical primary-pane compositing with production WGSL and synthetic scene I/O.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn primary_glass_gpu() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Vulkan");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let production = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/realtime/primary_glass.wgsl"
        ))
        .expect("primary glass shader");
        let start = production
            .find("fn composite_primary_glass(")
            .expect("production compositor");
        let glass = include_str!("../src/scene/thin_glass.wgsl")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("primary glass transport"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{glass}\n{}\n{}",
                    &production[start..],
                    include_str!("primary_glass_fixture.wgsl")
                )
                .into(),
            ),
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
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // tint + coverage; pane count, opaque distance, reflectance, emission.
        let r = 1.0 / 13.0;
        let cases: [(&str, [f32; 4], [f32; 4], [f32; 3]); 11] = [
            ("no pane", [1.0; 4], [0.0, 100.0, 0.5, 0.0], [1.0; 3]),
            (
                "clear",
                [1.0; 4],
                [1.0, 100.0, 0.5, 0.0],
                [1.0, 1.0 - r, 1.0 - r],
            ),
            (
                "tint",
                [0.2, 0.5, 0.8, 1.0],
                [1.0, 100.0, 0.5, 0.0],
                [r + 0.2 * (1.0 - r), 0.5 * (1.0 - r), 0.8 * (1.0 - r)],
            ),
            (
                "hole",
                [0.2, 0.5, 0.8, 0.0],
                [1.0, 100.0, 0.5, 0.0],
                [1.0; 3],
            ),
            ("opaque occlusion", [0.0; 4], [1.0, 0.5, 0.5, 0.0], [1.0; 3]),
            (
                "two tinted panes",
                [0.5, 0.5, 0.5, 1.0],
                [2.0, 100.0, 0.0, 0.0],
                [0.25; 3],
            ),
            (
                "black",
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 100.0, 0.5, 0.0],
                [r, 0.0, 0.0],
            ),
            (
                "perfect reflector",
                [1.0; 4],
                [1.0, 100.0, 4.0, 0.0],
                [1.0, 0.0, 0.0],
            ),
            (
                "coverage emission",
                [1.0, 1.0, 1.0, 0.5],
                [1.0, 100.0, 0.0, 2.0],
                [2.0; 3],
            ),
            ("exact budget", [1.0; 4], [32.0, 100.0, 0.0, 0.0], [1.0; 3]),
            ("chain bound", [1.0; 4], [33.0, 100.0, 0.0, 0.0], [0.0; 3]),
        ];
        let cases = cases
            .into_iter()
            .map(|(name, pane, config, expected)| (name, pane, config, [0.0; 4], expected))
            .chain([
                (
                    "hidden black wall",
                    [1.0; 4],
                    [1.0, 100.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0; 3],
                ),
                (
                    "hidden green wall",
                    [1.0; 4],
                    [1.0, 100.0, 0.5, 0.0],
                    [2.0, 0.0, 0.5, 0.0],
                    [r, 0.5 * (1.0 - r), 0.0],
                ),
                (
                    "opaque before glass preserves raster",
                    [1.0; 4],
                    [1.0, 100.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [1.0; 3],
                ),
                (
                    "wall beyond raster preserves raster",
                    [1.0; 4],
                    [1.0, 1.5, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [1.0; 3],
                ),
            ]);
        for (name, pane, config, hidden, expected) in cases {
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[pane, config, hidden]),
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
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 16);
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                sender.send(r).expect("map result");
            });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU completed");
            receiver.recv().expect("callback").expect("map");
            let data = readback.slice(..).get_mapped_range();
            let values: &[f32] = bytemuck::cast_slice(&data);
            for (actual, expected) in values.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{name}: {values:?} vs {expected}"
                );
            }
            drop(data);
            readback.unmap();
        }
    });
}
