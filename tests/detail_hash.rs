//! Regression for correlated two-dimensional scan-patch translations.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires Vulkan; serialize with native captures"]
#[expect(clippy::print_stdout, reason = "record measured GPU hash diagnostics")]
fn scan_translations_cover_two_dimensions() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let production = include_str!("../src/detail_sampling.wgsl");
        let start = production.find("fn detail_hash(").unwrap();
        let end = production.find("fn detail_plane(").unwrap();
        let shader = format!(
            "{}\n{}",
            &production[start..end],
            r"
@group(0) @binding(0) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
    let p = vec2i(i32(id.x % 64u) - 32, i32((id.x / 64u) % 64u) - 32);
    let layer = id.x / 4096u;
    let turn = detail_patch_turn(p, layer);
    let gradient = vec2f(0.2, -0.3);
    let delta = vec2f(0.001, 0.002);
    let transformed = detail_rotate(gradient, (4u-turn)%4u);
    let error = abs(dot(gradient, detail_rotate(delta, turn))-dot(transformed, delta));
    output[id.x] = vec4f(detail_hash(p, layer), error, f32(turn));
}"
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production scan translation hash"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let count = 32768;
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &vec![0u8; count * 16],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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
            pass.dispatch_workgroups(count as u32 / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = readback.slice(..).get_mapped_range();
        let values: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        let mut turns = [0usize; 4];
        let mut histogram = [0usize; 256];
        let mut covariance = 0.0;
        let mut variance = [0.0; 2];
        for (index, &[x, y, error, turn]) in values.iter().enumerate() {
            assert!(
                error < 1e-8,
                "rotated gradient violates height differential"
            );
            let layer = index / 4096;
            if layer == 5 || layer == 6 {
                turns[turn as usize] += 1;
            } else {
                assert_eq!(
                    turn, 0.0,
                    "directional rock/ice scans keep their orientation"
                );
            }
            assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
            histogram[(x * 16.0) as usize + 16 * (y * 16.0) as usize] += 1;
            let a = f64::from(x) - 0.5;
            let b = f64::from(y) - 0.5;
            covariance += a * b;
            variance[0] += a * a;
            variance[1] += b * b;
        }
        let correlation = covariance / (variance[0] * variance[1]).sqrt();
        assert!(
            correlation.abs() < 0.05,
            "scan UV translations correlate: {correlation}"
        );
        assert!(
            histogram.iter().all(|n| (60..200).contains(n)),
            "scan translations miss 2D regions: {histogram:?}"
        );
        assert!(
            turns.iter().all(|n| (1800..2300).contains(n)),
            "vegetation patch orientations missing: {turns:?}"
        );
        println!("32768 production translations: correlation {correlation}; all 256 bins occupied");
    });
}
