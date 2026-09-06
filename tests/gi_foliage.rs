//! Production `ReSTIR` GI foliage endpoint energy, side support, PDFs and history.
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

fn preprocess(source: &str, defines: &[&str]) -> String {
    let mut enabled = vec![true];
    let mut result = String::new();
    for line in source.lines() {
        if let Some(name) = line.strip_prefix("#ifdef ") {
            enabled.push(*enabled.last().expect("parent") && defines.contains(&name));
        } else if let Some(name) = line.strip_prefix("#ifndef ") {
            enabled.push(*enabled.last().expect("parent") && !defines.contains(&name));
        } else if line == "#else" {
            let previous = enabled.pop().expect("branch");
            enabled.push(*enabled.last().expect("parent") && !previous);
        } else if line == "#endif" {
            enabled.pop().expect("branch");
        } else if *enabled.last().expect("state") {
            result.push_str(line);
            result.push('\n');
        }
    }
    assert_eq!(enabled, [true], "balanced shader conditionals");
    result
}

async fn probe(source: String, inputs: &[[[f32; 4]; 6]]) -> Vec<Vec<[f32; 4]>> {
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
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production foliage cache"),
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
        size: 7 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut results = Vec::new();
    for input in inputs {
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(input),
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
        results.push(bytemuck::cast_slice(&data).to_vec());
        drop(data);
        readback.unmap();
    }
    results
}

fn check_endpoints(no_cache: bool) {
    let gi = include_str!("../src/realtime/restir_gi.wgsl");
    let sampling = include_str!("../src/scene/sampling.wgsl");
    let mut source = include_str!("gi_foliage_fixture.wgsl").to_owned();
    for name in [
        "generate_initial_reservoir",
        "empty_reservoir",
        "merge_reservoirs",
        "jacobian",
        "isinf",
        "age_temporal_reservoir",
    ] {
        source.push_str(&preprocess(
            &function(gi, name),
            if no_cache { &["NO_WORLD_CACHE"] } else { &[] },
        ));
    }
    source.push_str(&function(
        include_str!("../src/realtime/world_cache_query.wgsl"),
        "query_two_sided_world_cache",
    ));
    for name in [
        "sample_random_light_transmitted",
        "shade_gi_connection",
        "balance_heuristic",
        "isnan",
    ] {
        source.push_str(&function(sampling, name));
    }
    let mut inputs = Vec::new();
    for t in [0.0f32, 0.25, 0.5, 1.0] {
        for side in [-1.0, 1.0] {
            for light_side in [-1.0, 1.0] {
                for tilt in [0.0, 4.0] {
                    inputs.push([
                        [t, side, 0.0, tilt],
                        [0.2, 0.5, 0.8, 0.0],
                        [2.0, 4.0, 8.0, 0.0],
                        [10.0, 6.0, 3.0, 0.0],
                        [light_side, 0.8, 0.0, 0.0],
                        [0.25, 0.5, 0.75, 0.0],
                    ]);
                }
            }
        }
    }
    for mode in 1..=3 {
        let mut input = inputs[0];
        input[0] = [0.5, 1.0, mode as f32, 0.0];
        inputs.push(input);
    }
    // Black current connection must not destroy a reusable lit endpoint.
    let mut black = inputs[16];
    black[1] = [0.0; 4];
    inputs.push(black);
    let results = futures_lite::future::block_on(probe(source, &inputs));
    for (input, result) in inputs.iter().zip(results) {
        let [t, side, mode, tilt] = input[0];
        assert!(
            result.iter().flatten().all(|v| v.is_finite()),
            "nonfinite {input:?}: {result:?}"
        );
        let close = |actual: f32, expected: f32| {
            assert!(
                (actual - expected).abs() < 3e-5,
                "expected {expected}, got {actual}; {input:?}: {result:?}"
            );
        };
        let weight = if mode == 1.0 || mode == 3.0 {
            0.0
        } else if no_cache && mode == 0.0 {
            12.0
        } else {
            4.0
        };
        close(result[0][3], weight);
        for (channel, albedo) in [0.2, 0.5, 0.8].into_iter().enumerate() {
            let expected = if mode == 1.0 || mode == 3.0 {
                0.0
            } else if mode == 2.0 {
                [0.3, 0.6, 0.9][channel]
            } else if no_cache {
                let [light_side, cosine, _, _] = input[4];
                let signed_cosine = (tilt * (1.0 - cosine * cosine).sqrt()
                    - side * light_side * cosine)
                    / (1.0 + tilt * tilt).sqrt();
                let lobe = if signed_cosine >= 0.0 { 1.0 - t } else { t };
                input[2][channel] * input[5][channel] * signed_cosine.abs() * lobe * albedo
                    / core::f32::consts::PI
            } else {
                let (front, back) = if side < 0.0 {
                    (input[2][channel], input[3][channel])
                } else {
                    (input[3][channel], input[2][channel])
                };
                ((1.0 - t) * front + t * back) * albedo / core::f32::consts::PI
            };
            close(result[0][channel], expected);
            if mode == 0.0 {
                close(result[2][channel], expected * weight * input[1][channel]);
                close(result[3][channel], expected);
                if t > 0.0 {
                    close(result[4][channel], 0.0);
                }
            }
        }
        let queries = if no_cache || mode != 0.0 {
            0.0
        } else if t == 0.0 || t == 1.0 {
            1.0
        } else {
            2.0
        };
        close(result[5][2], queries);
        close(result[5][3], 0.0); // offsets, endpoint exclusion and cache lifetime
        close(
            result[2][3],
            1.0 + queries + if no_cache && mode == 0.0 { 1.0 } else { 0.0 },
        );
        close(result[1][3], if mode == 3.0 { 0.0 } else { 1.0 });
        if mode != 0.0 {
            continue;
        }
        if t > 0.0 {
            close(result[1][0], 0.0);
            close(result[1][2], -side);
            close(result[5][0], (4.0f32 / 4.04).powf(1.5));
            close(result[5][1], 0.0);
        } else {
            close(result[1][0], tilt / (1.0 + tilt * tilt).sqrt());
            close(result[1][2], -side / (1.0 + tilt * tilt).sqrt());
        }
        if result[0][..3].iter().any(|v| *v > 0.0) {
            close(result[3][3], weight);
        }
        close(result[4][3], 7.0);
        assert_eq!(result[6], [0.0; 4], "history expiry");
    }
}

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn gi_foliage_cache_gpu() {
    check_endpoints(false);
}

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn gi_foliage_direct_gpu() {
    check_endpoints(true);
}
