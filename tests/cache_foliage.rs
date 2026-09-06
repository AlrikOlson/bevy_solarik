//! Production cache recurrence with controlled hemisphere irradiance and ray transport.
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

async fn probe(source: String, inputs: &[[[f32; 4]; 4]]) -> Vec<Vec<[f32; 4]>> {
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
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let life = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4096 * 4,
        usage: wgpu::BufferUsages::STORAGE,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: life.as_entire_binding(),
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

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn cache_foliage_propagation_gpu() {
    let cache = include_str!("../src/realtime/world_cache_update.wgsl");
    let mut source = include_str!("cache_foliage_fixture.wgsl").to_owned();
    source.push_str(&function(
        include_str!("../src/realtime/world_cache_query.wgsl"),
        "query_two_sided_world_cache",
    ));
    source.push_str(
        &function(cache, "sample_gi")
            .replace("@builtin(workgroup_id) ", "")
            .replace("@builtin(global_invocation_id) ", ""),
    );
    let mut inputs = Vec::new();
    for t in [0.0f32, 0.25, 0.5, 1.0] {
        for normal in [1.0, -1.0] {
            for connection in [[1.0, 1.0, 1.0, 0.0], [0.2, 0.5, 0.8, 0.0], [0.0; 4]] {
                inputs.push([
                    [t, normal, 0.0, 0.0],
                    connection,
                    [2.0, 4.0, 8.0, 0.0],
                    [10.0, 6.0, 3.0, 0.0],
                ]);
            }
        }
    }
    for mode in 1..=4 {
        inputs.push([
            [0.5, 1.0, mode as f32, 0.0],
            [0.2, 0.5, 0.8, 0.0],
            [2.0; 4],
            [10.0; 4],
        ]);
    }
    let results = futures_lite::future::block_on(probe(source, &inputs));
    for (input, result) in inputs.iter().zip(results) {
        let [t, normal, mode, _] = input[0];
        for (channel, albedo) in [0.2, 0.5, 0.8].into_iter().enumerate() {
            let (front, back) = if normal > 0.0 {
                (input[2][channel], input[3][channel])
            } else {
                (input[3][channel], input[2][channel])
            };
            let expected = if mode == 0.0 {
                input[1][channel] * albedo * ((1.0 - t) * front + t * back)
            } else if mode == 1.0 {
                input[1][channel] * core::f32::consts::PI * [0.3, 0.6, 0.9][channel]
            } else {
                0.0
            };
            assert!(
                (result[0][channel] - expected).abs() < 2e-5,
                "input {input:?}: channel {channel}, expected {expected}, got {result:?}"
            );
        }
        let queries = if mode != 0.0 {
            0.0
        } else if t == 0.0 || t == 1.0 {
            1.0
        } else {
            2.0
        };
        assert_eq!(result[0][3], queries, "only query supported lobes");
        assert_eq!(
            result[1],
            [if mode >= 3.0 { 0.0 } else { 1.0 }, 0.0, 0.0, 0.0],
            "connection offset, endpoint exclusion, lifetime and dispatch: {input:?}"
        );
    }
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

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn cache_foliage_identity_gpu() {
    let cache = include_str!("../src/realtime/world_cache_query.wgsl");
    let fixture = r#"
@group(0) @binding(0) var<storage, read> config: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> world_cache_life: array<atomic<u32>>;
var<workgroup> world_cache_checksums: array<atomic<u32>, 64>;
struct Geometry { world_position: vec3<f32>, world_normal: vec3<f32> }
var<private> world_cache_geometry_data: array<Geometry, 64>;
var<private> world_cache_radiance: array<vec4<f32>, 64>;
const WORLD_CACHE_POSITION_BASE_CELL_SIZE = 0.15;
const WORLD_CACHE_POSITION_LOD_SCALE = 15.0;
const WORLD_CACHE_MAX_SEARCH_STEPS = 3u;
const WORLD_CACHE_EMPTY_CELL = 0u;
fn rand_f(rng: ptr<function, u32>) -> f32 { *rng += 1u; return 0.25; }
@compute @workgroup_size(1)
fn probe() {
    for (var i = 0u; i < 64u; i++) { atomicStore(&world_cache_life[i], 0u); }
    let p = vec3(1.0, 2.0, 3.0);
    let n = normalize(config[0].xyz);
    var rng = 0u;
    // Claim both sides, then populate their independent incident fields.
    let cold_front = query_world_cache(p, n, p, 1.0, 7u, &rng);
    let cold_back = query_world_cache(p, -n, p, 1.0, 3u, &rng);
    var front_key = 63u;
    var back_key = 63u;
    for (var i = 0u; i < 64u; i++) {
        if atomicLoad(&world_cache_checksums[i]) != 0u {
            if dot(world_cache_geometry_data[i].world_normal, n) > 0.99 {
                front_key = i;
                world_cache_radiance[i] = config[2];
            } else {
                back_key = i;
                world_cache_radiance[i] = config[3];
            }
        }
    }
    var direct_rng = 123u;
    var opaque_rng = 123u;
    let direct = query_world_cache(p, n, p, 1.0, 2u, &direct_rng);
    let opaque = query_two_sided_world_cache(p, n, 0.0, p, 1.0, 2u, &opaque_rng);
    output[1] = vec4(abs(direct - opaque), f32(direct_rng) - f32(opaque_rng));
    output[0] = vec4(query_two_sided_world_cache(p, n, config[0].w, p, 1.0, 5u, &rng), 0.0);
    output[2] = vec4(f32(front_key), f32(back_key), f32(atomicLoad(&world_cache_life[front_key])), f32(atomicLoad(&world_cache_life[back_key])));
    // Updating one field must leave the opposite entry intact.
    world_cache_radiance[front_key] = vec4(0.0);
    output[3] = vec4(query_world_cache(p, -n, p, 1.0, 1u, &rng) + cold_front + cold_back, 0.0);
}
"#;
    let mut source = fixture.to_owned();
    for name in [
        "query_two_sided_world_cache",
        "query_world_cache",
        "get_cell_size",
        "quantize_position",
        "quantize_normal",
        "compute_key",
        "compute_checksum",
        "pcg_hash",
        "iqint_hash",
        "wrap_key",
    ] {
        source.push_str(
            &preprocess(
                &function(cache, name),
                &[
                    "NO_JITTER_WORLD_CACHE",
                    "WORLD_CACHE_QUERY_ATOMIC_MAX_LIFETIME",
                ],
            )
            .replace("#{WORLD_CACHE_SIZE}", "32u"),
        );
    }
    let mut inputs = Vec::new();
    for normal in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [-0.5, -1.0, 0.01],
    ] {
        for t in [0.0, 0.25, 0.5, 1.0] {
            inputs.push([
                [normal[0], normal[1], normal[2], t],
                [0.0; 4],
                [2.0, 4.0, 8.0, 16.0],
                [10.0, 6.0, 3.0, 32.0],
            ]);
        }
    }
    let results = futures_lite::future::block_on(probe(source, &inputs));
    for (input, result) in inputs.iter().zip(results) {
        let t = input[0][3];
        for channel in 0..3 {
            let expected = (1.0 - t) * input[2][channel] + t * input[3][channel];
            assert!(
                (result[0][channel] - expected).abs() < 2e-5,
                "{input:?}: {result:?}"
            );
            assert_eq!(
                result[3][channel], input[3][channel],
                "opposite field changed"
            );
        }
        assert_eq!(result[1], [0.0; 4], "opaque energy or RNG changed");
        assert_ne!(result[2][0], result[2][1], "hemisphere cache alias");
        assert!(
            result[2][0] < 63.0 && result[2][1] < 63.0,
            "both sides must be allocated"
        );
        assert_eq!(result[2][2], 7.0, "atomic max must preserve front lifetime");
        assert_eq!(
            result[2][3],
            if t > 0.0 { 5.0 } else { 3.0 },
            "only consumed side gets refreshed"
        );
    }
}
