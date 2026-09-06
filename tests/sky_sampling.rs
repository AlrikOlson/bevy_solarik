//! Executes the actual sky-distribution WGSL with analytic sky inputs.
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a Vulkan GPU; run separately from builds and scene captures"]
fn sky_distribution_gpu() {
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
        let source = include_str!("../src/realtime/sky_sampling.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        // Substitute only the external scene, RNG and cosine sampler imports.
        let source = format!(
            "{}\n{}\n{}",
            r#"
const PI: f32 = 3.141592653589793;
@group(0) @binding(0) var<storage> sky_case: u32;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
fn sample_sky(d: vec3<f32>) -> vec3<f32> {
    if sky_case == 0u { return vec3(0.0); }
    if sky_case == 1u { return vec3(1.0); }
    if sky_case == 3u { return vec3(select(0.0, 20.0, d.z > 0.5)); }
    return vec3(select(1.0, 20.0, d.z > 0.5));
}
fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
fn rand_f(rng: ptr<function, u32>) -> f32 {
    *rng = *rng * 1664525u + 1013904223u;
    return f32(*rng >> 8u) / 16777216.0;
}
fn sample_cosine_hemisphere(n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let r = sqrt(rand_f(rng));
    let phi = 2.0 * PI * rand_f(rng);
    return vec3(r * cos(phi), sqrt(1.0 - r * r), r * sin(phi));
}
"#,
            source,
            r#"
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= 65536u { return; }
    var rng = id.x * 747796405u + 2891336453u;
    let s = sample_sky_mixture(vec3(0.0, 1.0, 0.0), &rng);
    let cosine = max(s.direction.y, 0.0);
    let estimate = sample_sky(s.direction).x * cosine * s.inverse_pdf;
    let address = sky_address(s.direction);
    let rebuilt = sky_direction(address.face, address.uv);
    output[id.x] = vec4(estimate, dot(rebuilt, s.direction), s.inverse_pdf, s.direction.z);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("actual sky WGSL with analytic sky"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let distribution = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (6 * 128 * 128 + 6 * 128) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
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
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[storage_entry(0, true), storage_entry(1, false)],
        });
        let distribution_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[storage_entry(24, false)],
            });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&input_layout), Some(&distribution_layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let rows = pipeline("build_sky_rows");
        let marginal = pipeline("build_sky_marginal");
        let probe = pipeline("probe");
        let distribution_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &distribution_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 24,
                resource: distribution.as_entire_binding(),
            }],
        });
        // End with black again to catch stale data after a bright frame.
        for case in [0u32, 1, 2, 3, 0] {
            let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&case),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &input_layout,
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
                pass.set_bind_group(0, &group, &[]);
                pass.set_bind_group(1, &distribution_group, &[]);
                pass.set_pipeline(&rows);
                pass.dispatch_workgroups(768, 1, 1);
                pass.set_pipeline(&marginal);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&probe);
                pass.dispatch_workgroups(1024, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).expect("send map result");
                });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU finished");
            receiver.recv().expect("map callback").expect("map");
            let data = readback.slice(..).get_mapped_range();
            let samples: &[[f32; 4]] = bytemuck::cast_slice(&data);
            for sample in samples {
                assert!(
                    sample.iter().all(|x| x.is_finite()),
                    "case {case}: {sample:?}"
                );
                assert!(
                    sample[1] > 0.99999,
                    "cubemap orientation roundtrip: {sample:?}"
                );
                assert!(sample[2] > 0.0, "positive reciprocal mixture density");
            }
            let mean = samples.iter().map(|s| f64::from(s[0])).sum::<f64>() / samples.len() as f64;
            // Independent projected-disk integral: constant sky is pi.
            // z > 0.5 is a circular segment of the unit disk.
            let segment = 0.5_f64.acos() - 0.5 * 0.75_f64.sqrt();
            let expected = match case {
                0 => 0.0,
                1 => core::f64::consts::PI,
                2 => core::f64::consts::PI + 19.0 * segment,
                _ => 20.0 * segment,
            };
            assert!(
                (mean - expected).abs() < 0.02 * expected.max(1.0),
                "case {case}: {mean} versus {expected}"
            );
            drop(data);
            readback.unmap();
        }
    });
}
