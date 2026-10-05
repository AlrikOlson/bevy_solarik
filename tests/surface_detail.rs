//! Precision contract for physically sized scan coordinates.
use bevy_math::{DVec3, Vec3};
use bevy_solarik::surface_detail::DetailCoordinates;

#[test]
fn neighbouring_origins_preserve_millimetres_on_a_planet() {
    let origin = DVec3::new(6_371_000.123_456, -4_000_000.654_321, 123.456);
    let sizes = [1.5, 1.8, 2.0, 2.51, 3.15, 2.0, 2.0, 2.0];
    let a = DetailCoordinates::new(origin, sizes).unwrap();
    let delta = DVec3::new(37.25, -14.75, 0.125);
    let b = DetailCoordinates::new(origin + delta, sizes).unwrap();
    for (i, size) in sizes.into_iter().enumerate() {
        let local = Vec3::new(0.001, 0.002, -0.003);
        let pa = a.reconstruct(i, local);
        let pb = b.reconstruct(i, local - delta.as_vec3());
        let expected = (origin + local.as_dvec3()) / size;
        assert!((pa - expected).abs().max_element() * size < 0.00001);
        assert!((pa - pb).abs().max_element() * size < 0.00001);
    }
    assert!(DetailCoordinates::new(origin, [0.0; 8]).is_none());
    assert!(DetailCoordinates::new(DVec3::NAN, sizes).is_none());
    assert!(DetailCoordinates::new(DVec3::ZERO, [f64::MIN_POSITIVE; 8]).is_none());
    assert!(DetailCoordinates::new(DVec3::ZERO, [f64::MAX; 8]).is_none());
}

#[test]
#[ignore = "requires Vulkan; run separately from builds and captures"]
fn gpu_surface_gradient_matches_height_differential() {
    use wgpu::util::DeviceExt;
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits {
                    max_binding_array_elements_per_shader_stage: 4,
                    max_binding_array_sampler_elements_per_shader_stage: 2,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();
        let mut enabled = true;
        let production = include_str!("../src/detail_sampling.wgsl")
            .lines()
            .filter(|line| {
                if line.starts_with("#ifdef") {
                    enabled = true;
                    return false;
                }
                if line.starts_with("#else") {
                    enabled = false;
                    return false;
                }
                if line.starts_with("#endif") {
                    enabled = true;
                    return false;
                }
                enabled && !line.starts_with('#')
            })
            .collect::<Vec<_>>()
            .join("\n")
            .replace(
                "binding_array<texture_2d_array<f32>>",
                "binding_array<texture_2d_array<f32>, 2>",
            )
            .replace("binding_array<sampler>", "binding_array<sampler, 2>");
        let source = format!(
            "{production}\n{}",
            r#"
@group(0) @binding(0) var<storage> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
    var coordinates: DetailCoordinates;
    coordinates.phase[0] = vec4f(0.0, 0.0, 0.0, 0.5);
    let index = u32(inputs[2u * id.x].w);
    let sample = sample_surface_detail(index, index, index, coordinates,
        vec3f(f32(id.x) * 0.1), inputs[2u * id.x].xyz, 0.001, vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0));
    outputs[id.x] = vec4f(detail_normal(inputs[2u * id.x].xyz, inputs[2u * id.x + 1u].xyz), sample.colour.x);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production surface-gradient differential"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let mut entries = Vec::new();
        for binding in 0..2 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 0,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 21,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: core::num::NonZeroU32::new(2),
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 22,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: core::num::NonZeroU32::new(2),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut inputs = Vec::new();
        for i in 0..64 {
            let t = i as f32 * core::f32::consts::TAU / 64.0;
            inputs.push(
                Vec3::new(t.cos(), 0.2, t.sin())
                    .normalize()
                    .extend(0.0)
                    .to_array(),
            );
            inputs.push([0.2, -0.3, 0.1, 0.0]);
        }
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 8,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            &vec![128u8; 16 * 16 * 8 * 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16 * 4),
                rows_per_image: Some(16),
            },
            texture.size(),
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
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
                    binding: 21,
                    resource: wgpu::BindingResource::TextureViewArray(&[&view, &view]),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: wgpu::BindingResource::SamplerArray(&[&sampler, &sampler]),
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
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range();
        let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        for (i, value) in values.iter().enumerate() {
            let n = Vec3::from_slice(&inputs[2 * i]);
            let g = Vec3::from_slice(&inputs[2 * i + 1]);
            // The displaced surface X(u,v) = uT + vB + h(u,v)N has
            // tangents T + (g.T)N and B + (g.B)N. Their cross is its normal.
            let t = n.any_orthonormal_vector();
            let b = n.cross(t);
            let expected = (t + g.dot(t) * n).cross(b + g.dot(b) * n).normalize();
            assert!((Vec3::from_slice(value) - expected).length() < 1e-6);
            assert!(
                (value[3] - 256.0 / 255.0).abs() < 1e-5,
                "neutral scan must preserve its mean: {value:?}"
            );
        }
    });
}
