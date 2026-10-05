//! Read back the production sky interpolation, including the longitude seam.
#[test]
#[ignore = "Vulkan GPU; run serially with builds/captures"]
fn physical_float_radiance_survives_sampling() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let production = include_str!("../src/radiance_sky_sample.wgsl")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let fixture = r#"
@group(0) @binding(0) var field:texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;
@compute @workgroup_size(32)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x;
 var a=f32(i)*0.731;
 var b=0.1+f32(i%10u)*0.31;
 if i==0u {a=0.000001;b=1.2;}
 if i==1u {a=-0.000001;b=1.2;}
 if i==2u {b=0.0;}
 if i==3u {b=3.14159265359;}
 let d=vec3(sin(b)*cos(a),cos(b),-sin(b)*sin(a));
 let sampled=sample_sky(field,d);
 let colour=vec3(0.5,2.0,0.25);
 output[i]=vec4(sampled,1.0);
 if i>=16u { output[i]=vec4(radiance_transform(sampled,vec4(colour,1.0))+radiance_transform(sampled,vec4(1.0,1.0,1.0,0.0)),1.0); }
}
"#;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(format!("{production}\n{fixture}").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let size = wgpu::Extent3d {
            width: 8,
            height: 4,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texels: Vec<[f32; 4]> = (0..32)
            .map(|i| {
                let x = (i % 8) as f32;
                let y = (i / 8) as f32;
                [
                    0.001 + x * 0.0001,
                    0.002 + y * 0.0002,
                    0.003 + (x + y) * 0.00005,
                    1.,
                ]
            })
            .collect();
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * 16),
                rows_per_image: None,
            },
            size,
        );
        let view = texture.create_view(&Default::default());
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 512,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 512,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
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
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &read, 0, 512);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = read.slice(..).get_mapped_range();
        let actual = bytemuck::cast_slice::<u8, [f32; 4]>(&bytes);
        for (i, value) in actual.iter().enumerate().take(32) {
            // Poles have arbitrary longitude; test finite radiance and the correct latitude.
            if i == 2 || i == 3 {
                assert!(value.iter().all(|v| v.is_finite()));
                let y = if i == 2 { 0.002 } else { 0.0026 };
                assert!((value[1] - y).abs() < 1e-8);
                continue;
            }
            let (a, b) = if i == 0 {
                (0.000001_f64, 1.2_f64)
            } else if i == 1 {
                (-0.000001, 1.2)
            } else {
                (i as f64 * 0.731, 0.1 + (i % 10) as f64 * 0.31)
            };
            let x = a.rem_euclid(core::f64::consts::TAU) / core::f64::consts::TAU * 8.0 - 0.5;
            let y = b / core::f64::consts::PI * 4.0 - 0.5;
            let ix = x.floor() as i32;
            let iy = y.floor() as i32;
            let fx = x - x.floor();
            let fy = y - y.floor();
            for (c, channel) in value.iter().enumerate().take(3) {
                let t = |dx: i32, dy: i32| {
                    f64::from(
                        texels[((iy + dy).clamp(0, 3) * 8 + (ix + dx).rem_euclid(8)) as usize][c],
                    )
                };
                let mut expected = (1. - fy) * ((1. - fx) * t(0, 0) + fx * t(1, 0))
                    + fy * ((1. - fx) * t(0, 1) + fx * t(1, 1));
                if i >= 16 {
                    let red = |dx: i32, dy: i32| {
                        f64::from(
                            texels[((iy + dy).clamp(0, 3) * 8 + (ix + dx).rem_euclid(8)) as usize]
                                [0],
                        )
                    };
                    let scalar = (1. - fy) * ((1. - fx) * red(0, 0) + fx * red(1, 0))
                        + fy * ((1. - fx) * red(0, 1) + fx * red(1, 1));
                    expected += scalar * [0.5, 2.0, 0.25][c];
                }
                assert!(
                    (f64::from(*channel) - expected).abs() < 2e-8,
                    "{i}/{c} {} {expected}",
                    channel
                );
            }
        }
        for (a, b) in actual[0].iter().zip(actual[1].iter()).take(3) {
            assert!((a - b).abs() < 1e-8);
        }
    });
}
