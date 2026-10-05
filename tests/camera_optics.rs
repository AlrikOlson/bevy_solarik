//! Production compute equations; rgba32 storage isolates math from half-float rounding.
use wgpu::util::DeviceExt;
fn shader(definition: &str) -> String {
    let mut stack = vec![true];
    let mut out = String::new();
    for line in include_str!("../src/optics/camera.wgsl").lines() {
        if let Some(d) = line.strip_prefix("#ifdef ") {
            stack.push(*stack.last().unwrap() && d == definition);
        } else if let Some(d) = line.strip_prefix("#ifndef ") {
            stack.push(*stack.last().unwrap() && d != definition);
        } else if line == "#else" {
            let v = stack.pop().unwrap();
            stack.push(*stack.last().unwrap() && !v);
        } else if line == "#endif" {
            stack.pop();
        } else if *stack.last().unwrap() {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.replace("rgba16float", "rgba32float")
}
#[test]
#[ignore = "production Vulkan camera optics readback"]
fn physical_meter_and_conservative_scatter() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let pipelines: [_; 3] = ["METER", "HORIZONTAL", ""].map(|d| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader(d).into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some(if d == "METER" { "meter" } else { "scatter" }),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let size = wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        };
        let texture = || {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (source, scratch, result) = (texture(), texture(), texture());
        let (a, b, c) = (
            source.create_view(&Default::default()),
            scratch.create_view(&Default::default()),
            result.create_view(&Default::default()),
        );
        let state = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 48,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut parameters = [
            0.0f32,
            15.0,
            0.0,
            0.1,
            2.2,
            0.01,
            1.0,
            4.0,
            -6.0,
            20.0,
            0.0,
            0.0,
            2.0f32.powi(-15) / 1.2,
            1.0,
            0.0,
            0.0,
        ];
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&parameters),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let group = |i: usize, resources: Vec<wgpu::BindingResource>| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines[i].get_bind_group_layout(0),
                entries: &resources
                    .into_iter()
                    .enumerate()
                    .map(|(i, r)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: r,
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let groups = [
            group(
                0,
                vec![
                    wgpu::BindingResource::TextureView(&a),
                    uniform.as_entire_binding(),
                    state.as_entire_binding(),
                ],
            ),
            group(
                1,
                vec![
                    wgpu::BindingResource::TextureView(&a),
                    uniform.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&b),
                ],
            ),
            group(
                2,
                vec![
                    wgpu::BindingResource::TextureView(&a),
                    wgpu::BindingResource::TextureView(&b),
                    uniform.as_entire_binding(),
                    state.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&c),
                ],
            ),
        ];
        for case in 0..9 {
            let mut pixels = vec![[0.0f32, 0.0, 0.0, 1.0]; 4096];
            if case == 8 {
                // A tiny centred source amid a faint sky, below the histogram's
                // area percentile. A centre spot must still meter the source.
                pixels.fill([0.0001, 0.0001, 0.0001, 1.0]);
                for y in 30..34 {
                    for x in 30..34 {
                        pixels[y * 64 + x] = [10000.0, 10000.0, 10000.0, 1.0];
                    }
                }
            } else if case == 7 {
                // Turn from a night-adapted view into daylight without a reset.
                pixels.fill([1.0; 4]);
                queue.write_buffer(&state, 0, bytemuck::cast_slice(&[-6.0f32; 12]));
            } else if case == 4 {
                pixels.fill([1e-9, 1e-9, 1e-9, 1.0]);
                for y in 0..64i32 {
                    for x in 0..64i32 {
                        if (x - 32).pow(2) + (y - 32).pow(2) < 100 {
                            pixels[(y * 64 + x) as usize] = [4096.0 * parameters[12]; 4];
                        }
                    }
                }
            } else if case == 0 {
                pixels.fill([4096.0 * parameters[12]; 4]);
            } else if case >= 5 {
                pixels.fill([16384.0 * parameters[12]; 4]);
            } else {
                pixels[if case == 1 { 2080 } else { 0 }] = [1.0, 0.5, 0.25, 1.0];
            }
            parameters[0] = f32::from(case == 3 || case == 5 || case >= 7);
            parameters[13] = f32::from(case != 6 && case != 7);
            queue.write_buffer(&uniform, 0, bytemuck::cast_slice(&parameters));
            queue.write_texture(
                source.as_image_copy(),
                bytemuck::cast_slice(&pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1024),
                    rows_per_image: Some(64),
                },
                size,
            );
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 65584,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            for i in 0..3 {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipelines[i]);
                pass.set_bind_group(0, &groups[i], &[]);
                if i == 0 {
                    pass.dispatch_workgroups(1, 1, 1);
                } else {
                    pass.dispatch_workgroups(8, 8, 1);
                }
            }
            encoder.copy_texture_to_buffer(
                result.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(1024),
                        rows_per_image: Some(64),
                    },
                },
                size,
            );
            encoder.copy_buffer_to_buffer(&state, 0, &staging, 65536, 48);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let data = staging.slice(..).get_mapped_range();
            let output = bytemuck::cast_slice::<u8, f32>(&data);
            if case < 3 {
                for channel in 0..3 {
                    let before: f64 = pixels.iter().map(|p| f64::from(p[channel])).sum();
                    let after: f64 = (0..4096).map(|i| f64::from(output[i * 4 + channel])).sum();
                    assert!(
                        (after / before - 1.0).abs() < 0.000002,
                        "case{case} flux {before} {after}"
                    );
                }
            }
            if case == 0 || case == 4 {
                let measured = output[16386];
                assert!(
                    (measured.log2() - 12.0).abs() < 60.0 / 512.0,
                    "case{case} luminance{measured}"
                );
            }
            if case == 3 {
                assert_eq!(output[16384], 15.0, "black centre must hold manual EV");
            }
            if case == 5 {
                let expected =
                    bevy_solarik::optics::adapt_ev(15.0, f64::from(output[16385]), 0.1, 1.0, 4.0);
                assert!(
                    (f64::from(output[16384]) - expected).abs() < 1e-5,
                    "automatic relaxation must match the f64 model"
                );
            }
            if case == 6 {
                assert_eq!(
                    output[16384], 15.0,
                    "manual override must be immediate without reset"
                );
            }
            if case == 7 {
                assert!(output[..16384].iter().all(|v| v.is_finite()));
                assert_eq!(output[0], 65000.0, "saturated HDR remains representable");
            }
            if case == 8 {
                assert_eq!(
                    output[16385], 20.0,
                    "centre solar spot reaches the exposure bound"
                );
            }
            drop(data);
            staging.unmap();
        }
    });
}
