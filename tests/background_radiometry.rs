//! Production mask/composition shader: neural surfaces survive, physical sky does too.
use wgpu::util::DeviceExt;
#[test]
#[ignore = "production Vulkan background preservation readback"]
fn preserve_only_guide_free_background() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/optics/background.wgsl")
                    .replace("rgba16float", "rgba32float")
                    .into(),
            ),
        });
        let entries = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: match binding {
                    0 | 1 => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    2 => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    _ => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                },
            })
            .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipes = ["save", "restore"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
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
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let original = texture();
        let neural = texture();
        let reference = texture();
        let result = texture();
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let views = [&original, &neural, &reference, &result, &depth]
            .map(|t| t.create_view(&Default::default()));
        let make = |indices: [usize; 4]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &indices
                    .into_iter()
                    .enumerate()
                    .map(|(binding, i)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: wgpu::BindingResource::TextureView(&views[i]),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let groups = [make([0, 0, 4, 2]), make([1, 2, 4, 3])];
        let physical = [1.0f32, 0.5, 0.1, 1.0];
        let processed = [0.2f32, 0.3, 0.4, 1.0];
        for (texture, pixel) in [(&original, physical), (&neural, processed)] {
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&pixel),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: None,
                },
                size,
            );
        }
        for depth_value in [0.0, 0.5] {
            let readback = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[0; 256],
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &views[4],
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth_value),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            for i in 0..2 {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipes[i]);
                pass.set_bind_group(0, &groups[i], &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_texture_to_buffer(
                result.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                size,
            );
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let data = readback.slice(..).get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&data[..16]);
            assert_eq!(
                values,
                if depth_value == 0.0 {
                    &physical
                } else {
                    &processed
                },
                "depth{depth_value}"
            );
            drop(data);
            readback.unmap();
        }
    });
}
