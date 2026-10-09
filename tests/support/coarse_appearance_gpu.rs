use super::{
    acceleration::Scene,
    appearance_input::{Appearance, Request},
};
use wgpu::util::DeviceExt;
pub struct Uploaded {
    buffers: [wgpu::Buffer; 2],
    views: [wgpu::TextureView; 3],
    pub bytes: u64,
}
pub fn upload(device: &wgpu::Device, queue: &wgpu::Queue, input: &Appearance) -> Uploaded {
    let storage = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("original appearance"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let buffers = [
        storage(bytemuck::cast_slice(&input.vertices)),
        storage(bytemuck::cast_slice(&input.materials)),
    ];
    let views = core::array::from_fn(|role| {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("original level-zero appearance array"),
                size: wgpu::Extent3d {
                    width: input.width,
                    height: input.height,
                    depth_or_array_layers: input.materials.len() as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if role == 0 {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &input.textures[role],
        );
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    });
    let bytes = buffers.iter().map(wgpu::Buffer::size).sum::<u64>()
        + input.textures.iter().map(|t| t.len() as u64).sum::<u64>();
    Uploaded {
        buffers,
        views,
        bytes,
    }
}
pub fn pipeline(device: &wgpu::Device, entry: &str) -> wgpu::ComputePipeline {
    let shared = bevy_solarik::coarse_appearance::SHADER
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("source appearance cook and material probes"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "enable wgpu_ray_query;\n{shared}\n{}\n{}",
                include_str!("../coarse_appearance.wgsl"),
                include_str!("../coarse_appearance_probe.wgsl")
            )
            .into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
pub fn run<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    scene: &Scene,
    appearance: &Uploaded,
    count: usize,
    requests: Option<&[Request]>,
) -> (Vec<T>, u64) {
    let bytes = (count * size_of::<T>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let alpha = scene.alpha.create_view(&Default::default());
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 7,
        resource: wgpu::BindingResource::Sampler(&scene.sampler),
    }];
    if requests.is_none() {
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: scene.tlas.as_binding(),
        });
        entries.extend(
            scene
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: b.as_entire_binding(),
                }),
        );
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::TextureView(&alpha),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: output.as_entire_binding(),
        });
    }
    let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: appearance.buffers[0].as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: appearance.buffers[1].as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&appearance.views[0]),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&appearance.views[1]),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(&appearance.views[2]),
        },
    ];
    let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(1),
        entries: &entries,
    });
    let requests = requests.map(|r| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(r),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    let group2 = requests.as_ref().map(|r| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    });
    let total = scene.explicit_bytes
        + appearance.bytes
        + 2 * bytes
        + requests.as_ref().map_or(0, wgpu::Buffer::size);
    assert!(total < 512 << 20);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        if let Some(group) = &group2 {
            pass.set_bind_group(2, group, &[]);
        }
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    (
        bytemuck::cast_slice::<u8, T>(&readback.get_mapped_range(..)).to_vec(),
        total,
    )
}
