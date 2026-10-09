use super::acceleration::Scene;
use wgpu::util::DeviceExt;
pub fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shared = bevy_solarik::coarse_spatial::PACK_SHADER
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("original spatial hit bake"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "enable wgpu_ray_query;\n{shared}\n{}",
                include_str!("../coarse_spatial_cook.wgsl")
            )
            .into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    scene: &Scene,
    tasks: &[[f32; 8]],
) -> Vec<[u32; 64]> {
    let tasks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(tasks),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let count = tasks.size() / 32;
    let size = count * 256;
    let buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let output = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let alpha = scene.alpha.create_view(&Default::default());
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: scene.tlas.as_binding(),
    }];
    entries.extend(
        scene.buffers[..4]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: b.as_entire_binding(),
            }),
    );
    entries.extend([
        wgpu::BindGroupEntry {
            binding: 5,
            resource: tasks.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::TextureView(&alpha),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::Sampler(&scene.sampler),
        },
        wgpu::BindGroupEntry {
            binding: 8,
            resource: output.as_entire_binding(),
        },
    ]);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    assert!(scene.explicit_bytes + size * 2 + tasks.size() < 512 << 20);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice::<u8, [u32; 64]>(&readback.get_mapped_range(..)).to_vec()
}
