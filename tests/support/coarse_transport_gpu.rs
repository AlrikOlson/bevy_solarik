use super::Probe;
use wgpu::util::DeviceExt;
pub fn pipeline(device: &wgpu::Device, entry: &str) -> wgpu::ComputePipeline {
    let source = [
        bevy_solarik::coarse_scene::SHADER,
        bevy_solarik::coarse_scene::WALK_SHADER,
        bevy_solarik::coarse_transport::SHADER,
        include_str!("../coarse_transport.wgsl"),
    ]
    .map(|s| {
        s.lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    })
    .join("\n");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bounded coarse transport diagnostic"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    inputs: &[(u32, &[u8])],
    count: usize,
) -> (Vec<Probe>, u64) {
    let buffers: Vec<_> = inputs
        .iter()
        .map(|(_, data)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    let size = (count * size_of::<Probe>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut entries: Vec<_> = buffers
        .iter()
        .zip(inputs)
        .map(|(buffer, (binding, _))| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: 4,
        resource: output.as_entire_binding(),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let bytes = buffers.iter().map(wgpu::Buffer::size).sum::<u64>() + size * 2;
    (
        bytemuck::cast_slice::<u8, Probe>(&readback.get_mapped_range(..)).to_vec(),
        bytes,
    )
}
