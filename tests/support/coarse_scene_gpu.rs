use super::{Probe, Request};
use bevy_solarik::coarse_scene::{self, PackedSource};
use wgpu::util::DeviceExt;

pub fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shared = coarse_scene::SHADER
        .lines()
        .filter(|line| !line.starts_with("#define_import_path"))
        .collect::<Vec<_>>()
        .join("\n");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production coarse sparse lookup/decoder"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{shared}\n{}", include_str!("../coarse_scene.wgsl")).into(),
        ),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
pub fn run(
    device: &bevy_render::renderer::RenderDevice,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    source: &PackedSource,
    requests: &[Request],
) -> (Vec<Probe>, u64) {
    let resident = source.upload(device);
    let device = device.wgpu_device();
    assert_eq!(
        resident.bytes,
        resident.grid.size() + resident.lookup.size() + resident.measurements.size()
    );
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coarse lookup requests"),
        contents: bytemuck::cast_slice(requests),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (requests.len() * size_of::<Probe>()) as u64;
    let buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let output = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let readback = buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let total = resident.bytes + input.size() + output.size() + readback.size();
    assert!(total < 128 << 20);
    let buffers: [&wgpu::Buffer; 5] = [
        &resident.grid,
        &resident.lookup,
        &resident.measurements,
        &input,
        &output,
    ];
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
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
        pass.dispatch_workgroups((requests.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    (
        bytemuck::cast_slice::<u8, Probe>(&readback.get_mapped_range(..)).to_vec(),
        total,
    )
}
