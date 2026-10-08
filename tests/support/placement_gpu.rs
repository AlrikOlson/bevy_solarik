use super::Probe;
use bevy_solarik::scene::placement::{GpuPlacementPage, SHADER};
use wgpu::util::DeviceExt;

pub async fn device() -> (wgpu::Device, wgpu::Queue, wgpu::AdapterInfo) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&Default::default())
        .await
        .expect("Vulkan adapter");
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&Default::default())
        .await
        .expect("Vulkan device");
    (device, queue, info)
}

pub fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    words: &[u32],
    pages: &[GpuPlacementPage],
    radii: &[f32],
    stride: usize,
) -> Vec<Probe> {
    let entry = if stride == 3 { "probe12" } else { "probe16" };
    let source = format!(
        "{SHADER}\n{}",
        include_str!("../scene_placement_fixture.wgsl")
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("packed placement contract"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let inputs = [
        bytemuck::cast_slice(words),
        bytemuck::cast_slice(pages),
        bytemuck::cast_slice(radii),
    ]
    .map(|bytes| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("placement fixture inputs"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    dispatch(device, queue, &pipeline, &inputs, pages.len())
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    inputs: &[wgpu::Buffer; 3],
    count: usize,
) -> Vec<Probe> {
    let output = buffer(
        device,
        count,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        device,
        count,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let buffers = [&inputs[0], &inputs[1], &inputs[2], &output];
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
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice::<u8, Probe>(&readback.get_mapped_range(..)).to_vec()
}

fn buffer(device: &wgpu::Device, count: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("placement numerical probe"),
        size: (count * size_of::<Probe>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}
