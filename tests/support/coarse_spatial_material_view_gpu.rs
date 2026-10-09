include!("coarse_spatial_gpu.rs");
pub fn run_material(
    device: &RenderDevice,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    inputs: &[(u32, &[u8])],
    surfaces: &bevy_render::render_resource::Buffer,
    count: usize,
) -> Vec<Probe> {
    use wgpu::util::DeviceExt;
    let packed = PackedSpatial::pack(bytemuck::cast_slice(
        inputs.iter().find(|(b, _)| *b == 2).unwrap().1,
    ))
    .unwrap()
    .upload(device)
    .unwrap();
    let device = device.wgpu_device();
    let buffers: Vec<_> = inputs
        .iter()
        .filter(|(b, _)| *b != 2)
        .map(|(binding, data)| {
            (
                *binding,
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            )
        })
        .collect();
    let mut entries: Vec<_> = buffers
        .iter()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: b.as_entire_binding(),
        })
        .collect();
    entries.extend([
        wgpu::BindGroupEntry {
            binding: 2,
            resource: packed.rows.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: packed.hits.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: surfaces.as_entire_binding(),
        },
    ]);
    execute(device, queue, pipeline, &entries, 4, count)
}
