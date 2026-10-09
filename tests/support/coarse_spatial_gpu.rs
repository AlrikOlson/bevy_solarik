// The test dispatcher uses the production immutable `RenderDevice` upload.
use super::Probe;
use bevy_render::renderer::RenderDevice;
use bevy_solarik::coarse_spatial_scene::{self, PackedSpatial};
use wgpu::util::DeviceExt;
fn execute<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    inputs: &[wgpu::BindGroupEntry<'_>],
    binding: u32,
    count: usize,
) -> Vec<T> {
    let size = (count * size_of::<T>()) as u64;
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
    let mut entries = inputs.to_vec();
    entries.push(wgpu::BindGroupEntry {
        binding,
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
    bytemuck::cast_slice::<u8, T>(&readback.get_mapped_range(..)).to_vec()
}
pub fn run(
    device: &RenderDevice,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    inputs: &[(u32, &[u8])],
    count: usize,
) -> (Vec<Probe>, u64) {
    let packed = inputs.iter().find(|(b, _)| *b == 2).map(|(_, bytes)| {
        PackedSpatial::pack(bytemuck::cast_slice(bytes))
            .unwrap()
            .upload(device)
            .unwrap()
    });
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
    if let Some(packed) = &packed {
        entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: packed.rows.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: packed.hits.as_entire_binding(),
        });
        assert_eq!(packed.bytes, packed.rows.size() + packed.hits.size());
    }
    let bytes = buffers.iter().map(|(_, b)| b.size()).sum::<u64>()
        + packed.as_ref().map_or(0, |p| p.bytes)
        + (count * size_of::<Probe>() * 2) as u64;
    (execute(device, queue, pipeline, &entries, 4, count), bytes)
}
pub fn decode(device: &RenderDevice, queue: &wgpu::Queue, words: &[u32]) -> u64 {
    let source = PackedSpatial::pack(words).unwrap();
    let resident = source.upload(device).unwrap();
    let shared = coarse_spatial_scene::SHADER
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let device = device.wgpu_device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("all spatial sparse words"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                r#"{shared}
@group(0) @binding(0) var<storage,read> rows:array<vec4u>;
@group(0) @binding(1) var<storage,read> hits:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3u) {{
    if id.x>=arrayLength(&output) {{return;}}
    let index=spatial_sample_index(rows[id.x/64u],id.x%64u);
    var word=0u;if index!=SPATIAL_MISSING {{word=hits[index];}}
    output[id.x]=word;
}}
"#
            )
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let result: Vec<u32> = execute(
        device,
        queue,
        &pipeline,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: resident.rows.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: resident.hits.as_entire_binding(),
            },
        ],
        2,
        words.len(),
    );
    assert_eq!(result, words, "lossless GPU spatial decoder");
    resident.bytes
}
