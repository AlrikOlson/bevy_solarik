//! Read exact production std430 publications back through shader arrayLength.
use bevy_platform::sync::Arc;
use bevy_render::{
    render_resource::{ShaderType, StorageBuffer},
    renderer::{RenderDevice, RenderQueue, WgpuWrapper},
};

async fn device() -> (RenderDevice, RenderQueue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
    (
        RenderDevice::from(device),
        RenderQueue(Arc::new(WgpuWrapper::new(queue))),
    )
}

fn read<T: ShaderType + bevy_render::render_resource::encase::internal::WriteInto>(
    device: &RenderDevice,
    queue: &RenderQueue,
    buffer: &StorageBuffer<T>,
) -> Vec<u32> {
    let binding = buffer.binding().unwrap();
    let wgpu::BindingResource::Buffer(ref b) = binding else {
        panic!("buffer")
    };
    let size = b.size.unwrap().get();
    let raw = device.wgpu_device();
    let output = raw.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size + 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = raw.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size + 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let shader = raw.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage,read> input:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let count = arrayLength(&input);
    if id.x == 0u { output[0] = count; }
    if id.x < count { output[id.x + 1u] = input[id.x]; }
}
"#
            .into(),
        ),
    });
    let pipeline = raw.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = raw.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: binding,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = raw.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((size as u32 / 4).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size + 4);
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
    raw.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let bytes = readback.get_mapped_range(..);
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn changed_publication_preserves_exact_large_small_empty_and_return_data() {
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let values: Vec<u32> = (0..50_001).map(|i| i * 97 + 31).collect();
        let mut buffer = StorageBuffer::from(values.clone());
        let first = buffer.write_buffer_changed(&device, &queue);
        assert!(first.allocated);
        assert_eq!(first.bytes, values.len() as u64 * 4);
        let id = buffer.buffer().unwrap().id();
        assert_eq!(read(&device, &queue, &buffer)[1..], values);
        let unchanged = buffer.write_buffer_changed(&device, &queue);
        assert_eq!(
            (unchanged.bytes, unchanged.ranges, unchanged.allocated),
            (0, 0, false)
        );

        buffer.get_mut()[20_000] ^= 0xabcdef;
        buffer.get_mut()[49_999] ^= 0x123456;
        let delta = buffer.write_buffer_changed(&device, &queue);
        assert!(!delta.allocated);
        assert!(delta.bytes < first.bytes);
        assert_eq!(buffer.buffer().unwrap().id(), id);
        assert_eq!(read(&device, &queue, &buffer)[1..], *buffer.get());

        buffer.get_mut().truncate(17);
        buffer.get_mut()[16] = 0x12345678;
        buffer.write_buffer_changed(&device, &queue);
        assert_eq!(buffer.buffer().unwrap().id(), id);
        let small = read(&device, &queue, &buffer);
        assert_eq!(
            small[0], 17,
            "shader sees logical length, not retained capacity"
        );
        assert_eq!(small[1..], *buffer.get());

        buffer.get_mut().clear();
        buffer.write_buffer_changed(&device, &queue);
        let empty = read(&device, &queue, &buffer);
        assert_eq!(
            empty,
            [1, 0],
            "empty runtime array retains only its zero dummy word"
        );
        buffer.set(values.clone());
        buffer.write_buffer_changed(&device, &queue);
        assert_eq!(read(&device, &queue, &buffer)[1..], values);

        buffer.get_mut().resize(100_001, 0x87654321);
        assert!(buffer.write_buffer_changed(&device, &queue).allocated);
        assert_ne!(buffer.buffer().unwrap().id(), id);
        assert_eq!(read(&device, &queue, &buffer)[1..], *buffer.get());
        buffer.set(vec![7, 8, 9]);
        buffer.write_buffer(&device, &queue);
        buffer.set(vec![13, 14, 15]);
        buffer.write_buffer_changed(&device, &queue);
        assert_eq!(read(&device, &queue, &buffer), [3, 13, 14, 15]);
    });
}

#[test]
fn changed_publication_keeps_current_previous_transform_layout() {
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let a = bevy_math::Mat4::from_translation(bevy_math::Vec3::new(2.0, 3.0, 4.0));
        let b = bevy_math::Mat4::from_scale(bevy_math::Vec3::new(1.0, 2.0, 3.0));
        let mut buffer = StorageBuffer::from(vec![a, b]);
        buffer.write_buffer_changed(&device, &queue);
        let mut expected = a.to_cols_array().map(f32::to_bits).to_vec();
        expected.extend(b.to_cols_array().map(f32::to_bits));
        assert_eq!(read(&device, &queue, &buffer)[1..], expected);
        buffer.set(vec![b]);
        buffer.write_buffer_changed(&device, &queue);
        assert_eq!(
            read(&device, &queue, &buffer)[1..],
            b.to_cols_array().map(f32::to_bits)
        );
    });
}
