//! Read exact production std430 publications back through shader arrayLength.

#[test]
fn packed_publication_shares_staging_and_preserves_queued_source_lifetimes() {
    use bevy_render::render_resource::StorageBufferUploadBatch;
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let mut a = StorageBuffer::from(vec![11u32; 16_384]);
        let mut b = StorageBuffer::from(vec![u32::MAX; 16_384]);
        a.write_buffer_indices(&device, &queue, &[]);
        b.write_buffer_indices(&device, &queue, &[]);
        let indices: Vec<_> = (0..16_384).step_by(4).collect();
        for generation in 0..8u32 {
            let mut batch = StorageBufferUploadBatch::default();
            for &i in &indices {
                a.get_mut()[i as usize] = generation * 100_000 + i;
                b.get_mut()[i as usize] = generation * 100_000 + i + 1;
            }
            let first = a.stage_buffer_indices(&device, &queue, &indices, &mut batch);
            let second = b.stage_buffer_indices(&device, &queue, &indices, &mut batch);
            assert_eq!(first.ranges + second.ranges, 8192);
            assert_eq!(batch.finish(&device, &queue), (32_768, 1));
        }
        // No CPU/GPU wait between generations: submitted copies must retain
        // their own data, and later writes must win for every destination.
        assert_eq!(read(&device, &queue, &a)[1..], *a.get());
        assert_eq!(read(&device, &queue, &b)[1..], *b.get());
        let mut batch = StorageBufferUploadBatch::default();
        a.stage_buffer_indices(&device, &queue, &indices, &mut batch);
        assert_eq!(batch.finish(&device, &queue), (0, 0));
    });
}

#[test]
fn packed_publication_splits_large_ranges_without_losing_boundaries() {
    use bevy_render::render_resource::{STORAGE_UPLOAD_BATCH_BYTES, StorageBufferUploadBatch};
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let count = STORAGE_UPLOAD_BATCH_BYTES / 4 + 17;
        let mut values = StorageBuffer::from(vec![0u32; count]);
        values.write_buffer_indices(&device, &queue, &[]);
        for (i, value) in values.get_mut().iter_mut().enumerate() {
            *value = i as u32 + 31;
        }
        let indices: Vec<_> = (0..count as u32).collect();
        let mut batch = StorageBufferUploadBatch::default();
        let upload = values.stage_buffer_indices(&device, &queue, &indices, &mut batch);
        assert_eq!(upload.ranges, 1);
        assert_eq!(batch.finish(&device, &queue), (count as u64 * 4, 2));
        assert_eq!(read(&device, &queue, &values)[1..], *values.get());
    });
}

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
#[test]
fn indexed_publication_reads_only_changes_and_preserves_growth_and_transform_bits() {
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let mut buffer = StorageBuffer::from(vec![7u32; 50_001]);
        buffer.write_buffer_indices(&device, &queue, &[0]);
        assert_eq!(
            buffer.cpu_backing_bytes(),
            50_001 * 4,
            "indexed init retains one byte receipt"
        );
        let id = buffer.buffer().unwrap().id();
        buffer.get_mut()[3] = 0;
        buffer.get_mut()[4] = 19;
        buffer.get_mut()[48_000] = 0xabcdef;
        let delta = buffer.write_buffer_indices(&device, &queue, &[48_000, 4, 3, 4]);
        assert_eq!((delta.bytes, delta.ranges, delta.allocated), (12, 2, false));
        assert_eq!(read(&device, &queue, &buffer)[1..], *buffer.get());
        assert_eq!(
            buffer.write_buffer_indices(&device, &queue, &[3, 4]).bytes,
            0
        );
        buffer.get_mut().resize(60_001, 31);
        let indices: Vec<_> = (50_001..60_001).collect();
        assert!(
            !buffer
                .write_buffer_indices(&device, &queue, &indices)
                .allocated
        );
        assert!(buffer.cpu_backing_bytes() <= buffer.buffer().unwrap().size() as usize);
        assert_eq!(read(&device, &queue, &buffer)[1..], *buffer.get());
        buffer.get_mut().truncate(3);
        buffer.write_buffer_indices(&device, &queue, &[]);
        assert_eq!(read(&device, &queue, &buffer), [3, 7, 7, 7]);
        buffer.get_mut().resize(5, 0);
        assert_eq!(
            buffer.write_buffer_indices(&device, &queue, &[3, 4]).bytes,
            8
        );
        assert_eq!(read(&device, &queue, &buffer), [5, 7, 7, 7, 0, 0]);
        buffer.get_mut().clear();
        buffer.write_buffer_indices(&device, &queue, &[0]);
        assert_eq!(read(&device, &queue, &buffer), [1, 0]);
        buffer.get_mut().resize(4, 17);
        buffer.write_buffer_indices(&device, &queue, &[0, 1, 2, 3]);
        assert_eq!(buffer.buffer().unwrap().id(), id);
        assert_eq!(read(&device, &queue, &buffer), [4, 17, 17, 17, 17]);
        buffer.get_mut().resize(100_001, 123);
        assert!(
            buffer
                .write_buffer_indices(&device, &queue, &[100_000])
                .allocated
        );
        assert!(buffer.cpu_backing_bytes() <= buffer.buffer().unwrap().size() as usize);
        assert_eq!(read(&device, &queue, &buffer)[1..], *buffer.get());

        let mut current = bevy_math::Mat4::IDENTITY;
        current.w_axis.x = -0.0;
        let previous = bevy_math::Mat4::from_translation(bevy_math::Vec3::new(3.0, 4.0, 5.0));
        let mut transforms = StorageBuffer::from(vec![current, previous]);
        transforms.write_buffer_indices(&device, &queue, &[0, 1]);
        current.w_axis.x = 0.0;
        transforms.get_mut()[0] = current;
        assert_eq!(
            transforms.write_buffer_indices(&device, &queue, &[0]).bytes,
            64
        );
        let expected: Vec<_> = current
            .to_cols_array()
            .into_iter()
            .chain(previous.to_cols_array())
            .map(f32::to_bits)
            .collect();
        assert_eq!(read(&device, &queue, &transforms)[1..], expected);
    });
}

#[test]
fn active_index_publication_reads_sparse_edits_and_erases_shrunk_or_empty_tails() {
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        let mut buffer = StorageBuffer::from((0..50_001).collect::<Vec<u32>>());
        buffer.write_buffer_indices(&device, &queue, &[]);
        let mut active = buffer.get().clone();
        active[3] = 60_003;
        active[48_000] = 98_000;
        active.truncate(49_999);
        let dirty = buffer.set_indices(&active);
        assert_eq!(dirty, [3, 48_000]);
        let upload = buffer.write_buffer_indices(&device, &queue, &dirty);
        assert_eq!(
            (upload.bytes, upload.ranges, upload.allocated),
            (8, 2, false)
        );
        assert_eq!(read(&device, &queue, &buffer)[1..], active);
        assert!(buffer.set_indices(&active).is_empty());
        let dirty = buffer.set_indices(&[]);
        buffer.write_buffer_indices(&device, &queue, &dirty);
        assert_eq!(read(&device, &queue, &buffer), [1, 0]);
        let dirty = buffer.set_indices(&[48_000, 7]);
        buffer.write_buffer_indices(&device, &queue, &dirty);
        assert_eq!(read(&device, &queue, &buffer), [2, 48_000, 7]);
    });
}

#[test]
fn completed_gpu_retirement_reuses_generations_and_readiness_without_stale_members() {
    bevy_platform::future::block_on(async {
        use bevy_render::scene_slots::SceneSlots;
        let (device, queue) = device().await;
        let mut world = bevy_ecs::world::World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let c = world.spawn_empty().id();
        let mut slots = SceneSlots::default();
        slots.begin();
        let sa = slots.touch(a);
        let sb = slots.touch(b);
        slots.activate(sa);
        let mut active = StorageBuffer::from(slots.active_indices().to_vec());
        active.write_buffer_changed(&device, &queue);
        assert_eq!(read(&device, &queue, &active), [1, sa.index]);
        slots.activate(sb);
        slots.deactivate(sa); // Temporarily unavailable asset: preserve identity.
        active.set(slots.active_indices().to_vec());
        active.write_buffer_changed(&device, &queue);
        assert_eq!(read(&device, &queue, &active), [1, sb.index]);
        slots.activate(sa);
        assert_eq!(slots.touch(a), sa);
        slots.begin();
        slots.touch(b);
        assert_eq!(slots.finish(&queue), [sa]);
        device
            .wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        slots.begin();
        slots.touch(b);
        let sc = slots.touch(c);
        assert_eq!(sc.index, sa.index);
        assert_eq!(sc.generation, sa.generation + 1);
        assert!(!slots.is_active(sa));
        slots.activate(sc);
        slots.finish(&queue);
        active.set(slots.active_indices().to_vec());
        active.write_buffer_changed(&device, &queue);
        assert_eq!(read(&device, &queue, &active)[1..], [sb.index, sc.index]);
        slots.begin(); // No residents: no stale active work may remain.
        slots.finish(&queue);
        active.set(slots.active_indices().to_vec());
        active.write_buffer_changed(&device, &queue);
        assert_eq!(read(&device, &queue, &active), [1, 0]);
    });
}
