//! Measure CPU serialization and GPU-copy costs at production ray-bank scale.
use super::*;
use bevy_math::Mat4;
use bevy_render::render_resource::{StorageBufferUpload, StorageBufferUploadBatch, encase};

fn stage<T: ShaderType + encase::internal::WriteInto + encase::ShaderSize>(
    values: &mut StorageBuffer<Vec<T>>,
    indices: &[u32],
    device: &RenderDevice,
    queue: &RenderQueue,
    batch: &mut StorageBufferUploadBatch,
) -> (StorageBufferUpload, f64) {
    let start = std::time::Instant::now();
    let upload = values.stage_buffer_indices(device, queue, indices, batch);
    (upload, start.elapsed().as_secs_f64() * 1000.0)
}

#[test]
#[ignore = "production-scale publication cost diagnostic"]
fn material_only_updates_do_not_need_to_serialize_unchanged_banks() {
    bevy_platform::future::block_on(async {
        let (device, queue) = device().await;
        const COUNT: usize = 262_144 * 3;
        let mut current = StorageBuffer::from(vec![Mat4::IDENTITY; COUNT]);
        let mut previous = StorageBuffer::from(vec![Mat4::IDENTITY; COUNT]);
        let mut geometry = StorageBuffer::from(vec![[17u32; 6]; COUNT]);
        let mut material = StorageBuffer::from(vec![17u32; COUNT]);
        current.write_buffer_indices(&device, &queue, &[]);
        previous.write_buffer_indices(&device, &queue, &[]);
        geometry.write_buffer_indices(&device, &queue, &[]);
        material.write_buffer_indices(&device, &queue, &[]);
        let indices: Vec<_> = (0..COUNT as u32).collect();
        let mut generation = 0;
        for coupled in [true, false] {
            for _ in 0..3 {
                generation += 1;
                material.get_mut().fill(generation);
                let unchanged: &[u32] = if coupled { &indices } else { &[] };
                let mut batch = StorageBufferUploadBatch::default();
                let results = [
                    stage(&mut current, unchanged, &device, &queue, &mut batch),
                    stage(&mut previous, unchanged, &device, &queue, &mut batch),
                    stage(&mut geometry, unchanged, &device, &queue, &mut batch),
                    stage(&mut material, &indices, &device, &queue, &mut batch),
                ];
                let start = std::time::Instant::now();
                let staged = batch.finish(&device, &queue);
                eprintln!(
                    "material-only: coupled={coupled} bank_ms={:?} finish_ms={:.3}",
                    results.map(|(_, ms)| ms),
                    start.elapsed().as_secs_f64() * 1000.0
                );
                assert!(
                    results[..3]
                        .iter()
                        .all(|(u, _)| u.bytes == 0 && u.ranges == 0)
                );
                assert_eq!(results[3].0.ranges, 1);
                assert_eq!(staged, ((COUNT * 4) as u64, 1));
            }
        }
        let identity = Mat4::IDENTITY.to_cols_array().map(f32::to_bits);
        assert!(
            read(&device, &queue, &current)[1..]
                .chunks_exact(16)
                .all(|m| m == identity)
        );
        assert!(
            read(&device, &queue, &previous)[1..]
                .chunks_exact(16)
                .all(|m| m == identity)
        );
        assert!(
            read(&device, &queue, &geometry)[1..]
                .iter()
                .all(|&x| x == 17)
        );
        assert!(
            read(&device, &queue, &material)[1..]
                .iter()
                .all(|&x| x == generation)
        );
    });
}
