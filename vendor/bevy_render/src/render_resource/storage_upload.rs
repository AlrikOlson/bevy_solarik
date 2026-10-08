//! Bounded, destination-indexed copies for sparse storage publication.
use super::{Buffer, BufferInitDescriptor, BufferUsages};
use crate::renderer::{RenderDevice, RenderQueue};

/// Maximum CPU packing arena and individual staging allocation.
/// Large publications split at aligned boundaries; queue ownership keeps each
/// submitted source alive until every copy has finished.
pub const STORAGE_UPLOAD_BATCH_BYTES: usize = 4 * 1024 * 1024;

struct Copy {
    target: Buffer,
    destination: u64,
    source: u64,
    size: u64,
}

/// Packs many exact dirty ranges into one staging allocation and submission.
/// Call `finish` before scheduling consumers of the destination buffers.
#[derive(Default)]
#[must_use = "finish the batch before using its destination buffers"]
pub struct StorageBufferUploadBatch {
    bytes: Vec<u8>,
    copies: Vec<Copy>,
    /// Submitted staging bytes, distinct from persistent GPU storage capacity.
    pub staged_bytes: u64,
    /// Actual source allocations/submissions, not the number of dirty ranges.
    pub staging_buffers: u32,
}

impl StorageBufferUploadBatch {
    pub(crate) fn push(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        target: &Buffer,
        mut destination: u64,
        mut data: &[u8],
    ) {
        assert_eq!(destination % wgpu::COPY_BUFFER_ALIGNMENT, 0);
        assert_eq!(data.len() as u64 % wgpu::COPY_BUFFER_ALIGNMENT, 0);
        while !data.is_empty() {
            let size = data
                .len()
                .min(STORAGE_UPLOAD_BATCH_BYTES - self.bytes.len());
            self.copies.push(Copy {
                target: target.clone(),
                destination,
                source: self.bytes.len() as u64,
                size: size as u64,
            });
            let required = self.bytes.len() + size;
            if self.bytes.capacity() < required {
                self.bytes.reserve_exact(
                    required.next_power_of_two().min(STORAGE_UPLOAD_BATCH_BYTES) - self.bytes.len(),
                );
            }
            self.bytes.extend_from_slice(&data[..size]);
            destination += size as u64;
            data = &data[size..];
            if self.bytes.len() == STORAGE_UPLOAD_BATCH_BYTES {
                self.flush(device, queue);
            }
        }
    }

    /// Submit remaining copies. wgpu retains staging and targets across the
    /// submission; neither a frame count nor a CPU receipt permits early reuse.
    pub fn finish(mut self, device: &RenderDevice, queue: &RenderQueue) -> (u64, u32) {
        self.flush(device, queue);
        (self.staged_bytes, self.staging_buffers)
    }

    fn flush(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.bytes.is_empty() {
            return;
        }
        let staging = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("storage_sparse_upload"),
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            contents: &self.bytes,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        for copy in self.copies.drain(..) {
            encoder.copy_buffer_to_buffer(
                &staging,
                copy.source,
                &copy.target,
                copy.destination,
                copy.size,
            );
        }
        queue.submit([encoder.finish()]);
        self.staged_bytes += self.bytes.len() as u64;
        self.staging_buffers += 1;
        self.bytes.clear();
    }
}
