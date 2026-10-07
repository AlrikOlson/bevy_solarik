use core::marker::PhantomData;

use super::Buffer;
use crate::{
    render_resource::make_buffer_label,
    renderer::{RenderDevice, RenderQueue},
};
use encase::{
    DynamicStorageBuffer as DynamicStorageBufferWrapper, ShaderType,
    StorageBuffer as StorageBufferWrapper, internal::WriteInto,
};
use wgpu::{BindingResource, BufferBinding, BufferSize, BufferUsages, util::BufferInitDescriptor};

use super::IntoBinding;

/// Stores data to be transferred to the GPU and made accessible to shaders as a storage buffer.
///
/// Storage buffers can be made available to shaders in some combination of read/write mode, and can store large amounts of data.
/// Note however that WebGL2 does not support storage buffers, so consider alternative options in this case.
///
/// Storage buffers can store runtime-sized arrays, but only if they are the last field in a structure.
///
/// The contained data is stored in system RAM. [`write_buffer`](StorageBuffer::write_buffer) queues
/// copying of the data from system RAM to VRAM. Storage buffers must conform to [std430 alignment/padding requirements], which
/// is automatically enforced by this structure.
///
/// Other options for storing GPU-accessible data are:
/// * [`BufferVec`](crate::render_resource::BufferVec)
/// * [`DynamicStorageBuffer`]
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`GpuArrayBuffer`](crate::render_resource::GpuArrayBuffer)
/// * [`RawBufferVec`](crate::render_resource::RawBufferVec)
/// * [`Texture`](crate::render_resource::Texture)
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
///
/// [std430 alignment/padding requirements]: https://www.w3.org/TR/WGSL/#address-spaces-storage
pub struct StorageBuffer<T: ShaderType> {
    value: T,
    scratch: StorageBufferWrapper<Vec<u8>>,
    buffer: Option<Buffer>,
    label: Option<String>,
    changed: bool,
    buffer_usage: BufferUsages,
    last_written_size: Option<BufferSize>,
    last_uploaded: Vec<u8>,
}

/// Actual publication work, independently of reserved GPU capacity.
#[derive(Default, Clone, Copy, Debug)]
pub struct StorageBufferUpload {
    pub bytes: u64,
    pub ranges: u32,
    pub allocated: bool,
}

fn changed_ranges(data: &[u8], previous: &[u8]) -> Vec<core::ops::Range<usize>> {
    const PAGE: usize = 64 * 1024;
    const MAX_RANGES: usize = 128;
    let mut ranges: Vec<core::ops::Range<usize>> = Vec::new();
    for start in (0..data.len()).step_by(PAGE) {
        let end = (start + PAGE).min(data.len());
        if previous.get(start..end) == Some(&data[start..end]) {
            continue;
        }
        if let Some(last) = ranges.last_mut().filter(|r| r.end == start) {
            last.end = end;
        } else {
            ranges.push(start..end);
            if ranges.len() > MAX_RANGES {
                return core::iter::once(0..data.len()).collect();
            }
        }
    }
    ranges
}

impl<T: ShaderType> From<T> for StorageBuffer<T> {
    fn from(value: T) -> Self {
        Self {
            value,
            scratch: StorageBufferWrapper::new(Vec::new()),
            buffer: None,
            label: None,
            changed: false,
            buffer_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            last_written_size: None,
            last_uploaded: Vec::new(),
        }
    }
}

impl<T: ShaderType + Default> Default for StorageBuffer<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            scratch: StorageBufferWrapper::new(Vec::new()),
            buffer: None,
            label: None,
            changed: false,
            buffer_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            last_written_size: None,
            last_uploaded: Vec::new(),
        }
    }
}

impl<T: ShaderType + WriteInto> StorageBuffer<T> {
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer()?,
            offset: 0,
            size: self.last_written_size,
        }))
    }

    pub fn set(&mut self, value: T) {
        self.value = value;
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.changed = true;
        }

        self.label = label;
    }

    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Add more [`BufferUsages`] to the buffer.
    ///
    /// This method only allows addition of flags to the default usage flags.
    ///
    /// The default values for buffer usage are `BufferUsages::COPY_DST` and `BufferUsages::STORAGE`.
    pub fn add_usages(&mut self, usage: BufferUsages) {
        self.buffer_usage |= usage;
        self.changed = true;
    }

    /// Queues writing of data from system RAM to VRAM using the [`RenderDevice`]
    /// and the provided [`RenderQueue`].
    ///
    /// If there is no GPU-side buffer allocated to hold the data currently stored, or if a GPU-side buffer previously
    /// allocated does not have enough capacity, a new GPU-side buffer is created.
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        // A full publication invalidates the optional changed-range receipt.
        self.last_uploaded.clear();
        self.scratch.write(&self.value).unwrap();

        let capacity = self.buffer.as_deref().map(wgpu::Buffer::size).unwrap_or(0);
        let size = self.scratch.as_ref().len() as u64;

        if capacity < size || self.changed {
            self.buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
                label: make_buffer_label::<Self>(&self.label),
                usage: self.buffer_usage,
                contents: self.scratch.as_ref(),
            }));
            self.changed = false;
        } else if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, self.scratch.as_ref());
        }

        self.last_written_size = BufferSize::new(size);
    }

    /// Publish exact changed std430 bytes while retaining the allocation.
    ///
    /// Logical binding size follows the current value, including after shrink.
    /// At most 128 contiguous 64 KiB page ranges are written; more fragmented
    /// changes fall back to one full write. This changes no data or precision.
    pub fn write_buffer_changed(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> StorageBufferUpload {
        self.scratch.as_mut().clear();
        self.scratch.write(&self.value).unwrap();
        let data = self.scratch.as_ref();
        let size = data.len() as u64;
        let capacity = self.buffer.as_deref().map(wgpu::Buffer::size).unwrap_or(0);
        let allocated = capacity < size || self.changed || self.buffer.is_none();
        let ranges = if allocated {
            let allocation_size = size.max(4).next_multiple_of(64 * 1024);
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: make_buffer_label::<Self>(&self.label),
                size: allocation_size,
                usage: self.buffer_usage,
                mapped_at_creation: false,
            }));
            self.changed = false;
            core::iter::once(0..data.len()).collect()
        } else {
            changed_ranges(data, &self.last_uploaded)
        };
        let mut upload = StorageBufferUpload {
            allocated,
            ranges: ranges.len() as u32,
            ..Default::default()
        };
        self.last_uploaded.resize(data.len(), 0);
        for range in ranges {
            if !range.is_empty() {
                queue.write_buffer(
                    self.buffer.as_ref().unwrap(),
                    range.start as u64,
                    &data[range.clone()],
                );
                self.last_uploaded[range.clone()].copy_from_slice(&data[range.clone()]);
                upload.bytes += range.len() as u64;
            }
        }
        self.last_written_size = BufferSize::new(size);
        upload
    }
}

impl<'a, T: ShaderType + WriteInto> IntoBinding<'a> for &'a StorageBuffer<T> {
    #[inline]
    fn into_binding(self) -> BindingResource<'a> {
        self.binding().expect("Failed to get buffer")
    }
}

/// Stores data to be transferred to the GPU and made accessible to shaders as a dynamic storage buffer.
///
/// This is just a [`StorageBuffer`], but also allows you to set dynamic offsets.
///
/// Dynamic storage buffers can be made available to shaders in some combination of read/write mode, and can store large amounts
/// of data. Note however that WebGL2 does not support storage buffers, so consider alternative options in this case. Dynamic
/// storage buffers support multiple separate bindings at dynamic byte offsets and so have a
/// [`push`](DynamicStorageBuffer::push) method.
///
/// The contained data is stored in system RAM. [`write_buffer`](DynamicStorageBuffer::write_buffer)
/// queues copying of the data from system RAM to VRAM. The data within a storage buffer binding must conform to
/// [std430 alignment/padding requirements]. `DynamicStorageBuffer` takes care of serializing the inner type to conform to
/// these requirements. Each item [`push`](DynamicStorageBuffer::push)ed into this structure
/// will additionally be aligned to meet dynamic offset alignment requirements.
///
/// Other options for storing GPU-accessible data are:
/// * [`BufferVec`](crate::render_resource::BufferVec)
/// * [`DynamicUniformBuffer`](crate::render_resource::DynamicUniformBuffer)
/// * [`GpuArrayBuffer`](crate::render_resource::GpuArrayBuffer)
/// * [`RawBufferVec`](crate::render_resource::RawBufferVec)
/// * [`StorageBuffer`]
/// * [`Texture`](crate::render_resource::Texture)
/// * [`UniformBuffer`](crate::render_resource::UniformBuffer)
///
/// [std430 alignment/padding requirements]: https://www.w3.org/TR/WGSL/#address-spaces-storage
pub struct DynamicStorageBuffer<T: ShaderType> {
    scratch: DynamicStorageBufferWrapper<Vec<u8>>,
    buffer: Option<Buffer>,
    label: Option<String>,
    changed: bool,
    buffer_usage: BufferUsages,
    last_written_size: Option<BufferSize>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: ShaderType> Default for DynamicStorageBuffer<T> {
    fn default() -> Self {
        Self {
            scratch: DynamicStorageBufferWrapper::new(Vec::new()),
            buffer: None,
            label: None,
            changed: false,
            buffer_usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            last_written_size: None,
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType + WriteInto> DynamicStorageBuffer<T> {
    #[inline]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    #[inline]
    pub fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer()?,
            offset: 0,
            size: self.last_written_size,
        }))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scratch.as_ref().is_empty()
    }

    #[inline]
    pub fn push(&mut self, value: T) -> u32 {
        self.scratch.write(&value).unwrap() as u32
    }

    pub fn set_label(&mut self, label: Option<&str>) {
        let label = label.map(str::to_string);

        if label != self.label {
            self.changed = true;
        }

        self.label = label;
    }

    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Add more [`BufferUsages`] to the buffer.
    ///
    /// This method only allows addition of flags to the default usage flags.
    ///
    /// The default values for buffer usage are `BufferUsages::COPY_DST` and `BufferUsages::STORAGE`.
    pub fn add_usages(&mut self, usage: BufferUsages) {
        self.buffer_usage |= usage;
        self.changed = true;
    }

    #[inline]
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        let capacity = self.buffer.as_deref().map(wgpu::Buffer::size).unwrap_or(0);
        let size = self.scratch.as_ref().len() as u64;

        if capacity < size || (self.changed && size > 0) {
            self.buffer = Some(device.create_buffer_with_data(&BufferInitDescriptor {
                label: make_buffer_label::<Self>(&self.label),
                usage: self.buffer_usage,
                contents: self.scratch.as_ref(),
            }));
            self.changed = false;
        } else if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, self.scratch.as_ref());
        }

        self.last_written_size = BufferSize::new(size);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.scratch.as_mut().clear();
        self.scratch.set_offset(0);
    }
}

impl<'a, T: ShaderType + WriteInto> IntoBinding<'a> for &'a DynamicStorageBuffer<T> {
    #[inline]
    fn into_binding(self) -> BindingResource<'a> {
        self.binding().expect("Failed to get buffer")
    }
}
