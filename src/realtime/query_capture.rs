//! Opt-in, requested-frame records from the original canopy query invocation.
use bevy_ecs::resource::Resource;
use bevy_render::render_resource::binding_types::storage_buffer_sized;
use bevy_render::{render_resource::*, renderer::RenderDevice};

/// Two 256-square crops, with eight vec4 rows per original/retrace query.
pub const QUERY_CAPTURE_BYTES: u64 = 32 + 2 * 256 * 256 * 16 * 16;
pub const LIGHT_CAPTURE_BYTES: u64 = 16 + 32 * 20 * 16;

#[derive(Resource)]
pub struct PrimaryQueryCapture {
    pub buffer: Buffer,
    pub frame: Option<u32>,
    pub direct: bool,
    pub(crate) layout: BindGroupLayoutDescriptor,
    pub(crate) empty_layout: BindGroupLayoutDescriptor,
    pub(crate) group: BindGroup,
    pub(crate) empty_group: BindGroup,
}

impl PrimaryQueryCapture {
    pub(crate) fn new(device: &RenderDevice, cache: &PipelineCache, direct: bool) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("original/retrace primary query evidence"),
            size: if direct {
                LIGHT_CAPTURE_BYTES
            } else {
                QUERY_CAPTURE_BYTES
            },
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = BindGroupLayoutDescriptor::new(
            "primary query capture",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer_sized(false, None),
            ),
        );
        let empty_layout = BindGroupLayoutDescriptor::new("unused RR group", &[]);
        let group = device.create_bind_group(
            "primary query capture",
            &cache.get_bind_group_layout(&layout),
            &BindGroupEntries::single(buffer.as_entire_binding()),
        );
        let empty_group = device.create_bind_group(
            "unused RR group",
            &cache.get_bind_group_layout(&empty_layout),
            &[],
        );
        Self {
            buffer,
            frame: None,
            direct,
            layout,
            empty_layout,
            group,
            empty_group,
        }
    }
}
