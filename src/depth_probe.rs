//! Opt-in native depth-row readback for geometric acceptance, independent of exposure.
use alloc::sync::Arc;
use bevy_app::{App, Plugin};
use bevy_ecs::{
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Query, Res},
};
use bevy_render::{
    Render, RenderApp, RenderSystems,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{
        BufferDescriptor, BufferUsages, Extent3d, MapMode, Origin3d, TexelCopyBufferInfo,
        TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    },
    renderer::{RenderDevice, RenderQueue},
    view::ViewDepthTexture,
};
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct DepthRow {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct DepthProbe {
    /// Caller changes this on a camera transition to reject stale data.
    pub ticket: u64,
    pub snapshot: Arc<Mutex<Option<DepthRow>>>,
    pending: Arc<AtomicBool>,
}
/// Install only in a single-view fixture. Camera3d depth usage needs `COPY_SRC`;
/// MSAA must be off. Reads the centre row, with no image or lighting changes.
/// Depth-copy API requires staging the whole texture.
pub struct DepthProbePlugin;
impl Plugin for DepthProbePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DepthProbe>();
        app.add_plugins(ExtractResourcePlugin::<DepthProbe>::default());
        app.sub_app_mut(RenderApp)
            .add_systems(Render, read_row.in_set(RenderSystems::Cleanup));
    }
}
fn read_row(
    views: Query<&ViewDepthTexture>,
    probe: Res<DepthProbe>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Ok(depth) = views.single() else {
        return;
    };
    if probe.pending.swap(true, Ordering::AcqRel) {
        return;
    }
    let (width, height) = (depth.texture.width(), depth.texture.height());
    let bytes = (width * 4).div_ceil(256) * 256;
    let buffer = Arc::new(device.create_buffer(&BufferDescriptor {
        label: Some("native depth centre row"),
        size: u64::from(bytes) * u64::from(height),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &depth.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::DepthOnly,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let destination = probe.snapshot.clone();
    let pending = probe.pending.clone();
    let ticket = probe.ticket;
    let mapped = buffer.clone();
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        if result.is_ok() {
            let data = mapped.slice(..).get_mapped_range();
            let start = (height / 2 * bytes) as usize;
            let values =
                bytemuck::cast_slice::<u8, f32>(&data[start..start + width as usize * 4]).to_vec();
            *destination.lock().expect("depth snapshot") = Some(DepthRow {
                ticket,
                width,
                height,
                values,
            });
            drop(data);
            mapped.unmap();
        }
        pending.store(false, Ordering::Release);
    });
}
