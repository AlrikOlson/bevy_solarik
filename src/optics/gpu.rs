use super::{CameraOptics, OpticsReadback};
use alloc::sync::Arc;
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, embedded_asset, load_embedded_asset};
use bevy_core_pipeline::{
    schedule::{Core3d, Core3dSystems},
    tonemapping::tonemapping,
};
use bevy_ecs::prelude::*;
use bevy_math::Vec4;
use bevy_platform::collections::HashMap;
use bevy_render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    extract_component::ExtractComponentPlugin,
    render_resource::{
        binding_types::{storage_buffer_sized, texture_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::{ExtractedView, ViewTarget},
};
use core::num::NonZeroU64;
use core::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub struct CameraOpticsPlugin;
impl Plugin for CameraOpticsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "camera.wgsl");
        app.add_plugins((
            ExtractComponentPlugin::<CameraOptics>::default(),
            ExtractComponentPlugin::<OpticsReadback>::default(),
        ));
        app.add_plugins(super::background::BackgroundPlugin);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<Views>()
                .add_systems(RenderStartup, initialize)
                .add_systems(
                    Core3d,
                    compose
                        .in_set(Core3dSystems::PostProcess)
                        .before(bevy_post_process::bloom::bloom)
                        .before(tonemapping),
                )
                .add_systems(Render, readback.in_set(RenderSystems::Cleanup));
        }
    }
}
#[derive(Clone, Copy, Default, ShaderType)]
struct Parameters {
    meter: Vec4,
    lens: Vec4,
    bounds: Vec4,
    frame: Vec4,
}
struct Stage {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}
#[derive(Resource)]
struct Pipelines([Stage; 3]);
struct ViewState {
    scratch: TextureView,
    width: u32,
    height: u32,
    state: Buffer,
    parameters: UniformBuffer<Parameters>,
    last: Instant,
    reset: u32,
    initial: bool,
    pending: Arc<AtomicBool>,
}
#[derive(Resource, Default)]
struct Views(HashMap<Entity, ViewState>);
fn initialize(mut commands: Commands, server: Res<AssetServer>, cache: Res<PipelineCache>) {
    let read = || texture_2d(TextureSampleType::Float { filterable: false });
    let output = || texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly);
    let layouts = [
        BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                read(),
                uniform_buffer::<Parameters>(false),
                storage_buffer_sized(false, NonZeroU64::new(48)),
            ),
        )
        .to_vec(),
        BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (read(), uniform_buffer::<Parameters>(false), output()),
        )
        .to_vec(),
        BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                read(),
                read(),
                uniform_buffer::<Parameters>(false),
                storage_buffer_sized(false, NonZeroU64::new(48)),
                output(),
            ),
        )
        .to_vec(),
    ];
    let stages = layouts
        .into_iter()
        .enumerate()
        .map(|(i, entries)| {
            let layout = BindGroupLayoutDescriptor::new("camera optics", &entries);
            let defs = match i {
                0 => vec!["METER".into()],
                1 => vec!["HORIZONTAL".into()],
                _ => vec![],
            };
            let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("camera optics".into()),
                layout: vec![layout.clone()],
                shader: load_embedded_asset!(&*server, "camera.wgsl"),
                shader_defs: defs,
                entry_point: Some(if i == 0 { "meter" } else { "scatter" }.into()),
                ..Default::default()
            });
            Stage { layout, pipeline }
        })
        .collect::<Vec<_>>();
    commands.insert_resource(Pipelines(
        stages.try_into().ok().expect("three optical passes"),
    ));
}
fn allocate(device: &RenderDevice, width: u32, height: u32, reset: u32) -> ViewState {
    let scratch = device
        .create_texture(&TextureDescriptor {
            label: Some("optical scattering scratch"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default());
    ViewState {
        scratch,
        width,
        height,
        state: device.create_buffer(&BufferDescriptor {
            label: Some("photometric meter"),
            size: 48,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        parameters: UniformBuffer::default(),
        last: Instant::now(),
        reset,
        initial: true,
        pending: Arc::new(AtomicBool::new(false)),
    }
}
fn compose(
    view: ViewQuery<(&CameraOptics, &ExtractedCamera, &ExtractedView, &ViewTarget)>,
    mut views: ResMut<Views>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    let id = view.entity();
    let (settings, camera, extracted, target) = view.into_inner();
    if settings.validate().is_err() || !camera.hdr || camera.viewport.is_some() {
        return;
    }
    let Some(size) = camera.physical_target_size else {
        return;
    };
    let sigma =
        settings.scatter_sigma * extracted.clip_from_view.y_axis.y.abs() * size.y as f32 * 0.5;
    if sigma > 32.0 || extracted.clip_from_view.w_axis.w != 0.0 {
        return;
    }
    let ready: Vec<_> = pipelines
        .0
        .iter()
        .filter_map(|s| cache.get_compute_pipeline(s.pipeline))
        .collect();
    if ready.len() != 3 {
        return;
    }
    let state = views
        .0
        .entry(id)
        .or_insert_with(|| allocate(&device, size.x, size.y, settings.reset));
    if state.width != size.x || state.height != size.y {
        *state = allocate(&device, size.x, size.y, settings.reset);
    }
    let now = Instant::now();
    let dt = now.duration_since(state.last).as_secs_f32();
    state.last = now;
    let manual = -(camera.exposure * 1.2).log2();
    let reset = state.initial || state.reset != settings.reset;
    state.initial = false;
    state.reset = settings.reset;
    state.parameters.set(Parameters {
        meter: Vec4::new(
            f32::from(settings.automatic),
            manual,
            settings.compensation,
            dt,
        ),
        lens: Vec4::new(
            sigma,
            settings.scatter_fraction,
            settings.brighten_speed,
            settings.darken_speed,
        ),
        bounds: Vec4::new(settings.min_ev, settings.max_ev, 0.0, 0.0),
        frame: Vec4::new(camera.exposure, f32::from(reset), 0.0, 0.0),
    });
    state.parameters.write_buffer(&device, &queue);
    let write = target.post_process_write();
    let uniform = state
        .parameters
        .binding()
        .expect("uploaded optics parameters");
    let groups = [
        device.create_bind_group(
            "meter",
            &cache.get_bind_group_layout(&pipelines.0[0].layout),
            &BindGroupEntries::sequential((
                write.source,
                uniform.clone(),
                state.state.as_entire_buffer_binding(),
            )),
        ),
        device.create_bind_group(
            "scatter horizontal",
            &cache.get_bind_group_layout(&pipelines.0[1].layout),
            &BindGroupEntries::sequential((write.source, uniform.clone(), &state.scratch)),
        ),
        device.create_bind_group(
            "scatter vertical",
            &cache.get_bind_group_layout(&pipelines.0[2].layout),
            &BindGroupEntries::sequential((
                write.source,
                &state.scratch,
                uniform,
                state.state.as_entire_buffer_binding(),
                write.destination,
            )),
        ),
    ];
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "camera/optics");
    for i in 0..3 {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("camera optics"),
                timestamp_writes: None,
            });
        pass.set_pipeline(ready[i]);
        pass.set_bind_group(0, &groups[i], &[]);
        if i == 0 {
            pass.dispatch_workgroups(1, 1, 1);
        } else {
            pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
        }
    }
    span.end(ctx.command_encoder());
}
fn readback(
    mut views: ResMut<Views>,
    cameras: Query<(&CameraOptics, Option<&OpticsReadback>)>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // Headless runners do not guarantee a surface/device poll each frame.
    // Drive map callbacks without waiting for GPU completion.
    let _ = device.poll(PollType::Poll);
    views.0.retain(|entity, _| cameras.contains(*entity));
    for (entity, state) in &views.0 {
        let Ok((_, Some(probe))) = cameras.get(*entity) else {
            continue;
        };
        if state.pending.swap(true, Ordering::AcqRel) {
            continue;
        }
        let buffer = Arc::new(device.create_buffer(&BufferDescriptor {
            label: Some("camera meter readback"),
            size: 48,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&state.state, 0, &buffer, 0, 48);
        queue.submit([encoder.finish()]);
        let destination = probe.0.clone();
        let pending = state.pending.clone();
        let mapped = buffer.clone();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let bytes = mapped.slice(..).get_mapped_range();
                *destination.lock().expect("camera meter snapshot") =
                    Some(*bytemuck::from_bytes::<[f32; 12]>(&bytes));
                drop(bytes);
                mapped.unmap();
            }
            pending.store(false, Ordering::Release);
        });
    }
}
