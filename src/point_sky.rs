//! Flux-conserving catalogue points. Shared arbitrary length unit; flux is lux at one unit.
use alloc::sync::Arc;
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, embedded_asset, load_embedded_asset};
use bevy_camera::{CameraMainTextureUsages, Hdr};
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    prepass::{DepthPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_math::{Vec3, Vec4};
use bevy_platform::collections::HashMap;
use bevy_render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    diagnostic::RecordDiagnostics,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_resource::{
        binding_types::{
            storage_buffer, storage_buffer_read_only, texture_depth_2d, texture_storage_2d,
            uniform_buffer,
        },
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
/// Position xyz in a shared length unit; w is illuminance at one such unit (lux).
/// Colour xyz is linear RGB normalized to unit photopic Y. Set w=0 for a
/// positioned point, or w=1 for a direction-only measurement with unknown distance.
/// Direction-only points ignore observer translation; their position length still
/// defines the unit at which position.w is measured. Inputs must be finite and positive.
#[derive(Clone, Copy, ShaderType)]
pub struct PhotometricPoint {
    pub position: Vec4,
    pub colour: Vec4,
}
/// Point positions and observer must use the same physical length unit.
/// A camera-only sky: no secondary-ray illumination or stellar surface geometry.
#[derive(Component, Clone, ExtractComponent)]
#[require(Hdr,DepthPrepass,Msaa::Off,CameraMainTextureUsages=CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct PointSky {
    pub points: Arc<Vec<PhotometricPoint>>,
    pub observer: Vec3,
}
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointBackground;
pub struct PointSkyPlugin;
struct Allocation {
    size: (u32, u32),
    source: usize,
    points: StorageBuffer<Vec<PhotometricPoint>>,
    pixels: Buffer,
    observer: UniformBuffer<Vec4>,
}
#[derive(Resource)]
struct Gpu {
    layout: BindGroupLayoutDescriptor,
    scatter: CachedComputePipelineId,
    compose: CachedComputePipelineId,
    allocations: HashMap<Entity, Allocation>,
}
impl Plugin for PointSkyPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "point_sky.wgsl");
        app.add_plugins(ExtractComponentPlugin::<PointSky>::default());
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .add_systems(RenderStartup, initialize)
                .add_systems(Render, cleanup.in_set(RenderSystems::PrepareResources))
                .add_systems(
                    Core3d,
                    compose
                        .after(main_opaque_pass_3d)
                        .after(crate::radiance_sky::RadianceBackground)
                        .before(crate::atmosphere::AtmosphereBackground)
                        .in_set(PointBackground)
                        .in_set(Core3dSystems::MainPass),
                );
        }
    }
}
fn initialize(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "point_sky",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ViewUniform>(true),
                texture_depth_2d(),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                storage_buffer_read_only::<Vec<PhotometricPoint>>(false),
                storage_buffer::<Vec<u32>>(false),
                uniform_buffer::<Vec4>(false),
            ),
        ),
    );
    let make = |entry: &'static str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("point_sky_{entry}").into()),
            layout: vec![layout.clone()],
            shader: load_embedded_asset!(assets.as_ref(), "point_sky.wgsl"),
            entry_point: Some(entry.into()),
            ..Default::default()
        })
    };
    commands.insert_resource(Gpu {
        scatter: make("scatter"),
        compose: make("composite"),
        layout,
        allocations: Default::default(),
    });
}
fn cleanup(mut gpu: ResMut<Gpu>, views: Query<Entity, With<PointSky>>) {
    gpu.allocations.retain(|e, _| views.contains(*e));
}
fn compose(
    view: ViewQuery<(
        Entity,
        &PointSky,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewUniformOffset,
    )>,
    gpu: Option<ResMut<Gpu>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    uniforms: Res<ViewUniforms>,
    mut ctx: RenderContext,
) {
    let (entity, sky, target, prepass, offset) = view.into_inner();
    if sky.points.is_empty() {
        return;
    }
    let Some(mut gpu) = gpu else {
        return;
    };
    let (Some(scatter), Some(composite), Some(depth), Some(view_binding)) = (
        cache.get_compute_pipeline(gpu.scatter),
        cache.get_compute_pipeline(gpu.compose),
        prepass.depth_view(),
        uniforms.uniforms.binding(),
    ) else {
        return;
    };
    let layout = cache.get_bind_group_layout(&gpu.layout);
    let size = (prepass.size.width, prepass.size.height);
    let source = Arc::as_ptr(&sky.points) as usize;
    let allocate = || {
        let mut points = StorageBuffer::from(sky.points.as_ref().clone());
        points.write_buffer(&device, &queue);
        let pixels = device.create_buffer(&BufferDescriptor {
            label: Some("point_sky_accumulator"),
            size: u64::from(size.0) * u64::from(size.1) * 12,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut observer = UniformBuffer::from(sky.observer.extend(sky.points.len() as f32));
        observer.write_buffer(&device, &queue);
        Allocation {
            size,
            source,
            points,
            pixels,
            observer,
        }
    };
    let entry = gpu.allocations.entry(entity).or_insert_with(allocate);
    if entry.size != size || entry.source != source {
        *entry = allocate();
    }
    *entry.observer.get_mut() = sky.observer.extend(sky.points.len() as f32);
    entry.observer.write_buffer(&device, &queue);
    let group = device.create_bind_group(
        "point_sky",
        &layout,
        &BindGroupEntries::sequential((
            view_binding,
            depth,
            target.main_texture_view(),
            entry.points.binding().unwrap(),
            entry.pixels.as_entire_binding(),
            entry.observer.binding().unwrap(),
        )),
    );
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    ctx.command_encoder().clear_buffer(&entry.pixels, 0, None);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("point_sky"),
            ..Default::default()
        });
    let span = diagnostics.time_span(&mut pass, "point_sky");
    pass.set_bind_group(0, &group, &[offset.offset]);
    pass.set_pipeline(scatter);
    pass.dispatch_workgroups((sky.points.len() as u32).div_ceil(64), 1, 1);
    pass.set_pipeline(composite);
    pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
    span.end(&mut pass);
}
