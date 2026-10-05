//! Preserve physical background radiance across geometry-guided display effects.
use super::CameraOptics;
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, embedded_asset, load_embedded_asset};
use bevy_core_pipeline::{
    prepass::ViewPrepassTextures,
    schedule::{Core3d, Core3dSystems},
    tonemapping::tonemapping,
};
use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::{
        binding_types::{texture_2d, texture_depth_2d, texture_storage_2d},
        *,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::ViewTarget,
};

/// Put an external display-space neural pass in `ExternalDisplayEffects`.
/// It still executes for the whole image. Only guide-free background pixels
/// are restored; rasterized surfaces retain the external renderer's result.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CameraOpticsSystems {
    SaveDisplay,
    ExternalDisplayEffects,
    RestoreBackground,
}
pub(super) struct BackgroundPlugin;
impl Plugin for BackgroundPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "background.wgsl");
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<References>()
                .add_systems(RenderStartup, initialize)
                .configure_sets(
                    Core3d,
                    (
                        CameraOpticsSystems::SaveDisplay,
                        CameraOpticsSystems::ExternalDisplayEffects,
                        CameraOpticsSystems::RestoreBackground,
                    )
                        .chain()
                        .after(tonemapping),
                )
                .add_systems(
                    Core3d,
                    (
                        save.in_set(CameraOpticsSystems::SaveDisplay)
                            .in_set(Core3dSystems::PostProcess),
                        restore
                            .in_set(CameraOpticsSystems::RestoreBackground)
                            .in_set(Core3dSystems::PostProcess),
                    ),
                )
                .add_systems(Render, cleanup.in_set(RenderSystems::Cleanup));
        }
    }
}
struct Reference {
    view: TextureView,
    width: u32,
    height: u32,
}
#[derive(Resource, Default)]
struct References(HashMap<Entity, Reference>);
#[derive(Resource)]
struct Pipelines {
    layout: BindGroupLayoutDescriptor,
    save: CachedComputePipelineId,
    restore: CachedComputePipelineId,
}
fn initialize(mut commands: Commands, server: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "radiometric background",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_depth_2d(),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let pipeline = |entry: &str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("radiometric background".into()),
            layout: vec![layout.clone()],
            shader: load_embedded_asset!(&*server, "background.wgsl"),
            entry_point: Some(entry.to_owned().into()),
            ..Default::default()
        })
    };
    commands.insert_resource(Pipelines {
        save: pipeline("save"),
        restore: pipeline("restore"),
        layout,
    });
}
fn save(
    view: ViewQuery<(
        &CameraOptics,
        &ViewTarget,
        &ViewPrepassTextures,
        &bevy_render::camera::ExtractedCamera,
    )>,
    mut references: ResMut<References>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let entity = view.entity();
    let (settings, target, prepass, camera) = view.into_inner();
    if !camera.hdr || camera.viewport.is_some() || settings.validate().is_err() {
        return;
    }
    let (Some(pipeline), Some(depth)) =
        (cache.get_compute_pipeline(pipelines.save), &prepass.depth)
    else {
        return;
    };
    let size = target.main_texture().size();
    let reference = references
        .0
        .entry(entity)
        .or_insert_with(|| allocate(&device, size.width, size.height));
    if (reference.width, reference.height) != (size.width, size.height) {
        *reference = allocate(&device, size.width, size.height);
    }
    // Unused reference binding points at source, avoiding aliasing the storage output.
    let group = device.create_bind_group(
        "save physical background",
        &cache.get_bind_group_layout(&pipelines.layout),
        &BindGroupEntries::sequential((
            target.main_texture_view(),
            target.main_texture_view(),
            &depth.texture.default_view,
            &reference.view,
        )),
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
}
fn restore(
    view: ViewQuery<(
        &CameraOptics,
        &ViewTarget,
        &ViewPrepassTextures,
        &bevy_render::camera::ExtractedCamera,
    )>,
    references: Res<References>,
    pipelines: Res<Pipelines>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let entity = view.entity();
    let (settings, target, prepass, camera) = view.into_inner();
    if !camera.hdr || camera.viewport.is_some() || settings.validate().is_err() {
        return;
    }
    let (Some(pipeline), Some(depth), Some(reference)) = (
        cache.get_compute_pipeline(pipelines.restore),
        &prepass.depth,
        references.0.get(&entity),
    ) else {
        return;
    };
    let size = target.main_texture().size();
    if (reference.width, reference.height) != (size.width, size.height) {
        return;
    }
    let output = target.post_process_write();
    let group = device.create_bind_group(
        "restore physical background",
        &cache.get_bind_group_layout(&pipelines.layout),
        &BindGroupEntries::sequential((
            output.source,
            &reference.view,
            &depth.texture.default_view,
            output.destination,
        )),
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
}
fn allocate(device: &RenderDevice, width: u32, height: u32) -> Reference {
    let view = device
        .create_texture(&TextureDescriptor {
            label: Some("physical display reference"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    Reference {
        view,
        width,
        height,
    }
}
fn cleanup(mut references: ResMut<References>, views: Query<(), With<CameraOptics>>) {
    references.0.retain(|entity, _| views.contains(*entity));
}
