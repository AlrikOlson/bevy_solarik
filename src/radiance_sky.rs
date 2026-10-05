//! Linear photometric sky fields, composed before atmosphere and temporal effects.
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, Handle, embedded_asset, load_embedded_asset};
use bevy_camera::{CameraMainTextureUsages, Hdr};
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    prepass::{DepthPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_render::{
    RenderApp, RenderStartup,
    diagnostic::RecordDiagnostics,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    texture::GpuImage,
    view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_shader::load_shader_library;

/// Equirectangular RGB cd/m² field; +X at u=0, -Z at u=1/4, +Y at v=0.
/// Linear float texture, no exposure baked in. Alpha ignored. Does not light surfaces.
#[derive(Component, Clone, ExtractComponent)]
#[require(Hdr,DepthPrepass,Msaa::Off,CameraMainTextureUsages=CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct RadianceSky(pub Handle<Image>);
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct RadianceBackground;
pub struct RadianceSkyPlugin;
#[derive(Resource)]
struct Gpu {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}
impl Plugin for RadianceSkyPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "radiance_sky_sample.wgsl");
        embedded_asset!(app, "radiance_sky.wgsl");
        app.add_plugins(ExtractComponentPlugin::<RadianceSky>::default());
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.add_systems(RenderStartup, initialize).add_systems(
                Core3d,
                compose
                    .after(main_opaque_pass_3d)
                    .before(crate::atmosphere::AtmosphereBackground)
                    .in_set(RadianceBackground)
                    .in_set(Core3dSystems::MainPass),
            );
        }
    }
}
fn initialize(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "radiance_sky",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ViewUniform>(true),
                texture_depth_2d(),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("radiance_sky".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(assets.as_ref(), "radiance_sky.wgsl"),
        ..Default::default()
    });
    commands.insert_resource(Gpu { layout, pipeline });
}
fn compose(
    view: ViewQuery<(
        &RadianceSky,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewUniformOffset,
    )>,
    gpu: Option<Res<Gpu>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    uniforms: Res<ViewUniforms>,
    images: Res<RenderAssets<GpuImage>>,
    mut ctx: RenderContext,
) {
    let (sky, target, prepass, offset) = view.into_inner();
    let Some(gpu) = gpu else {
        return;
    };
    let (Some(pipeline), Some(depth), Some(view_binding), Some(image)) = (
        cache.get_compute_pipeline(gpu.pipeline),
        prepass.depth_view(),
        uniforms.uniforms.binding(),
        images.get(&sky.0),
    ) else {
        return;
    };
    let group = device.create_bind_group(
        "radiance_sky",
        &cache.get_bind_group_layout(&gpu.layout),
        &BindGroupEntries::sequential((
            view_binding,
            depth,
            target.main_texture_view(),
            &image.texture_view,
        )),
    );
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("radiance_sky"),
            ..Default::default()
        });
    let span = diagnostics.time_span(&mut pass, "radiance_sky");
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[offset.offset]);
    pass.dispatch_workgroups(
        prepass.size.width.div_ceil(8),
        prepass.size.height.div_ceil(8),
        1,
    );
    span.end(&mut pass);
}
