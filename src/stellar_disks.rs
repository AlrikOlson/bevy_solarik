//! Physical finite stellar disks, independent of whether a planet has air.
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, embedded_asset, load_embedded_asset};
use bevy_camera::{CameraMainTextureUsages, Hdr};
use bevy_core_pipeline::{
    prepass::{DepthPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_math::Vec4;
use bevy_render::{
    RenderApp, RenderStartup,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_resource::{
        binding_types::{texture_depth_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};

/// Up to two apparent sources. Direction.xyz is a unit vector, w angular radius
/// in radians. Irradiance.rgb is linear RGB lux at the observer. Zero disables.
/// Direct surface illumination must be supplied separately with matching values.
#[derive(Component, Clone, ExtractComponent, ShaderType)]
#[require(Hdr,DepthPrepass,Msaa::Off,CameraMainTextureUsages=CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct StellarDisks {
    pub direction: [Vec4; 2],
    pub irradiance: [Vec4; 2],
}
pub struct StellarDisksPlugin;
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct StellarBackground;
#[derive(Resource)]
struct Gpu {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}
impl Plugin for StellarDisksPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "stellar_disks.wgsl");
        app.add_plugins(ExtractComponentPlugin::<StellarDisks>::default());
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.add_systems(RenderStartup, initialize).add_systems(
                Core3d,
                compose
                    .in_set(StellarBackground)
                    .after(crate::point_sky::PointBackground)
                    .after(crate::radiance_sky::RadianceBackground)
                    .before(crate::atmosphere::AtmosphereBackground)
                    .in_set(Core3dSystems::MainPass),
            );
        }
    }
}
fn initialize(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "stellar_disks",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ViewUniform>(true),
                texture_depth_2d(),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                uniform_buffer::<StellarDisks>(false),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("stellar_disks".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(assets.as_ref(), "stellar_disks.wgsl"),
        ..Default::default()
    });
    commands.insert_resource(Gpu { layout, pipeline });
}
fn compose(
    view: ViewQuery<(
        &StellarDisks,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewUniformOffset,
    )>,
    gpu: Option<Res<Gpu>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    uniforms: Res<ViewUniforms>,
    mut ctx: RenderContext,
) {
    let (disks, target, prepass, offset) = view.into_inner();
    let Some(gpu) = gpu else {
        return;
    };
    let (Some(pipeline), Some(depth), Some(view_binding)) = (
        cache.get_compute_pipeline(gpu.pipeline),
        prepass.depth_view(),
        uniforms.uniforms.binding(),
    ) else {
        return;
    };
    let mut data = UniformBuffer::from(disks.clone());
    data.write_buffer(&device, &queue);
    let group = device.create_bind_group(
        "stellar_disks",
        &cache.get_bind_group_layout(&gpu.layout),
        &BindGroupEntries::sequential((
            view_binding,
            depth,
            target.main_texture_view(),
            data.binding().unwrap(),
        )),
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("stellar_disks"),
            ..Default::default()
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[offset.offset]);
    pass.dispatch_workgroups(
        prepass.size.width.div_ceil(8),
        prepass.size.height.div_ceil(8),
        1,
    );
}
