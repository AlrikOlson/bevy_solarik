//! Finite axisymmetric particulate rings: physical opacity and once-scattered light.
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
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{
        binding_types::{
            storage_buffer_read_only, texture_depth_2d, texture_storage_2d, uniform_buffer,
        },
        *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_shader::load_shader_library;
#[derive(Clone, Default, PartialEq, ShaderType)]
pub struct RingParameters {
    /// xyz centre relative to render origin (shadow) or camera (view); w half thickness, m.
    pub centre: Vec4,
    /// xyz unit normal, w inner radius, m.
    pub normal: Vec4,
    /// xyz unit axis in the ring plane, w outer radius, m.
    pub axis: Vec4,
    /// xyz direction towards star, w stellar angular radius, radians.
    pub sun: Vec4,
    /// RGB incident illuminance; w enables planet-on-ring shadows.
    pub irradiance: Vec4,
    /// xyz ellipsoid radii in ring coordinates, w first opacity sample radius.
    pub planet: Vec4,
    /// x radial sample spacing; z enables ring-on-surface shadows.
    pub profile: Vec4,
}
#[derive(Clone, PartialEq, ShaderType)]
pub struct RingData {
    pub p: RingParameters,
    /// Normal optical depth, particle albedo, HG g, unused; uniform radial bins.
    #[shader(size(runtime))]
    pub samples: Vec<Vec4>,
}
impl Default for RingData {
    fn default() -> Self {
        Self {
            p: Default::default(),
            samples: vec![Vec4::ZERO],
        }
    }
}
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct RingShadow(pub RingData);
#[derive(Component, Clone, Default, ExtractComponent)]
#[require(Hdr,DepthPrepass,Msaa::Off,CameraMainTextureUsages=CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct RingView(pub RingData);
pub struct RingPlugin;
#[derive(Resource)]
struct Gpu {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}
impl Plugin for RingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "rings.wgsl");
        load_shader_library!(app, "ring_math.wgsl");
        load_shader_library!(app, "ring_transport.wgsl");
        app.init_resource::<RingShadow>().add_plugins((
            ExtractComponentPlugin::<RingView>::default(),
            ExtractResourcePlugin::<RingShadow>::default(),
        ));
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.add_systems(RenderStartup, initialize).add_systems(
                Core3d,
                compose
                    .after(crate::stellar_disks::StellarBackground)
                    .after(crate::radiance_sky::RadianceBackground)
                    .after(crate::point_sky::PointBackground)
                    .before(crate::atmosphere::AtmosphereBackground)
                    .in_set(Core3dSystems::MainPass),
            );
        }
    }
}
fn initialize(mut commands: Commands, assets: Res<AssetServer>, cache: Res<PipelineCache>) {
    let layout = BindGroupLayoutDescriptor::new(
        "ring transport",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<ViewUniform>(true)),
                (1, texture_depth_2d()),
                (
                    2,
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                ),
                (24, storage_buffer_read_only::<RingData>(false)),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("ring transport".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(assets.as_ref(), "rings.wgsl"),
        ..Default::default()
    });
    commands.insert_resource(Gpu { layout, pipeline });
}
fn compose(
    view: ViewQuery<(
        &RingView,
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
    let (ring, target, prepass, offset) = view.into_inner();
    if ring.0.p.centre.w <= 0.0 {
        return;
    }
    let Some(gpu) = gpu else {
        return;
    };
    let (Some(pipeline), Some(depth), Some(binding)) = (
        cache.get_compute_pipeline(gpu.pipeline),
        prepass.depth_view(),
        uniforms.uniforms.binding(),
    ) else {
        return;
    };
    let mut data = StorageBuffer::from(ring.0.clone());
    data.write_buffer(&device, &queue);
    let group = device.create_bind_group(
        "ring transport",
        &cache.get_bind_group_layout(&gpu.layout),
        &BindGroupEntries::with_indices((
            (0, binding),
            (1, depth),
            (2, target.main_texture_view()),
            (24, data.binding().unwrap()),
        )),
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("ring transport"),
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
