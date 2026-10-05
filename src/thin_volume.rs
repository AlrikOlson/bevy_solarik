//! Optically thin emission/scattering kernels, integrated in physical metres.
//! Caller supplies radiance density (cd/m³ linear RGB) and must ensure tau << 1.
//! No self extinction, multiple scattering or within-kernel lighting variation.
use bevy_app::{App, Plugin};
use bevy_asset::{AssetServer, embedded_asset, load_embedded_asset};
use bevy_camera::{CameraMainTextureUsages, Hdr};
use bevy_core_pipeline::{
    prepass::{DepthPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_math::{DVec3, Vec4};
use bevy_render::{
    RenderApp, RenderStartup,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
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

#[derive(Clone, ShaderType)]
pub struct GaussianKernel {
    /// Camera-relative centre, metres, in world axes. Update after camera motion.
    pub centre: Vec4,
    pub x: Vec4,
    pub y: Vec4,
    pub z: Vec4,
    pub emission: Vec4,
}
impl GaussianKernel {
    /// Axially symmetric Gaussian; `integrated_emission` is integral j dV,
    /// where j has units of linear RGB cd/m³, not a display colour.
    pub fn axial(
        centre: DVec3,
        along: DVec3,
        transverse_sigma: f64,
        longitudinal_sigma: f64,
        integrated_emission: DVec3,
    ) -> Option<Self> {
        if !centre.is_finite()
            || !along.is_finite()
            || along.length_squared() < 1e-20
            || !transverse_sigma.is_finite()
            || transverse_sigma <= 0.0
            || !longitudinal_sigma.is_finite()
            || longitudinal_sigma <= 0.0
            || !integrated_emission.is_finite()
            || integrated_emission.min_element() < 0.0
        {
            return None;
        }
        let z = along.normalize();
        let axis = if z.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
        let x = z.cross(axis).normalize();
        let y = z.cross(x);
        let norm =
            (2.0 * core::f64::consts::PI).powf(1.5) * transverse_sigma.powi(2) * longitudinal_sigma;
        let k = Self {
            centre: centre.as_vec3().extend(0.0),
            x: (x / transverse_sigma).as_vec3().extend(0.0),
            y: (y / transverse_sigma).as_vec3().extend(0.0),
            z: (z / longitudinal_sigma).as_vec3().extend(0.0),
            emission: (integrated_emission / norm).as_vec3().extend(0.0),
        };
        if [k.centre, k.x, k.y, k.z, k.emission]
            .iter()
            .all(|v| v.is_finite())
        {
            Some(k)
        } else {
            None
        }
    }
}
#[derive(Component, Clone, Default, ExtractComponent)]
#[require(Hdr,DepthPrepass,Msaa::Off,CameraMainTextureUsages=CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct ThinVolumes(pub Vec<GaussianKernel>);
pub struct ThinVolumePlugin;
#[derive(Resource)]
struct Gpu {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}
impl Plugin for ThinVolumePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "thin_volume.wgsl");
        load_shader_library!(app, "thin_volume_math.wgsl");
        app.add_plugins(ExtractComponentPlugin::<ThinVolumes>::default());
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
        "thin_volume",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ViewUniform>(true),
                texture_depth_2d(),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                storage_buffer_read_only::<Vec<GaussianKernel>>(false),
            ),
        ),
    );
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("thin_volume".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(assets.as_ref(), "thin_volume.wgsl"),
        ..Default::default()
    });
    commands.insert_resource(Gpu { layout, pipeline });
}
fn compose(
    view: ViewQuery<(
        &ThinVolumes,
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
    let (volumes, target, prepass, offset) = view.into_inner();
    if volumes.0.is_empty() {
        return;
    }
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
    let mut data = StorageBuffer::from(volumes.0.clone());
    data.write_buffer(&device, &queue);
    let group = device.create_bind_group(
        "thin_volume",
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
            label: Some("thin_volume"),
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
