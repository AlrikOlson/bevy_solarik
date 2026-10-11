use super::{
    CanopyRegion, CanopySources,
    arena::Arena,
    asset::GpuRegion,
    scene::{self, CanopyGpu, Extracted, PrepareCanopy},
};
use bevy_app::{App, Plugin};
use bevy_asset::{AssetApp, embedded_asset};
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    extract_resource::ExtractResourcePlugin, render_asset::RenderAssetPlugin,
    scene_readiness::SceneGeometryReadiness,
};
use bevy_shader::load_shader_library;
pub struct CanopyPlugin;
impl Plugin for CanopyPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "filtered_canopy_prepass.wgsl");
        load_shader_library!(app, "filtered_canopy_transport.wgsl");
        load_shader_library!(app, "filtered_canopy_region.wgsl");
        load_shader_library!(app, "filtered_canopy_scene.wgsl");
        load_shader_library!(app, "filtered_canopy_conditioning.wgsl");
        app.init_asset::<CanopyRegion>()
            .init_resource::<CanopySources>()
            .init_resource::<SceneGeometryReadiness>()
            .add_plugins((
                RenderAssetPlugin::<GpuRegion>::default(),
                ExtractResourcePlugin::<CanopySources>::default(),
            ));
        let Some(render) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render
            .init_resource::<Arena>()
            .init_resource::<CanopyGpu>()
            .init_resource::<Extracted>()
            .init_resource::<CanopySources>()
            .add_systems(ExtractSchedule, scene::extract)
            .add_systems(
                Render,
                scene::prepare
                    .in_set(RenderSystems::PrepareResources)
                    .in_set(PrepareCanopy),
            )
            .add_systems(RenderStartup, super::prepass::init)
            .add_systems(
                Core3d,
                super::prepass::render
                    .after(Core3dSystems::Prepass)
                    .before(Core3dSystems::MainPass),
            );
    }
}
