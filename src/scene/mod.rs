mod assembly;
mod input_roots;
mod instance_rows;
pub use assembly::{RaytracingAssembly3d, RaytracingAssemblyPart};
mod binder;
mod blas;
pub mod collimated;
mod dependencies;
mod extract;
pub(crate) mod history;
mod instance_changes;
pub mod placement;

/// Enable full active-index receipts only for an explicit diagnostic capture.
#[derive(bevy_ecs::resource::Resource)]
pub struct CaptureSceneIndices;

#[cfg(feature = "graphics_debug")]
pub mod graphics_debug;
mod light_sampling;
#[cfg(feature = "graphics_debug")]
mod memory_profile;
mod ray_settings;
mod tlas;
mod types;

use bevy_shader::load_shader_library;
pub use binder::{RaytracingSceneBindings, SolarikAlphaTesting, SolarikSkyLight};
pub use ray_settings::SolarikRaySettings;
pub use types::RaytracingMesh3d;

use crate::SolarikPlugins;
use bevy_app::{App, Plugin};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
    extract_resource::ExtractResourcePlugin,
    mesh::{
        RenderMesh,
        allocator::{MeshAllocatorSettings, allocate_and_free_meshes},
    },
    render_asset::prepare_assets,
    render_resource::BufferUsages,
    renderer::RenderDevice,
};
use binder::prepare_raytracing_scene_bindings;
use blas::{BlasManager, compact_raytracing_blas, prepare_raytracing_blas};
use extract::{StandardMaterialAssets, extract_raytracing_scene};
use tracing::warn;

/// Creates acceleration structures and binding arrays of resources for raytracing.
pub struct RaytracingScenePlugin;

impl Plugin for RaytracingScenePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::surface_detail::SurfaceDetailPlugin>() {
            app.add_plugins(crate::surface_detail::SurfaceDetailPlugin);
        }
        if !app.is_plugin_added::<collimated::CollimatedEmissionPlugin>() {
            app.add_plugins(collimated::CollimatedEmissionPlugin);
        }
        app.add_plugins(crate::gaussian::GaussianDielectricPlugin);
        app.add_plugins(crate::lommel::LommelSeeligerPlugin);
        app.add_plugins(crate::rings::RingPlugin);
        crate::coarse_scene::load_shader(app);
        load_shader_library!(app, "brdf.wgsl");
        load_shader_library!(app, "light_medium.wgsl");
        load_shader_library!(app, "thin_glass.wgsl");
        load_shader_library!(app, "raytracing_scene_bindings.wgsl");
        load_shader_library!(app, "sampling.wgsl");

        // The sky is optional; the resource exists so the binder can read it
        // (no image = no sky, upstream behaviour).
        app.add_systems(bevy_app::First, assembly::previous_transforms);
        app.init_resource::<bevy_render::scene_readiness::SceneGeometryReadiness>();
        app.init_resource::<SolarikSkyLight>();
        app.init_resource::<SolarikAlphaTesting>();
        app.init_resource::<SolarikRaySettings>();
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        let render_device = render_app.world().resource::<RenderDevice>();
        let features = render_device.features();
        if !features.contains(SolarikPlugins::required_wgpu_features()) {
            warn!(
                "RaytracingScenePlugin not loaded. GPU lacks support for required features: {:?}.",
                SolarikPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        let readiness = app
            .world()
            .resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
            .clone();
        app.sub_app_mut(RenderApp).insert_resource(readiness);
        app.add_plugins((
            ExtractResourcePlugin::<StandardMaterialAssets>::default(),
            ExtractResourcePlugin::<SolarikSkyLight>::default(),
            ExtractResourcePlugin::<SolarikAlphaTesting>::default(),
            ExtractResourcePlugin::<SolarikRaySettings>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::BLAS_INPUT | BufferUsages::STORAGE;
        render_app.init_resource::<SolarikSkyLight>();
        render_app.init_resource::<SolarikAlphaTesting>();
        render_app.init_resource::<SolarikRaySettings>();

        render_app
            .init_gpu_resource::<BlasManager>()
            .init_gpu_resource::<StandardMaterialAssets>()
            .insert_resource(RaytracingSceneBindings::new())
            .add_systems(
                ExtractSchedule,
                (extract_raytracing_scene, assembly::extract),
            )
            .add_systems(
                Render,
                (
                    prepare_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .before(prepare_assets::<RenderMesh>)
                        .after(allocate_and_free_meshes),
                    compact_raytracing_blas
                        .in_set(RenderSystems::PrepareAssets)
                        .after(prepare_raytracing_blas),
                    prepare_raytracing_scene_bindings.in_set(RenderSystems::PrepareResources),
                    blas::publish_readiness
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_raytracing_scene_bindings),
                ),
            );
    }
}
