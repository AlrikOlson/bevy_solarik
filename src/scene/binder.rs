#[cfg(test)]
#[path = "input_tests.rs"]
mod input_tests;

use super::collimated::CollimatedMaterials;
use super::instance_rows::SceneRows;
use super::{
    SolarikRaySettings,
    blas::BlasManager,
    extract::StandardMaterialAssets,
    light_sampling::{build_alias_table, local_flux, luminance},
};
use crate::gaussian::GaussianRayMaterials;
use crate::surface_detail::{DetailedRayMaterials, GpuSurfaceDetail};
use bevy_asset::{AssetId, Handle};
use bevy_color::{ColorToComponents, LinearRgba};
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::Entity,
    resource::Resource,
    system::{Local, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_material::AlphaMode;
use bevy_math::{Affine3A, Mat4, UVec4, Vec3, Vec4, ops::cos};
use bevy_mesh::Mesh;
use bevy_pbr::{DfgLut, ExtractedDirectionalLight, ExtractedPointLight, StandardMaterial};
use bevy_platform::{
    collections::{HashMap, HashSet},
    hash::FixedHasher,
};
use bevy_render::scene_slots::SceneInstance;
use bevy_render::{
    extract_resource::ExtractResource,
    mesh::allocator::MeshAllocator,
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{FallbackImage, GpuImage},
};
use core::{
    f32::consts::{PI, TAU},
    hash::Hash,
    num::NonZeroU32,
    ops::Deref,
};

const MAX_MESH_SLAB_COUNT: NonZeroU32 = NonZeroU32::new(500).unwrap();
const MAX_TEXTURE_COUNT: NonZeroU32 = NonZeroU32::new(5_000).unwrap();
// Shared scan atlases, not one texture per mesh. Keep sampler descriptors
// bounded independently of the much larger per-material 2D texture table.
const MAX_SCAN_ATLASES: NonZeroU32 = NonZeroU32::new(64).unwrap();

const TEXTURE_MAP_NONE: u32 = u32::MAX;
const LIGHT_NOT_PRESENT_THIS_FRAME: u32 = u32::MAX;

/// The sky: a cubemap image (the one a `Skybox` or an `EnvironmentMapLight`
/// would show, in the scene's radiance units) scaled by `intensity`. Rays
/// that leave the scene see it: the `ReSTIR` GI first bounce, the world
/// cache's GI rays, the specular GI paths and the pathtracer. No image, or an
/// intensity of zero, means no sky (upstream behaviour: escaped rays are
/// black).
#[derive(Resource, ExtractResource, Clone)]
pub struct SolarikSkyLight {
    /// A cubemap (`TextureViewDimension::Cube`) in the scene's radiance units.
    pub image: Option<Handle<Image>>,
    /// Multiplier on the cubemap's radiance.
    pub intensity: f32,
}

impl Default for SolarikSkyLight {
    fn default() -> Self {
        Self {
            image: None,
            intensity: 1.0,
        }
    }
}

/// Whether alpha-masked and blended materials get alpha-tested acceleration
/// structures (default) or are treated as opaque quads the way upstream
/// does. Off is the escape hatch for a scene where the any-hit cost is too
/// high.
#[derive(Resource, ExtractResource, Clone, Copy, Debug)]
pub struct SolarikAlphaTesting(pub bool);

impl Default for SolarikAlphaTesting {
    fn default() -> Self {
        Self(true)
    }
}

/// The intensity the shaders see: the sky's own when its image is bound,
/// zero when the fallback cubemap stands in for a missing or unloaded image
/// (the fallback is white, and a white sky is not what "no sky" means).
fn sky_shader_intensity(intensity: f32, image_bound: bool) -> f32 {
    if image_bound { intensity.max(0.0) } else { 0.0 }
}

#[derive(Resource)]
pub struct RaytracingSceneBindings {
    pub bind_group: Option<BindGroup>,
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// Render entities whose blended material and geometry are in this frame's TLAS.
    pub(crate) glass_entities: HashSet<Entity>,
    /// Whether this frame's TLAS contains explicitly transmissive foliage.
    pub(crate) has_foliage: bool,
    /// Region about the render origin untouched by this frame's geometry.
    /// Infinity means unchanged; zero means unknown or global lighting change.
    pub(crate) history_stable_radius: f32,
    pub(crate) history_generation: u64,
    pub(crate) history_regional: bool,
    history_regions: StorageBuffer<Vec<Vec4>>,
    previous_frame_light_entities: Vec<SceneInstance>,
    settle_light_history: bool,
    #[cfg(feature = "graphics_debug")]
    /// Capture-only prepared scene generations and readback handles.
    pub debug: super::graphics_debug::SceneSnapshot,
}

#[derive(Default)]
pub(crate) struct SceneCache {
    inputs: Option<SceneInputs>,
    storage: SceneStorage,
    tlas: super::tlas::SceneTlas,
    slots: bevy_render::scene_slots::SceneSlots,
    instance_inputs: Vec<Option<RayInstanceInput>>,
    input_roots: super::input_roots::InputRoots,
    history_stable_radius: f32,
    regions: super::history::Regions,
    dependencies: super::dependencies::Dependencies,
    blas_generation: u64,
}

#[derive(Default)]
struct SceneStorage {
    materials: StorageBufferList<GpuMaterial>,
    detail_parameters: StorageBufferList<GpuSurfaceDetail>,
    transforms: StorageBufferList<Mat4>,
    previous_frame_transforms: StorageBufferList<Mat4>,
    geometry_ids: StorageBufferList<GpuInstanceGeometryIds>,
    material_ids: StorageBufferList<u32>,
    active_indices: StorageBufferList<u32>,
    light_sources: StorageBufferList<GpuLightSource>,
    directional_lights: StorageBufferList<GpuDirectionalLight>,
    local_lights: StorageBufferList<GpuLocalLight>,
    previous_frame_light_id_translations: StorageBufferList<u32>,
    sky: StorageBuffer<GpuSkyLight>,
    rings: StorageBuffer<crate::rings::RingData>,
}
#[derive(Clone)]
pub(super) struct RayInstanceInput {
    pub(super) entity: SceneInstance,
    pub(super) mesh: AssetId<Mesh>,
    pub(super) material: AssetId<StandardMaterial>,
    pub(super) transform: Affine3A,
    pub(super) previous: Affine3A,
}
impl PartialEq for RayInstanceInput {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
            && self.mesh == other.mesh
            && self.material == other.material
            && self.transform.to_cols_array().map(f32::to_bits)
                == other.transform.to_cols_array().map(f32::to_bits)
            && self.previous.to_cols_array().map(f32::to_bits)
                == other.previous.to_cols_array().map(f32::to_bits)
    }
}
#[derive(PartialEq)]
struct SceneInputs {
    directional: Vec<(Entity, GpuDirectionalLight)>,
    local: Vec<(Entity, GpuLocalLight)>,
    sky: Option<AssetId<Image>>,
    sky_intensity: f32,
    ray_max: f32,
    ray_min: f32,
    alpha_testing: bool,
    medium: GpuLightMedium,
    weather: Option<AssetId<Image>>,
    dfg: AssetId<Image>,
    rings: crate::rings::RingData,
}

fn update_instance_inputs(
    cache: &mut SceneCache,
    rows: &SceneRows<'_, '_>,
    queue: &RenderQueue,
    (dependencies, roots): (bool, Option<&HashSet<Entity>>),
    blas: &BlasManager,
    materials: &StandardMaterialAssets,
    changes: &super::dependencies::Changes,
) -> (Vec<u32>, usize) {
    let _profile = bevy_render::diagnostic::profile_scope("scene.prepare_inputs");
    cache.slots.begin();
    let mut dirty = Vec::new();
    let mut added = 0usize;
    let mut visited = 0usize;
    let (removed, roots_visited) = if let Some(roots) = roots {
        let mut retiring = Vec::new();
        for &root in roots {
            let inputs = rows.for_root(root);
            for part in cache.input_roots.replace(root, &inputs) {
                if let Some(slot) = cache.slots.get_part(part) {
                    retiring.push((part, slot));
                }
            }
            visited += inputs.len();
            for input in inputs {
                update_input_history(cache, &input, blas, materials, changes);
                added += usize::from(observe_instance(cache, input, dependencies, &mut dirty));
            }
        }
        (cache.slots.remove_parts(queue, retiring), roots.len())
    } else {
        cache.input_roots.clear();
        for input in rows.iter() {
            cache.input_roots.observe(&input);
            update_input_history(cache, &input, blas, materials, changes);
            added += usize::from(observe_instance(cache, input, dependencies, &mut dirty));
            visited += 1;
        }
        (cache.slots.finish(queue), cache.input_roots.len())
    };
    profile_inputs(roots.is_some(), roots_visited, visited);
    for slot in &removed {
        if let Some(old) = &cache.instance_inputs[slot.index as usize] {
            cache.regions.record(
                changes.old_bounds(old, history_bounds(old, blas, materials)),
                old.transform,
            );
            cache.history_stable_radius = cache.history_stable_radius.min(
                changes
                    .old_bounds(old, history_bounds(old, blas, materials))
                    .map_or(0.0, |bounds| bounds.stable_radius(old.transform)),
            );
        }
        cache.instance_inputs[slot.index as usize] = None;
    }
    profile_slots(cache, added, removed.len(), dirty.len());
    (dirty, removed.len())
}
fn profile_inputs(partial: bool, roots: usize, parts: usize) {
    bevy_render::diagnostic::profile_value(
        "scene.partial_input_update",
        f64::from(partial),
        "count",
    );
    bevy_render::diagnostic::profile_value("scene.input_roots_visited", roots as f64, "count");
    bevy_render::diagnostic::profile_value("scene.input_parts_visited", parts as f64, "count");
}
fn update_input_history(
    cache: &mut SceneCache,
    input: &RayInstanceInput,
    blas: &BlasManager,
    materials: &StandardMaterialAssets,
    changes: &super::dependencies::Changes,
) {
    let previous_input = cache
        .slots
        .get_part(input.entity)
        .and_then(|slot| cache.instance_inputs[slot.index as usize].as_ref());
    // Previous-transform settling does not change an occluder.
    if changes.affects(input) || previous_input.is_none_or(|old| !same_occluder(old, input)) {
        if let Some(old) = previous_input {
            cache.regions.record(
                changes.old_bounds(old, history_bounds(old, blas, materials)),
                old.transform,
            );
        }
        cache
            .regions
            .record(history_bounds(input, blas, materials), input.transform);
        let previous_radius = previous_input.map_or(f32::INFINITY, |old| {
            changes
                .old_bounds(old, history_bounds(old, blas, materials))
                .map_or(0.0, |bounds| bounds.stable_radius(old.transform))
        });
        cache.history_stable_radius = cache
            .history_stable_radius
            .min(previous_radius)
            .min(history_distance(input, blas, materials));
    }
}
fn same_occluder(old: &RayInstanceInput, new: &RayInstanceInput) -> bool {
    old.entity == new.entity
        && old.mesh == new.mesh
        && old.material == new.material
        && old.transform.to_cols_array().map(f32::to_bits)
            == new.transform.to_cols_array().map(f32::to_bits)
}

fn history_bounds(
    input: &RayInstanceInput,
    blas: &BlasManager,
    materials: &StandardMaterialAssets,
) -> Option<super::history::Bounds> {
    let material = materials.get(&input.material)?;
    if material.emissive.red != 0.0
        || material.emissive.green != 0.0
        || material.emissive.blue != 0.0
    {
        return None;
    }
    blas.bounds.get(&input.mesh).copied()
}

fn history_distance(
    input: &RayInstanceInput,
    blas: &BlasManager,
    materials: &StandardMaterialAssets,
) -> f32 {
    let Some(material) = materials.get(&input.material) else {
        return 0.0;
    };
    // Moving an emitter can change selection probabilities and lighting anywhere.
    if material.emissive.red != 0.0
        || material.emissive.green != 0.0
        || material.emissive.blue != 0.0
    {
        return 0.0;
    }
    blas.bounds
        .get(&input.mesh)
        .map_or(0.0, |bounds| bounds.stable_radius(input.transform))
}

fn observe_instance(
    cache: &mut SceneCache,
    input: RayInstanceInput,
    dependencies: bool,
    dirty: &mut Vec<u32>,
) -> bool {
    let fresh = cache.slots.get_part(input.entity).is_none();
    let slot = cache.slots.touch_part(input.entity);
    let index = slot.index as usize;
    if cache.instance_inputs.len() <= index {
        cache.instance_inputs.resize_with(index + 1, || None);
    }
    if dependencies || cache.instance_inputs[index].as_ref() != Some(&input) {
        dirty.push(slot.index);
        cache.instance_inputs[index] = Some(input);
    }
    fresh
}
fn profile_slots(cache: &SceneCache, added: usize, removed: usize, dirty: usize) {
    let (live, free, retired) = cache.slots.counts();
    for (name, count) in [
        ("scene.slots_live", live),
        ("scene.slots_free", free),
        ("scene.slots_retired", retired),
    ] {
        bevy_render::diagnostic::profile_value(name, count as f64, "count");
    }
    bevy_render::diagnostic::profile_value("scene.slots_added", added as f64, "count");
    bevy_render::diagnostic::profile_value("scene.slots_removed", removed as f64, "count");
    bevy_render::diagnostic::profile_value("scene.slots_changed", dirty as f64, "count");
    bevy_render::diagnostic::profile_value(
        "scene.slot_capacity",
        cache.slots.capacity() as f64,
        "count",
    );
}

pub(crate) fn prepare_raytracing_scene_bindings(
    (instances_query, mut input_changes): (
        SceneRows<'_, '_>,
        super::instance_changes::RayInstanceChanges,
    ),
    directional_lights_query: Query<(Entity, &ExtractedDirectionalLight)>,
    // Point and spot lights: bevy_pbr extracts both into ExtractedPointLight
    // (spot_light_angles tells them apart).
    local_lights_query: Query<(Entity, &ExtractedPointLight)>,
    mesh_allocator: Res<MeshAllocator>,
    mut blas_manager: ResMut<BlasManager>,
    material_assets: Res<StandardMaterialAssets>,
    (collimated, detailed, gaussian, lommel): (
        Res<CollimatedMaterials>,
        Res<DetailedRayMaterials>,
        Res<GaussianRayMaterials>,
        Res<crate::lommel::LommelRayMaterials>,
    ),
    texture_assets: Res<RenderAssets<GpuImage>>,
    fallback_texture: Res<FallbackImage>,
    dfg_lut: Res<DfgLut>,
    sky_light: Res<SolarikSkyLight>,
    (alpha_testing, ray_settings, planet, atmosphere, rings, readiness): (
        Res<SolarikAlphaTesting>,
        Res<SolarikRaySettings>,
        Option<Res<crate::atmosphere::PlanetaryAtmosphere>>,
        Option<Res<crate::atmosphere::AtmosphereState>>,
        Res<crate::rings::RingShadow>,
        Res<bevy_render::scene_readiness::SceneGeometryReadiness>,
    ),
    (render_device, diagnostics, mut scene_cache, frame, _capture_indices): (
        Res<RenderDevice>,
        Option<Res<bevy_render::diagnostic::DiagnosticsRecorder>>,
        Local<SceneCache>,
        Option<Res<bevy_diagnostic::FrameCount>>,
        Option<Res<super::CaptureSceneIndices>>,
    ),
    pipeline_cache: Res<PipelineCache>,
    render_queue: Res<RenderQueue>,
    mut raytracing_scene_bindings: ResMut<RaytracingSceneBindings>,
) {
    let _source = frame.and_then(|f| bevy_render::diagnostic::profile_source_frame(f.0));
    let _cpu_profile = bevy_render::diagnostic::profile_scope("scene.prepare");
    bevy_render::diagnostic::profile_value(
        "scene.ray_instances",
        instances_query.len() as f64,
        "count",
    );

    // Meshes with an alpha-masked or blended material on any instance need a
    // BLAS the shader can alpha-test; the manager rebuilds the ones built the
    // other way (they are missing from the TLAS for the frame it takes).
    let opacity_profile = bevy_render::diagnostic::profile_scope("scene.prepare_opacity");
    let changed_roots = input_changes.collect();
    let instance_inputs_unchanged = changed_roots.is_empty();
    if !instance_inputs_unchanged || material_assets.is_changed() || alpha_testing.is_changed() {
        let mut non_opaque_meshes: HashSet<AssetId<Mesh>> = instances_query
            .iter()
            .filter(|input| {
                alpha_testing.0
                    && material_assets
                        .get(&input.material)
                        .is_some_and(|material| {
                            material_alpha(material).flags & MATERIAL_FLAG_OPAQUE == 0
                        })
            })
            .map(|input| input.mesh)
            .collect();
        non_opaque_meshes.extend(readiness.alpha_requested::<Mesh>());
        blas_manager.set_non_opaque_meshes(non_opaque_meshes);
    }

    drop(opacity_profile);
    let weather = planet
        .as_deref()
        .filter(|p| p.cloud_coverage > 0.0)
        .and_then(|p| p.weather.as_ref());
    let inputs = SceneInputs {
        directional: directional_lights_query
            .iter()
            .map(|(entity, light)| (entity, GpuDirectionalLight::new(light)))
            .collect(),
        local: local_lights_query
            .iter()
            .map(|(entity, light)| (entity, GpuLocalLight::new(light)))
            .collect(),
        sky: sky_light.image.as_ref().map(Handle::id),
        sky_intensity: sky_light.intensity,
        ray_max: ray_settings.max_distance(),
        ray_min: ray_settings.relative_min_distance(),
        alpha_testing: alpha_testing.0,
        medium: GpuLightMedium::new(
            planet.as_deref(),
            atmosphere.as_deref(),
            weather
                .and_then(|image| texture_assets.get(image.id()))
                .is_some(),
        ),
        weather: weather.map(Handle::id),
        dfg: dfg_lut.texture.id(),
        rings: rings.0.clone(),
    };
    let blas_changed = scene_cache.blas_generation != blas_manager.generation;
    scene_cache.blas_generation = blas_manager.generation;
    let resources_changed = blas_changed
        || mesh_allocator.is_changed()
        || material_assets.is_changed()
        || collimated.is_changed()
        || detailed.is_changed()
        || gaussian.is_changed()
        || lommel.is_changed()
        || texture_assets.is_changed()
        || fallback_texture.is_changed();
    // Pending geometry retains ownership but needs another full readiness pass.
    // Count retained inputs before applying membership deltas, not the new query.
    let full_inputs = resources_changed
        || !scene_cache.input_roots.initialized
        || scene_cache.slots.counts().0 != scene_cache.slots.active_indices().len();
    let unchanged_inputs = instance_inputs_unchanged && !full_inputs;
    let dependency_profile = bevy_render::diagnostic::profile_scope("scene.prepare_dependencies");
    let changes = if unchanged_inputs {
        super::dependencies::Changes::default()
    } else {
        scene_cache.dependencies.observe(
            instances_query.iter(),
            &blas_manager,
            &material_assets,
            &texture_assets,
            (&detailed, &gaussian),
            [inputs.sky, inputs.weather, Some(inputs.dfg)],
        )
    };
    drop(dependency_profile);
    // Asset arrivals and BLAS compaction are publication events, not global
    // lighting changes. Used geometry/material dependencies invalidate their
    // old/new support; emitter, environment and unknown extensions remain global.
    let global_history_change = changes.global
        || scene_cache.inputs.as_ref() != Some(&inputs)
        || collimated.is_changed()
        || detailed.is_changed()
        || gaussian.is_changed()
        || lommel.is_changed()
        || fallback_texture.is_changed();
    scene_cache.history_stable_radius = if global_history_change {
        0.0
    } else {
        f32::INFINITY
    };
    scene_cache.regions.clear(global_history_change);
    let (mut dirty_indices, removed) = if unchanged_inputs {
        profile_slots(&scene_cache, 0, 0, 0);
        profile_inputs(false, 0, 0);
        (Vec::new(), 0)
    } else {
        update_instance_inputs(
            &mut scene_cache,
            &instances_query,
            &render_queue,
            (resources_changed, (!full_inputs).then_some(&changed_roots)),
            &blas_manager,
            &material_assets,
            &changes,
        )
    };
    if scene_cache.history_stable_radius.is_finite() {
        raytracing_scene_bindings.history_stable_radius = scene_cache.history_stable_radius;
        raytracing_scene_bindings.history_regional = !scene_cache.regions.global;
        raytracing_scene_bindings
            .history_regions
            .set(scene_cache.regions.packed());
        raytracing_scene_bindings
            .history_regions
            .write_buffer_changed(&render_device, &render_queue);
        raytracing_scene_bindings.history_generation = raytracing_scene_bindings
            .history_generation
            .wrapping_add(1)
            .max(1);
    }
    bevy_render::diagnostic::profile_value(
        "scene.history_changed",
        f64::from(scene_cache.history_stable_radius.is_finite()),
        "count",
    );
    bevy_render::diagnostic::profile_value(
        "scene.history_full_reset",
        f64::from(scene_cache.history_stable_radius == 0.0),
        "count",
    );
    let reused = dirty_indices.is_empty()
        && removed == 0
        && !resources_changed
        && !raytracing_scene_bindings.settle_light_history
        && scene_cache.inputs.as_ref() == Some(&inputs)
        && raytracing_scene_bindings.bind_group.is_some();
    bevy_render::diagnostic::profile_value("scene.reused", f64::from(reused), "count");
    bevy_render::diagnostic::profile_value(
        "scene.dependencies_changed",
        f64::from(resources_changed),
        "count",
    );
    if reused {
        return;
    }
    scene_cache.inputs = Some(inputs);
    raytracing_scene_bindings.settle_light_history = false;
    raytracing_scene_bindings.bind_group = None;
    raytracing_scene_bindings.glass_entities.clear();
    #[cfg(feature = "graphics_debug")]
    {
        raytracing_scene_bindings.debug = super::graphics_debug::SceneSnapshot::default();
    }
    raytracing_scene_bindings.has_foliage = false;

    let mut this_frame_entity_to_light_id = HashMap::<SceneInstance, u32>::default();
    let previous_frame_light_entities: Vec<_> = raytracing_scene_bindings
        .previous_frame_light_entities
        .drain(..)
        .collect();

    if instances_query.len() == 0 {
        scene_cache.tlas.prepare(&render_device, 0);
        scene_cache.storage.active_indices.get_mut().clear();
        return;
    }

    let mut vertex_buffers = CachedBindingArray::new();
    let mut index_buffers = CachedBindingArray::new();
    let mut textures = CachedBindingArray::new();
    let mut samplers = Vec::new();
    let SceneCache {
        storage,
        tlas,
        slots,
        instance_inputs,
        ..
    } = &mut *scene_cache;
    let slot_capacity = slots.capacity();
    storage
        .transforms
        .get_mut()
        .resize(slot_capacity, Mat4::IDENTITY);
    storage
        .previous_frame_transforms
        .get_mut()
        .resize(slot_capacity, Mat4::IDENTITY);
    storage
        .geometry_ids
        .get_mut()
        .resize(slot_capacity, GpuInstanceGeometryIds::default());
    storage.material_ids.get_mut().resize(slot_capacity, 0);
    let (tlas, tlas_reused) = tlas.prepare(&render_device, instances_query.len());
    bevy_render::diagnostic::profile_value(
        "scene.tlas_allocation_reused",
        f64::from(tlas_reused),
        "count",
    );
    let SceneStorage {
        materials,
        detail_parameters,
        transforms,
        previous_frame_transforms,
        geometry_ids,
        material_ids,
        active_indices,
        light_sources,
        directional_lights,
        local_lights,
        previous_frame_light_id_translations,
        sky: sky_buffer,
        rings: ring_buffer,
    } = storage;
    materials.get_mut().clear();
    detail_parameters.get_mut().clear();
    let mut active_dirty_indices = Vec::new();
    light_sources.get_mut().clear();
    directional_lights.get_mut().clear();
    local_lights.get_mut().clear();
    previous_frame_light_id_translations.get_mut().clear();
    let mut scan_textures = CachedBindingArray::new();
    let mut scan_samplers = Vec::new();
    let mut light_fluxes = Vec::new();

    let mut material_id_map: HashMap<AssetId<StandardMaterial>, u32, FixedHasher> =
        HashMap::default();
    let mut material_id = 0;
    let mut process_texture = |texture_handle: &Option<Handle<_>>| -> Option<u32> {
        match texture_handle {
            Some(texture_handle) => match texture_assets.get(texture_handle.id()) {
                Some(texture) => {
                    let (texture_id, is_new) =
                        textures.push_if_absent(texture.texture_view.deref(), texture_handle.id());
                    if is_new {
                        samplers.push(texture.sampler.deref());
                    }
                    Some(texture_id)
                }
                None => None,
            },
            None => Some(TEXTURE_MAP_NONE),
        }
    };
    let materials_profile = bevy_render::diagnostic::profile_scope("scene.prepare_materials");
    for (asset_id, material) in material_assets.iter() {
        if material_alpha(material).flags
            & (MATERIAL_FLAG_OPAQUE
                | MATERIAL_FLAG_ALPHA_MASK
                | MATERIAL_FLAG_ALPHA_BLEND
                | MATERIAL_FLAG_DIFFUSE_BLEND)
            == 0
        {
            continue;
        }
        let Some(base_color_texture_id) = process_texture(&material.base_color_texture) else {
            continue;
        };
        let Some(normal_map_texture_id) = process_texture(&material.normal_map_texture) else {
            continue;
        };
        let Some(emissive_texture_id) = process_texture(&material.emissive_texture) else {
            continue;
        };
        let Some(metallic_roughness_texture_id) =
            process_texture(&material.metallic_roughness_texture)
        else {
            continue;
        };

        let mut gaussian_parameters = Vec4::ZERO;
        if let Some(extension) = gaussian.0.get(asset_id) {
            let Some(mask) = process_texture(&Some(extension.mask.clone())) else {
                continue;
            };
            gaussian_parameters = extension.parameters;
            gaussian_parameters.w = mask as f32;
        }
        let mut detail = GpuSurfaceDetail {
            textures: UVec4::splat(TEXTURE_MAP_NONE),
            ..Default::default()
        };
        if let Some(extension) = detailed.0.get(asset_id) {
            let (
                Some(cover0),
                Some(cover1),
                Some(colour),
                Some(normal),
                Some(meso_colour),
                Some(meso_normal),
            ) = (
                process_texture(&Some(extension.coverage0.clone())),
                process_texture(&Some(extension.coverage1.clone())),
                texture_assets.get(&extension.colour),
                texture_assets.get(&extension.detail),
                texture_assets.get(&extension.meso_colour),
                texture_assets.get(&extension.meso_detail),
            )
            else {
                continue;
            };
            let mut scan_ids = [0; 4];
            for (slot, (image, id)) in [
                (colour, extension.colour.id()),
                (normal, extension.detail.id()),
                (meso_colour, extension.meso_colour.id()),
                (meso_normal, extension.meso_detail.id()),
            ]
            .into_iter()
            .enumerate()
            {
                let (index, new) = scan_textures.push_if_absent(image.texture_view.deref(), id);
                if new {
                    scan_samplers.push(image.sampler.deref());
                }
                scan_ids[slot] = index;
            }
            detail.coordinates = extension.coordinates.clone();
            detail.textures = UVec4::new(cover0, cover1, scan_ids[0], scan_ids[1]);
            detail.meso = UVec4::new(scan_ids[2], scan_ids[3], 0, 0);
        }
        detail_parameters.get_mut().push(detail);
        let alpha = material_alpha(material);
        materials.get_mut().push(GpuMaterial {
            normal_map_texture_id,
            base_color_texture_id,
            emissive_texture_id,
            metallic_roughness_texture_id,

            base_color: LinearRgba::from(material.base_color).to_vec3(),
            perceptual_roughness: material.perceptual_roughness,
            emissive: material.emissive.to_vec3(),
            metallic: material.metallic,
            alpha_cutoff: alpha.cutoff,
            flags: alpha.flags | if lommel.0.contains(asset_id) { 32 } else { 0 },
            base_color_alpha: LinearRgba::from(material.base_color).alpha,
            reflectance: material.reflectance,
            gaussian: gaussian_parameters,
            emission_cone: collimated
                .0
                .get(asset_id)
                .map_or(Vec4::ZERO, |cone| cone.gpu()),
        });

        material_id_map.insert(*asset_id, material_id);
        material_id += 1;
    }

    drop(materials_profile);
    if material_id == 0 {
        return;
    }

    if textures.is_empty() {
        textures.vec.push(fallback_texture.d2.texture_view.deref());
        samplers.push(fallback_texture.d2.sampler.deref());
    }

    let instances_profile = bevy_render::diagnostic::profile_scope("scene.prepare_instances");
    let mut instance_id = 0;
    let mut material_transport_flags = 0;
    // Invocation-local: allocator slabs and BLAS handles cannot outlive this
    // resource snapshot. Missing sources are cached only until the next call.
    let mut mesh_sources: HashMap<AssetId<Mesh>, Option<(&Blas, GpuInstanceGeometryIds)>> =
        HashMap::default();
    for (index, input) in instance_inputs.iter().enumerate() {
        let Some(input) = input else {
            continue;
        };
        let entity = input.entity;
        let mesh = input.mesh;
        let material = input.material;
        let slot = slots.get_part(entity).expect("retained ray part");
        let Some(material_id) = material_id_map.get(&material).copied() else {
            slots.deactivate(slot);
            continue;
        };
        let Some(material) = materials.get().get(material_id as usize) else {
            slots.deactivate(slot);
            continue;
        };

        // Resolve bindings at the first material-ready use, preserving the
        // original binding-array insertion order for every eligible source.
        let Some((blas, geometry)) = mesh_sources
            .entry(mesh)
            .or_insert_with(|| {
                let blas = blas_manager.get(&mesh)?;
                let vertex_slice = mesh_allocator.mesh_vertex_slice(&mesh)?;
                let index_slice = mesh_allocator.mesh_index_slice(&mesh)?;
                let (vertex_buffer_id, _) = vertex_buffers.push_if_absent(
                    vertex_slice.buffer.as_entire_buffer_binding(),
                    vertex_slice.buffer.id(),
                );
                let (index_buffer_id, _) = index_buffers.push_if_absent(
                    index_slice.buffer.as_entire_buffer_binding(),
                    index_slice.buffer.id(),
                );
                Some((
                    blas,
                    GpuInstanceGeometryIds {
                        vertex_buffer_id,
                        vertex_buffer_offset: vertex_slice.range.start,
                        index_buffer_id,
                        index_buffer_offset: index_slice.range.start,
                        triangle_count: (index_slice.range.len() / 3) as u32,
                        light_probability: 0.0,
                    },
                ))
            })
            .as_ref()
        else {
            slots.deactivate(slot);
            continue;
        };
        material_transport_flags |= material.flags;
        raytracing_scene_bindings.has_foliage |= alpha_testing.0 && (material.flags >> 16) != 0;
        if alpha_testing.0
            && material.flags & (MATERIAL_FLAG_ALPHA_BLEND | MATERIAL_FLAG_DIFFUSE_BLEND) != 0
        {
            raytracing_scene_bindings
                .glass_entities
                .insert(entity.entity);
        }
        let transform = Mat4::from(input.transform);
        *tlas.get_mut_single(instance_id).unwrap() = Some(TlasInstance::new(
            blas,
            tlas_transform(&transform),
            slot.index,
            0xFF,
        ));

        assert!(slot.index < (1 << 24), "TLAS custom slot capacity");
        transforms.get_mut()[index] = transform;
        previous_frame_transforms.get_mut()[index] = Mat4::from(input.previous);
        slots.activate(slot);
        if active_indices.set_index(instance_id, slot.index) {
            active_dirty_indices.push(u32::try_from(instance_id).expect("active index capacity"));
        }

        if geometry_ids.get()[index] != *geometry || material_ids.get()[index] != material_id {
            dirty_indices.push(slot.index);
        }
        geometry_ids.get_mut()[index] = geometry.clone();
        material_ids.get_mut()[index] = material_id;

        if material.emissive != Vec3::ZERO {
            // Texture-average emission is not available on the CPU. The
            // material factor is a power estimate; PDFs remain exact.
            light_fluxes.push(
                luminance(material.emissive)
                    * blas_manager.mesh_world_area(&mesh, transform)
                    * f64::from(PI)
                    * if material.emission_cone.w > 0.0 {
                        let radius = f64::from(material.emission_cone.w);
                        radius * radius / (1.0 + radius * radius)
                    } else {
                        1.0
                    },
            );
            light_sources
                .get_mut()
                .push(GpuLightSource::new_emissive_mesh_light(
                    slot.index,
                    geometry.triangle_count,
                ));

            this_frame_entity_to_light_id
                .insert(entity.into(), light_sources.get().len() as u32 - 1);
            raytracing_scene_bindings
                .previous_frame_light_entities
                .push(entity.into());
        }

        instance_id += 1;
    }

    bevy_render::diagnostic::profile_value(
        "scene.mesh_sources_resolved",
        mesh_sources.len() as f64,
        "count",
    );
    bevy_render::diagnostic::profile_value(
        "scene.mesh_source_instances",
        instance_id as f64,
        "count",
    );
    drop(instances_profile);
    active_indices.get_mut().truncate(instance_id);
    if instance_id == 0 {
        return;
    }

    for (entity, directional_light) in &directional_lights_query {
        let directional_lights = directional_lights.get_mut();
        let directional_light_id = directional_lights.len() as u32;

        directional_lights.push(GpuDirectionalLight::new(directional_light));
        // A nominal 1 m² collection area puts lux in the same power scale
        // as local lumens. This changes variance, never emitted radiance.
        light_fluxes.push(
            luminance(directional_light.color.to_vec3())
                * f64::from(directional_light.illuminance.max(0.0)),
        );

        light_sources
            .get_mut()
            .push(GpuLightSource::new_directional_light(directional_light_id));

        this_frame_entity_to_light_id.insert(entity.into(), light_sources.get().len() as u32 - 1);
        raytracing_scene_bindings
            .previous_frame_light_entities
            .push(entity.into());
    }

    // Point and spot lights are sphere lights of the light's radius, in the
    // raster path's candela; a light that cannot light anything (no power)
    // is left out so it costs no candidates.
    for (entity, local_light) in &local_lights_query {
        if local_light.intensity <= 0.0 || local_light.color == LinearRgba::BLACK {
            continue;
        }
        let local_lights = local_lights.get_mut();
        let local_light_id = local_lights.len() as u32;

        local_lights.push(GpuLocalLight::new(local_light));
        light_fluxes.push(local_flux(
            local_light.color.to_vec3(),
            local_light.intensity,
            local_light.spot_light_angles,
        ));

        light_sources
            .get_mut()
            .push(if local_light.spot_light_angles.is_some() {
                GpuLightSource::new_spot_light(local_light_id)
            } else {
                GpuLightSource::new_point_light(local_light_id)
            });

        this_frame_entity_to_light_id.insert(entity.into(), light_sources.get().len() as u32 - 1);
        raytracing_scene_bindings
            .previous_frame_light_entities
            .push(entity.into());
    }

    raytracing_scene_bindings.settle_light_history =
        previous_frame_light_entities != raytracing_scene_bindings.previous_frame_light_entities;
    for previous_frame_light_entity in previous_frame_light_entities {
        let current_frame_index = this_frame_entity_to_light_id
            .get(&previous_frame_light_entity)
            .copied()
            .unwrap_or(LIGHT_NOT_PRESENT_THIS_FRAME);
        previous_frame_light_id_translations
            .get_mut()
            .push(current_frame_index);
    }

    if light_sources.get().len() > u16::MAX as usize {
        panic!("Too many light sources in the scene, maximum is 65535.");
    }

    for (source, bin) in light_sources
        .get_mut()
        .iter_mut()
        .zip(build_alias_table(&light_fluxes))
    {
        source.selection_probability = bin.probability;
        source.alias_threshold = bin.threshold;
        source.alias_index = bin.alias;
        if source.kind & 1 == 0 {
            geometry_ids.get_mut()[source.id as usize].light_probability = bin.probability;
        }
    }

    let upload_profile = bevy_render::diagnostic::profile_scope("scene.upload");
    #[cfg(feature = "graphics_debug")]
    {
        if active_indices.buffer().is_none() {
            active_indices.set_label(Some("ray_active_indices"));
            active_indices.add_usages(BufferUsages::COPY_SRC);
        }
        if materials.buffer().is_none() {
            materials.set_label(Some("scene_materials"));
            materials.add_usages(BufferUsages::COPY_SRC);
        }
        if transforms.buffer().is_none() {
            transforms.set_label(Some("scene_transforms"));
            transforms.add_usages(BufferUsages::COPY_SRC);
        }
        if previous_frame_transforms.buffer().is_none() {
            previous_frame_transforms.set_label(Some("scene_previous_transforms"));
            previous_frame_transforms.add_usages(BufferUsages::COPY_SRC);
        }
        if geometry_ids.buffer().is_none() {
            geometry_ids.set_label(Some("scene_geometry_ids"));
            geometry_ids.add_usages(BufferUsages::COPY_SRC);
        }
        if material_ids.buffer().is_none() {
            material_ids.set_label(Some("scene_material_ids"));
            material_ids.add_usages(BufferUsages::COPY_SRC);
        }
    }
    let mut batch = StorageBufferUploadBatch::default();
    let uploads = [
        active_indices.stage_buffer_indices(
            &render_device,
            &render_queue,
            &active_dirty_indices,
            &mut batch,
        ),
        materials.write_buffer_changed(&render_device, &render_queue),
        detail_parameters.write_buffer_changed(&render_device, &render_queue),
        transforms.stage_buffer_indices(&render_device, &render_queue, &dirty_indices, &mut batch),
        previous_frame_transforms.stage_buffer_indices(
            &render_device,
            &render_queue,
            &dirty_indices,
            &mut batch,
        ),
        geometry_ids.stage_buffer_indices(
            &render_device,
            &render_queue,
            &dirty_indices,
            &mut batch,
        ),
        material_ids.stage_buffer_indices(
            &render_device,
            &render_queue,
            &dirty_indices,
            &mut batch,
        ),
        light_sources.write_buffer_changed(&render_device, &render_queue),
        directional_lights.write_buffer_changed(&render_device, &render_queue),
        local_lights.write_buffer_changed(&render_device, &render_queue),
        previous_frame_light_id_translations.write_buffer_changed(&render_device, &render_queue),
    ];
    let (staging_bytes, staging_buffers) = batch.finish(&render_device, &render_queue);
    bevy_render::diagnostic::profile_value("scene.staging_bytes", staging_bytes as f64, "bytes");
    bevy_render::diagnostic::profile_value(
        "scene.staging_buffers",
        staging_buffers as f64,
        "count",
    );
    bevy_render::diagnostic::profile_value(
        "scene.upload_bytes",
        uploads.iter().map(|u| u.bytes as f64).sum(),
        "bytes",
    );
    bevy_render::diagnostic::profile_value(
        "scene.upload_ranges",
        uploads.iter().map(|u| f64::from(u.ranges)).sum(),
        "count",
    );
    bevy_render::diagnostic::profile_value(
        "scene.buffer_allocations",
        uploads.iter().filter(|u| u.allocated).count() as f64,
        "count",
    );
    drop(upload_profile);
    #[cfg(feature = "graphics_debug")]
    {
        raytracing_scene_bindings.debug = super::graphics_debug::SceneSnapshot {
            buffers: [
                ("ray_transforms", transforms.buffer()),
                (
                    "ray_previous_transforms",
                    previous_frame_transforms.buffer(),
                ),
                ("ray_geometry_ids", geometry_ids.buffer()),
                ("ray_material_ids", material_ids.buffer()),
                ("ray_active_indices", active_indices.buffer()),
                ("ray_materials", materials.buffer()),
            ]
            .into_iter()
            .filter_map(|(role, buffer)| buffer.map(|b| (role, b.clone())))
            .collect(),
            instances: instances_query.len(),
            blas_generation: blas_manager.generation,
            material_change_tick: material_assets.last_changed().get(),
            tlas_capacity: tlas.get().len(),
            tlas_active: tlas.get().iter().filter(|i| i.is_some()).count(),
            active_indices: _capture_indices
                .as_ref()
                .map_or_else(Vec::new, |_| active_indices.get().clone()),
        };
    }

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_tlas_command_encoder"),
    });
    use bevy_render::diagnostic::RecordDiagnostics;
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(&mut command_encoder, "scene/tlas_build");
    command_encoder.build_acceleration_structures(&[], [&*tlas]);
    span.end(&mut command_encoder);
    render_queue.submit([command_encoder.finish()]);

    let (dfg_view, dfg_sampler) = texture_assets
        .get(&dfg_lut.texture)
        .map(|img| (&img.texture_view, &img.sampler))
        .unwrap_or((
            &fallback_texture.d2.texture_view,
            &fallback_texture.d2.sampler,
        ));

    let sky_image = sky_light
        .image
        .as_ref()
        .and_then(|image| texture_assets.get(image.id()));
    let (sky_view, sky_sampler) = sky_image
        .map(|image| (&image.texture_view, &image.sampler))
        .unwrap_or((
            &fallback_texture.cube.texture_view,
            &fallback_texture.cube.sampler,
        ));
    // The weather of the planet's cloud layer, for cloud shadows.
    let weather_image = planet
        .as_deref()
        .filter(|planet| planet.cloud_coverage > 0.0)
        .and_then(|planet| planet.weather.as_ref())
        .and_then(|image| texture_assets.get(image.id()));
    let (weather_view, weather_sampler) = weather_image
        .map(|image| (&image.texture_view, &image.sampler))
        .unwrap_or((
            &fallback_texture.cube.texture_view,
            &fallback_texture.cube.sampler,
        ));
    // A storage buffer: wgpu refuses a uniform buffer in a bind group that
    // also holds binding arrays (the textures and mesh slabs above).
    sky_buffer.set(GpuSkyLight {
        intensity: sky_shader_intensity(sky_light.intensity, sky_image.is_some()),
        ray_max_distance: ray_settings.max_distance(),
        relative_ray_min: ray_settings.relative_min_distance(),
        material_transport_flags,
        medium: GpuLightMedium::new(
            planet.as_deref(),
            atmosphere.as_deref(),
            weather_image.is_some(),
        ),
    });
    sky_buffer.write_buffer_changed(&render_device, &render_queue);
    ring_buffer.set(rings.0.clone());
    ring_buffer.write_buffer_changed(&render_device, &render_queue);

    if scan_textures.is_empty() {
        scan_textures
            .vec
            .push(fallback_texture.d2_array.texture_view.deref());
        scan_samplers.push(fallback_texture.d2_array.sampler.deref());
    }
    raytracing_scene_bindings.bind_group = Some(render_device.create_bind_group(
        "raytracing_scene_bind_group",
        &pipeline_cache.get_bind_group_layout(&raytracing_scene_bindings.bind_group_layout),
        &BindGroupEntries::sequential((
            vertex_buffers.as_slice(),
            index_buffers.as_slice(),
            textures.as_slice(),
            samplers.as_slice(),
            materials.binding().unwrap(),
            tlas.as_binding(),
            transforms.binding().unwrap(),
            previous_frame_transforms.binding().unwrap(),
            geometry_ids.binding().unwrap(),
            material_ids.binding().unwrap(),
            light_sources.binding().unwrap(),
            directional_lights.binding().unwrap(),
            local_lights.binding().unwrap(),
            previous_frame_light_id_translations.binding().unwrap(),
            dfg_view,
            dfg_sampler,
            sky_view,
            sky_sampler,
            sky_buffer.binding().unwrap(),
            weather_view,
            weather_sampler,
            scan_textures.as_slice(),
            scan_samplers.as_slice(),
            detail_parameters.binding().unwrap(),
            ring_buffer.binding().unwrap(),
            raytracing_scene_bindings.history_regions.binding().unwrap(),
        )),
    ));
}

impl RaytracingSceneBindings {
    pub fn new() -> Self {
        Self {
            bind_group: None,
            bind_group_layout: BindGroupLayoutDescriptor::new(
                "raytracing_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        storage_buffer_read_only_sized(false, None).count(MAX_MESH_SLAB_COUNT),
                        texture_2d(TextureSampleType::Float { filterable: true })
                            .count(MAX_TEXTURE_COUNT),
                        sampler(SamplerBindingType::Filtering).count(MAX_TEXTURE_COUNT),
                        storage_buffer_read_only_sized(false, None),
                        acceleration_structure(),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                        texture_cube(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                        storage_buffer_read_only::<GpuSkyLight>(false),
                        texture_cube(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                        texture_2d_array(TextureSampleType::Float { filterable: true })
                            .count(MAX_SCAN_ATLASES),
                        sampler(SamplerBindingType::Filtering).count(MAX_SCAN_ATLASES),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only::<crate::rings::RingData>(false),
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
            previous_frame_light_entities: Vec::new(),
            settle_light_history: false,
            #[cfg(feature = "graphics_debug")]
            debug: super::graphics_debug::SceneSnapshot::default(),
            glass_entities: HashSet::default(),
            has_foliage: false,
            history_stable_radius: 0.0,
            history_generation: 0,
            history_regional: false,
            history_regions: StorageBuffer::default(),
        }
    }
}

impl Default for RaytracingSceneBindings {
    fn default() -> Self {
        Self::new()
    }
}

struct CachedBindingArray<T, I: Eq + Hash> {
    map: HashMap<I, u32>,
    vec: Vec<T>,
}

impl<T, I: Eq + Hash> CachedBindingArray<T, I> {
    fn new() -> Self {
        Self {
            map: HashMap::default(),
            vec: Vec::default(),
        }
    }

    fn push_if_absent(&mut self, item: T, item_id: I) -> (u32, bool) {
        let mut is_new = false;
        let i = *self.map.entry(item_id).or_insert_with(|| {
            is_new = true;
            let i = self.vec.len() as u32;
            self.vec.push(item);
            i
        });
        (i, is_new)
    }

    fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
}

type StorageBufferList<T> = StorageBuffer<Vec<T>>;

#[derive(ShaderType, Clone, Default, PartialEq)]
struct GpuInstanceGeometryIds {
    vertex_buffer_id: u32,
    vertex_buffer_offset: u32,
    index_buffer_id: u32,
    index_buffer_offset: u32,
    triangle_count: u32,
    light_probability: f32,
}

#[derive(ShaderType)]
struct GpuMaterial {
    normal_map_texture_id: u32,
    base_color_texture_id: u32,
    emissive_texture_id: u32,
    metallic_roughness_texture_id: u32,

    base_color: Vec3,
    perceptual_roughness: f32,
    emissive: Vec3,
    metallic: f32,
    // Upstream's vec3 padding, spent on the alpha handling the rays need.
    alpha_cutoff: f32,
    flags: u32,
    base_color_alpha: f32,
    reflectance: f32,
    emission_cone: Vec4,
    gaussian: Vec4,
}

/// `Material.flags` bits, mirrored in `raytracing_scene_bindings.wgsl`.
const MATERIAL_FLAG_OPAQUE: u32 = 1;
const MATERIAL_FLAG_ALPHA_MASK: u32 = 2;
const MATERIAL_FLAG_ALPHA_BLEND: u32 = 4;
const MATERIAL_FLAG_DIFFUSE_BLEND: u32 = 16;
const MATERIAL_FLAG_DOUBLE_SIDED: u32 = 8;

/// How the rays treat a material's alpha: exactly one of `OPAQUE`,
/// `ALPHA_MASK` (test the base colour alpha against `cutoff`) or
/// `ALPHA_BLEND` (explicit thin glass), plus `DOUBLE_SIDED`.
/// `DIFFUSE_BLEND` shades an ordinary surface with fractional coverage.
/// No mode bit means raster fallback: never bind it in the TLAS.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MaterialAlpha {
    flags: u32,
    cutoff: f32,
}

fn material_alpha(material: &StandardMaterial) -> MaterialAlpha {
    let (mode, cutoff) = match material.alpha_mode {
        AlphaMode::Opaque => (MATERIAL_FLAG_OPAQUE, 0.0),
        AlphaMode::Mask(cutoff) => (MATERIAL_FLAG_ALPHA_MASK, cutoff),
        AlphaMode::AlphaToCoverage => (MATERIAL_FLAG_ALPHA_MASK, 0.5),
        AlphaMode::Blend if material.specular_transmission == 1.0 => {
            (MATERIAL_FLAG_ALPHA_BLEND, 0.0)
        }
        AlphaMode::Blend if material.specular_transmission == 0.0 => {
            (MATERIAL_FLAG_DIFFUSE_BLEND, 0.0)
        }
        // Unsupported blend semantics stay in Bevy's raster pass, outside TLAS.
        AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Add | AlphaMode::Multiply => {
            (0, 0.0)
        }
    };
    let double_sided = if material.double_sided {
        MATERIAL_FLAG_DOUBLE_SIDED
    } else {
        0
    };
    // The high 16 flag bits hold unorm16
    // diffuse transmission. Only explicitly authored two-sided masks opt in.
    let transmission = if material.double_sided
        && matches!(material.alpha_mode, AlphaMode::Mask(_))
        && material.diffuse_transmission.is_finite()
    {
        (material.diffuse_transmission.clamp(0.0, 1.0) * 65535.0).round() as u32
    } else {
        0
    };
    MaterialAlpha {
        flags: mode | double_sided | (transmission << 16),
        cutoff,
    }
}

#[derive(ShaderType, Default)]
struct GpuLightSource {
    kind: u32,
    id: u32,
    selection_probability: f32,
    alias_threshold: f32,
    alias_index: u32,
}

impl GpuLightSource {
    fn new_emissive_mesh_light(instance_id: u32, triangle_count: u32) -> GpuLightSource {
        if triangle_count > u16::MAX as u32 {
            panic!("Too many triangles ({triangle_count}) in an emissive mesh, maximum is 65535.");
        }

        Self {
            kind: triangle_count << 1,
            id: instance_id,
            ..Default::default()
        }
    }

    fn new_directional_light(directional_light_id: u32) -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_DIRECTIONAL,
            id: directional_light_id,
            ..Default::default()
        }
    }

    fn new_point_light(local_light_id: u32) -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_POINT,
            id: local_light_id,
            ..Default::default()
        }
    }

    fn new_spot_light(local_light_id: u32) -> GpuLightSource {
        Self {
            kind: LIGHT_SOURCE_KIND_SPOT,
            id: local_light_id,
            ..Default::default()
        }
    }
}

/// `LightSource.kind` values, mirrored in `raytracing_scene_bindings.wgsl`.
/// The low bit clear is an emissive mesh (upstream: the triangle count sits
/// above it); the low bit set is any other light, with the kind above it.
const LIGHT_SOURCE_KIND_DIRECTIONAL: u32 = 1;
const LIGHT_SOURCE_KIND_POINT: u32 = 3;
const LIGHT_SOURCE_KIND_SPOT: u32 = 5;

/// The smallest sphere a point or spot light is sampled as (metres). glTF
/// and Bevy's defaults give lights a radius of 0, and the sampler needs an
/// area; at a centimetre the light is a point to anything it can light.
const LOCAL_LIGHT_MIN_RADIUS: f32 = 0.01;

/// A point or spot light as a sphere light. Mirrors `LocalLight` in
/// `raytracing_scene_bindings.wgsl`.
#[derive(ShaderType, Default, Debug, Clone, Copy, PartialEq)]
struct GpuLocalLight {
    position: Vec3,
    radius: f32,
    /// Radiance of the sphere's surface: the light's intensity (candela, as
    /// `bevy_pbr` extracts it) over π r², so the sphere's intensity in every
    /// direction is the light's.
    radiance: Vec3,
    /// The sphere's area, 4π r²: the inverse pdf of a uniform area sample.
    inverse_pdf: f32,
    /// Spot cone axis (the transform's forward), unused for point lights.
    direction: Vec3,
    /// The raster path's cut-off; its smooth window is applied so the two
    /// paths agree by construction.
    range: f32,
    /// Filament's cone: cos of the outer and inner angles. -1 for both means
    /// no cone (a point light).
    cos_outer: f32,
    cos_inner: f32,
    _padding: bevy_math::Vec2,
}

impl GpuLocalLight {
    fn new(light: &ExtractedPointLight) -> Self {
        let radius = light.radius.max(LOCAL_LIGHT_MIN_RADIUS);
        let area = 4.0 * PI * radius * radius;
        let radiance = light.color.to_vec3() * (light.intensity / (PI * radius * radius));
        let (cos_inner, cos_outer) = match light.spot_light_angles {
            Some((inner, outer)) => (cos(inner), cos(outer)),
            None => (-1.0, -1.0),
        };
        Self {
            position: light.transform.translation(),
            radius,
            radiance,
            inverse_pdf: area,
            direction: light.transform.forward().into(),
            range: light.range,
            cos_outer,
            cos_inner,
            _padding: bevy_math::Vec2::ZERO,
        }
    }
}

#[derive(ShaderType, Default, PartialEq)]
struct GpuDirectionalLight {
    direction_to_light: Vec3,
    cos_theta_max: f32,
    luminance: Vec3,
    inverse_pdf: f32,
}

impl GpuDirectionalLight {
    fn new(directional_light: &ExtractedDirectionalLight) -> Self {
        let cos_theta_max = cos(directional_light.sun_disk_angular_size / 2.0);
        let solid_angle = TAU * (1.0 - cos_theta_max);
        let luminance =
            (directional_light.color.to_vec3() * directional_light.illuminance) / solid_angle;

        Self {
            direction_to_light: directional_light.transform.back().into(),
            cos_theta_max,
            luminance,
            inverse_pdf: solid_angle,
        }
    }
}

/// Mirrors `SkyLight` in `raytracing_scene_bindings.wgsl`.
#[derive(ShaderType, Default)]
struct GpuSkyLight {
    intensity: f32,
    ray_max_distance: f32,
    relative_ray_min: f32,
    material_transport_flags: u32,
    medium: GpuLightMedium,
}

/// Mirrors `LightMedium` in `light_medium.wgsl`. Metres.
#[derive(ShaderType, Clone, Copy, Debug, PartialEq, Default)]
struct GpuLightMedium {
    centre: Vec3,
    radius: f32,
    rayleigh: Vec3,
    top: f32,
    ozone: Vec3,
    mie: f32,
    cloud: Vec4,
    physical: [Vec4; 5],
}

/// Extinction of Earth's air at the ground, per metre: the atmosphere pass's
/// Rayleigh and ozone coefficients, and its Mie scattering over a
/// single-scattering albedo of 0.9.
const RAYLEIGH_EXTINCTION: Vec3 = Vec3::new(5.802e-6, 13.558e-6, 33.1e-6);
const OZONE_ABSORPTION: Vec3 = Vec3::new(0.65e-6, 1.881e-6, 0.085e-6);
const MIE_EXTINCTION: f32 = 3.996e-6 / 0.9;
/// Share of cloud extinction left after the droplets' forward peak is
/// counted as unscattered: the atmosphere pass's `CLOUD_DRAINE_WEIGHT`.
const CLOUD_SCALED_SHARE: f32 = 0.498_159;

impl GpuLightMedium {
    /// The air directional light crosses: that of the planetary atmosphere,
    /// when the scene has a valid one.
    fn new(
        planet: Option<&crate::atmosphere::PlanetaryAtmosphere>,
        atmosphere: Option<&crate::atmosphere::AtmosphereState>,
        weather_bound: bool,
    ) -> Self {
        let none = Self {
            centre: Vec3::ZERO,
            radius: 0.0,
            rayleigh: Vec3::ZERO,
            top: 0.0,
            ozone: Vec3::ZERO,
            mie: 0.0,
            cloud: Vec4::ZERO,
            physical: [Vec4::ZERO; 5],
        };
        let (Some(planet), Some(atmosphere)) = (planet, atmosphere) else {
            return none;
        };
        if planet.validate().is_err() || atmosphere.validate().is_err() {
            return none;
        }
        Self {
            physical: atmosphere.physical.map_or(
                [Vec4::ZERO; 5],
                crate::atmosphere::PhysicalAtmosphere::gpu_fields,
            ),
            centre: planet.world_centre,
            radius: planet.radius,
            rayleigh: RAYLEIGH_EXTINCTION * atmosphere.medium.rayleigh,
            top: planet.radius + planet.height,
            ozone: OZONE_ABSORPTION * atmosphere.medium.ozone,
            mie: MIE_EXTINCTION * atmosphere.medium.mie,
            // Clouds cast shadows when a weather map gives them shape; the
            // built-in noise cover has no map to look up.
            cloud: Vec4::new(
                planet.radius + planet.cloud_base,
                planet.radius + planet.cloud_top,
                planet.cloud_extinction * CLOUD_SCALED_SHARE,
                f32::from(weather_bound),
            ),
        }
    }
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_transform::components::GlobalTransform;
    use bevy_transform::components::Transform;

    #[test]
    fn history_scope_uses_authored_bounds_and_rejects_unknown_or_emitting_sources() {
        use bevy_asset::{Assets, RenderAssetUsages};
        let mut meshes = Assets::<Mesh>::default();
        let mesh = meshes.add(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD,
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[-1.0f32; 3], [1.0; 3]]),
        );
        let mut source = Assets::<StandardMaterial>::default();
        let material = source.add(StandardMaterial::default());
        let entity = bevy_ecs::world::World::new().spawn_empty().id();
        let input = RayInstanceInput {
            entity: entity.into(),
            mesh: mesh.id(),
            material: material.id(),
            transform: Affine3A::from_translation(Vec3::X * 1000.0),
            previous: Affine3A::IDENTITY,
        };
        let mut blas = BlasManager::default();
        let materials = StandardMaterialAssets::extract_resource(&source);
        assert_eq!(
            history_distance(&input, &blas, &materials),
            0.0,
            "unknown geometry"
        );
        blas.bounds.insert(
            mesh.id(),
            super::super::history::Bounds::from_mesh(meshes.get(&mesh).unwrap()).unwrap(),
        );
        let radius = history_distance(&input, &blas, &materials);
        assert!(
            radius > 998.0 && radius < 999.0,
            "full authored bounds, not root distance"
        );
        let mut settled = input.clone();
        settled.previous = settled.transform;
        assert!(
            same_occluder(&input, &settled),
            "previous-transform settling preserves history"
        );
        settled.transform.translation.x = 2.0;
        assert!(!same_occluder(&input, &settled));
        assert!(
            history_distance(&settled, &blas, &materials) < 1.0,
            "old and new positions both constrain validity"
        );
        settled.entity = bevy_ecs::world::World::new().spawn_empty().id().into();
        settled.mesh = AssetId::from(bevy_asset::uuid::Uuid::from_u128(7));
        assert!(!same_occluder(&input, &settled), "source replacement");
        source.get_mut(&material).unwrap().emissive = LinearRgba::rgb(1.0, 0.0, 0.0);
        assert_eq!(
            history_distance(
                &input,
                &blas,
                &StandardMaterialAssets::extract_resource(&source)
            ),
            0.0,
            "emitter changes are global"
        );
        source.remove(material.id());
        assert_eq!(
            history_distance(
                &input,
                &blas,
                &StandardMaterialAssets::extract_resource(&source)
            ),
            0.0,
            "missing material fails closed"
        );
    }

    #[test]
    fn foliage_flags_restrict_and_quantize_authored_transmission() {
        let mut material = StandardMaterial {
            diffuse_transmission: 0.5,
            double_sided: true,
            alpha_mode: AlphaMode::Mask(0.5),
            ..Default::default()
        };
        assert_eq!(material_alpha(&material).flags >> 16, 32768);
        assert_eq!(material_alpha(&material).flags & 0xffff, 10);
        for (value, expected) in [
            (0.0, 0),
            (1.0, 65535),
            (2.0, 65535),
            (-1.0, 0),
            (f32::NAN, 0),
            (f32::INFINITY, 0),
        ] {
            material.diffuse_transmission = value;
            assert_eq!(material_alpha(&material).flags >> 16, expected);
        }
        material.diffuse_transmission = 1.0;
        material.double_sided = false;
        assert_eq!(material_alpha(&material).flags >> 16, 0);
        material.double_sided = true;
        for mode in [
            AlphaMode::Opaque,
            AlphaMode::Blend,
            AlphaMode::AlphaToCoverage,
        ] {
            material.alpha_mode = mode;
            assert_eq!(material_alpha(&material).flags >> 16, 0);
        }
        assert_eq!(GpuMaterial::min_size().get(), 96);
    }

    #[test]
    fn sky_intensity_is_zero_without_a_bound_image() {
        assert_eq!(sky_shader_intensity(1.5, true), 1.5);
        assert_eq!(sky_shader_intensity(1.5, false), 0.0);
        assert_eq!(sky_shader_intensity(-2.0, true), 0.0, "no negative sky");
        assert_eq!(sky_shader_intensity(0.0, true), 0.0);
    }

    #[test]
    fn only_explicit_blend_transmission_is_thin_glass() {
        let mut material = StandardMaterial::default();
        for mode in [
            AlphaMode::Blend,
            AlphaMode::Premultiplied,
            AlphaMode::Add,
            AlphaMode::Multiply,
        ] {
            material.alpha_mode = mode;
            for transmission in [0.5, f32::NAN, f32::INFINITY] {
                material.specular_transmission = transmission;
                assert_eq!(material_alpha(&material).flags & 7, 0);
            }
            material.specular_transmission = 0.0;
            assert_eq!(
                material_alpha(&material).flags & 23,
                if mode == AlphaMode::Blend {
                    MATERIAL_FLAG_DIFFUSE_BLEND
                } else {
                    0
                }
            );
            material.specular_transmission = 1.0;
            assert_eq!(
                material_alpha(&material).flags & 7,
                if mode == AlphaMode::Blend {
                    MATERIAL_FLAG_ALPHA_BLEND
                } else {
                    0
                }
            );
        }
    }
    #[test]
    fn material_alpha_picks_one_mode_and_keeps_double_sidedness() {
        let mut material = StandardMaterial::default();
        assert_eq!(
            material_alpha(&material),
            MaterialAlpha {
                flags: MATERIAL_FLAG_OPAQUE,
                cutoff: 0.0
            }
        );

        material.alpha_mode = AlphaMode::Mask(0.35);
        material.double_sided = true;
        assert_eq!(
            material_alpha(&material),
            MaterialAlpha {
                flags: MATERIAL_FLAG_ALPHA_MASK | MATERIAL_FLAG_DOUBLE_SIDED,
                cutoff: 0.35
            }
        );

        material.alpha_mode = AlphaMode::AlphaToCoverage;
        assert_eq!(material_alpha(&material).cutoff, 0.5);

        for blended in [
            AlphaMode::Blend,
            AlphaMode::Premultiplied,
            AlphaMode::Add,
            AlphaMode::Multiply,
        ] {
            material.alpha_mode = blended;
            let alpha = material_alpha(&material);
            assert_eq!(alpha.flags & MATERIAL_FLAG_ALPHA_BLEND, 0);
            assert_eq!(
                alpha.flags & MATERIAL_FLAG_OPAQUE,
                0,
                "{blended:?} blocks no ray"
            );
        }
    }

    #[test]
    fn light_source_kinds_keep_the_low_bit_convention() {
        // Emissive meshes: triangle count in the upper bits, low bit clear;
        // the fork relies on this staying as upstream left it. Every other
        // light sets the low bit and picks its kind above it.
        assert_eq!(GpuLightSource::new_emissive_mesh_light(7, 12).kind, 24);
        assert_eq!(GpuLightSource::new_directional_light(0).kind, 1);
        assert_eq!(GpuLightSource::new_point_light(3).kind, 3);
        assert_eq!(GpuLightSource::new_point_light(3).id, 3);
        assert_eq!(GpuLightSource::new_spot_light(4).kind, 5);
    }

    fn extracted_light(lumens: f32, radius: f32, spot: Option<(f32, f32)>) -> ExtractedPointLight {
        ExtractedPointLight {
            color: LinearRgba::WHITE,
            // What bevy_pbr's extract_lights stores: candela.
            intensity: lumens / (4.0 * PI),
            range: 20.0,
            radius,
            transform: GlobalTransform::from(
                Transform::from_xyz(1.0, 2.0, 3.0).looking_to(Vec3::NEG_Y, Vec3::X),
            ),
            shadow_maps_enabled: false,
            contact_shadows_enabled: false,
            shadow_depth_bias: 0.0,
            shadow_normal_bias: 0.0,
            shadow_map_near_z: 0.1,
            spot_light_angles: spot,
            volumetric: false,
            soft_shadows_enabled: false,
            affects_lightmapped_mesh_diffuse: true,
        }
    }

    #[test]
    fn point_light_photometry_matches_the_raster_path() {
        // An 800 lm bulb: I = 800 / 4π cd. A sphere of uniform radiance L and
        // radius r has intensity L π r², so a receiver facing it at distance d
        // sees I / d², the raster path's value. The sampler's estimator is
        // L × area × cos / d² over the whole sphere, whose expectation on the
        // facing side is L × π r² / d² (a quarter of the area sees the
        // receiver on average), so L × area / 4 must equal I.
        let light = GpuLocalLight::new(&extracted_light(800.0, 0.05, None));
        let intensity = 800.0 / (4.0 * PI);
        assert_eq!(light.radius, 0.05);
        assert!((light.radiance.x * light.inverse_pdf / 4.0 - intensity).abs() < 1e-3);
        assert_eq!(light.radiance.x, light.radiance.z, "white light");
        assert_eq!(light.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(light.range, 20.0);
        assert_eq!(light.cos_outer, -1.0, "a point light has no cone");
        assert_eq!(light.cos_inner, -1.0);
    }

    #[test]
    fn zero_radius_lights_get_the_floor_radius() {
        // glTF lights arrive with radius 0; the sphere must stay finite and
        // its photometry unchanged.
        let light = GpuLocalLight::new(&extracted_light(800.0, 0.0, None));
        assert_eq!(light.radius, LOCAL_LIGHT_MIN_RADIUS);
        let intensity = 800.0 / (4.0 * PI);
        assert!((light.radiance.x * light.inverse_pdf / 4.0 - intensity).abs() < 1e-3);
        assert!(light.radiance.x.is_finite());
    }

    #[test]
    fn spot_light_carries_its_cone_and_direction() {
        let inner = 0.3f32;
        let outer = 0.5f32;
        let light = GpuLocalLight::new(&extracted_light(1000.0, 0.0, Some((inner, outer))));
        assert!((light.cos_outer - outer.cos()).abs() < 1e-6);
        assert!((light.cos_inner - inner.cos()).abs() < 1e-6);
        // The cone points along the transform's forward axis, as bevy_pbr.
        assert!((light.direction - Vec3::NEG_Y).length() < 1e-5);
        // Same candela conversion as a point light (Bevy divides spot lumens
        // by 4π too, so a lit patch keeps its brightness when a point light
        // becomes a spot).
        let intensity = 1000.0 / (4.0 * PI);
        assert!((light.radiance.x * light.inverse_pdf / 4.0 - intensity).abs() < 1e-3);
    }
}
