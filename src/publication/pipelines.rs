//! Prewarm static surface products through Bevy's production specializers.
//! A material bind group alone does not establish a compiled raster pipeline.
use alloc::sync::{Arc, Weak};
use bevy_asset::{AssetId, UntypedAssetId};
use bevy_core_pipeline::prepass::DeferredPrepass;
use bevy_ecs::{entity::Entity, resource::Resource, world::World};
use bevy_material::{
    AlphaMode, MaterialProperties, OpaqueRendererMethod,
    key::{ErasedMaterialPipelineKey, ErasedMeshPipelineKey},
};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_pbr::{
    LightKeyCache, MeshPipelineKey, PreparedMaterial, PrepassFragmentShader, PrepassVertexShader,
    RenderMaterialInstances, RenderMeshInstances, SpecializedMaterialPipelineCache,
    SpecializedPrepassMaterialPipelineCache, SpecializedShadowMaterialPipelineCache, ViewKeyCache,
    ViewKeyPrepassCache, alpha_mode_pipeline_key,
};
use bevy_render::{
    erased_render_asset::ErasedRenderAssets,
    extract_resource::ExtractResource,
    mesh::RenderMesh,
    render_asset::RenderAssets,
    render_resource::PipelineCache,
    scene_readiness::SceneGeometryReadiness,
    sync_world::MainEntity,
    view::{ExtractedView, Msaa, RenderVisibleEntities},
};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Renderer receipts for each immutable candidate's raster pipelines.
///
/// Like `SceneGeometryReadiness`, the renderer owns publication of these receipts.
/// CPU preparation and asset handles must never mark a production candidate ready.
#[derive(Resource, Default, Clone)]
pub struct RasterPipelineReadiness(SceneGeometryReadiness);
impl RasterPipelineReadiness {
    #[must_use]
    pub fn ready(&self, mesh: AssetId<Mesh>) -> bool {
        self.0.ready(mesh.untyped())
    }

    /// Publish the result of checking every required GPU pipeline for this request.
    /// Also permits controlled renderer substitution in headless ownership tests.
    pub fn publish(&self, mesh: AssetId<Mesh>, ready: bool) {
        self.0.publish(mesh.untyped(), ready);
    }
}

pub(super) struct Request {
    mesh: AssetId<Mesh>,
    entity: MainEntity,
    material: UntypedAssetId,
    readiness: RasterPipelineReadiness,
    pub(super) published: AtomicBool,
    pub(super) compiled: AtomicUsize,
    pub(super) first_frame: AtomicUsize,
}
impl core::fmt::Debug for Request {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipelineRequest")
            .field("mesh", &self.mesh)
            .field("material", &self.material)
            .finish_non_exhaustive()
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        self.readiness.0.release(self.mesh.untyped());
    }
}

#[derive(Resource, Default, Clone, ExtractResource)]
pub(super) struct Requests(Vec<Weak<Request>>);

pub(super) fn request(
    world: &mut World,
    entity: Entity,
    mesh: AssetId<Mesh>,
    material: UntypedAssetId,
) -> Arc<Request> {
    world.init_resource::<RasterPipelineReadiness>();
    world.init_resource::<Requests>();
    let readiness = world.resource::<RasterPipelineReadiness>().clone();
    readiness.0.request(mesh.untyped());
    let request = Arc::new(Request {
        mesh,
        entity: entity.into(),
        material,
        readiness,
        published: AtomicBool::new(false),
        compiled: AtomicUsize::new(0),
        first_frame: AtomicUsize::new(0),
    });
    let mut requests = world.resource_mut::<Requests>();
    requests.0.retain(|weak| {
        weak.upgrade()
            .is_some_and(|r| r.first_frame.load(Ordering::Acquire) == 0)
    });
    requests.0.push(Arc::downgrade(&request));
    request
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PassKey {
    prepass: bool,
    mesh: MeshPipelineKey,
}
fn reads_material(properties: &MaterialProperties) -> bool {
    properties.get_shader(PrepassVertexShader).is_some()
        || properties.get_shader(PrepassFragmentShader).is_some()
}
fn prepass_key(
    mut key: MeshPipelineKey,
    properties: &MaterialProperties,
    deferred: bool,
) -> Option<MeshPipelineKey> {
    if !properties.prepass_enabled
        || properties.reads_view_transmission_texture
        || !matches!(
            properties.alpha_mode,
            AlphaMode::Opaque | AlphaMode::AlphaToCoverage | AlphaMode::Mask(_)
        )
    {
        return None;
    }
    key |= alpha_mode_pipeline_key(
        properties.alpha_mode,
        &Msaa::from_samples(key.msaa_samples()),
    );
    key.set(
        MeshPipelineKey::DEFERRED_PREPASS,
        deferred && properties.render_method == OpaqueRendererMethod::Deferred,
    );
    key.set(
        MeshPipelineKey::PREPASS_READS_MATERIAL,
        reads_material(properties),
    );
    Some(key)
}
fn shadow_key(mut key: MeshPipelineKey, properties: &MaterialProperties) -> MeshPipelineKey {
    key.set(
        MeshPipelineKey::MAY_DISCARD,
        matches!(
            properties.alpha_mode,
            AlphaMode::Mask(_)
                | AlphaMode::Blend
                | AlphaMode::Premultiplied
                | AlphaMode::Add
                | AlphaMode::AlphaToCoverage
        ),
    );
    key.set(
        MeshPipelineKey::PREPASS_READS_MATERIAL,
        reads_material(properties),
    );
    key
}

fn required_keys(
    world: &mut World,
    mesh: MeshPipelineKey,
    properties: &MaterialProperties,
) -> Vec<PassKey> {
    let views: Vec<_> = world
        .query::<(&ExtractedView, Option<&DeferredPrepass>)>()
        .iter(world)
        .map(|(view, deferred)| (view.retained_view_entity, deferred.is_some()))
        .collect();
    let mut keys = Vec::new();
    for (view, deferred) in views {
        if let Some(key) = world
            .get_resource::<ViewKeyCache>()
            .and_then(|cache| cache.get(&view))
        {
            keys.push(PassKey {
                prepass: false,
                mesh: *key
                    | mesh
                    | properties
                        .mesh_pipeline_key_bits
                        .downcast::<MeshPipelineKey>()
                    | alpha_mode_pipeline_key(
                        properties.alpha_mode,
                        &Msaa::from_samples(key.msaa_samples()),
                    ),
            });
        }
        if let Some(key) = world
            .get_resource::<ViewKeyPrepassCache>()
            .and_then(|cache| cache.get(&view))
            && (key.intersects(MeshPipelineKey::ALL_PREPASS_BITS) || deferred)
            && let Some(key) = prepass_key(*key | mesh, properties, deferred)
        {
            keys.push(PassKey {
                prepass: true,
                mesh: key,
            });
        }
        if properties.shadows_enabled
            && let Some(key) = world
                .get_resource::<LightKeyCache>()
                .and_then(|cache| cache.get(&view))
        {
            keys.push(PassKey {
                prepass: true,
                mesh: shadow_key(*key | mesh, properties),
            });
        }
    }
    let mut unique = Vec::new();
    for key in keys {
        if !unique.contains(&key) {
            unique.push(key);
        }
    }
    unique
}

fn prepare(world: &mut World, request: &Request) -> Option<usize> {
    let mesh = world
        .get_resource::<RenderAssets<RenderMesh>>()?
        .get(request.mesh)?;
    let bits = MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());
    let layout = mesh.layout.clone();
    let properties = world
        .get_resource::<ErasedRenderAssets<PreparedMaterial>>()?
        .get(request.material)?
        .properties
        .clone();
    let keys = required_keys(world, bits, &properties);
    // With no current view, there is no evidence about required pipeline variants.
    if keys.is_empty() || !world.contains_resource::<PipelineCache>() {
        return None;
    }
    let mut ready = true;
    for key in &keys {
        let specialize = if key.prepass {
            properties.prepass_specialize
        } else {
            properties.base_specialize
        };
        let specialize = specialize?;
        let key = ErasedMaterialPipelineKey {
            type_id: request.material.type_id(),
            mesh_key: ErasedMeshPipelineKey::new(key.mesh),
            material_key: properties.material_key.clone(),
        };
        let Ok(id) = specialize(world, key, &layout, &properties) else {
            return None;
        };
        // The specializer deduplicates. A queued ID is not a compiled GPU object.
        ready &= world
            .resource::<PipelineCache>()
            .get_render_pipeline(id)
            .is_some();
    }
    ready.then_some(keys.len())
}

pub(super) fn prepare_pipelines(world: &mut World) {
    let Some(requests) = world.get_resource::<Requests>() else {
        return;
    };
    let requests: Vec<_> = requests
        .0
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|r| r.first_frame.load(Ordering::Acquire) == 0)
        .collect();
    for request in requests {
        if request.published.load(Ordering::Acquire) {
            if let Some(count) = first_frame(world, &request) {
                request.first_frame.store(count + 1, Ordering::Release);
            }
            continue;
        }
        let count = prepare(world, &request).unwrap_or(0);
        request.compiled.store(count, Ordering::Release);
        request.readiness.publish(request.mesh, count > 0);
    }
}

// Observe the actual production specialization caches after extraction selected
// this replacement, before its first draw. Zero reports a queued/failed pipeline.
fn first_frame(world: &mut World, request: &Request) -> Option<usize> {
    let visible = world
        .query::<&RenderVisibleEntities>()
        .iter(world)
        .any(|view| {
            view.get::<Mesh3d>().is_some_and(|entities| {
                entities
                    .iter_visible()
                    .any(|(_, entity)| *entity == request.entity)
            })
        });
    if !visible {
        return None;
    }
    let mesh = world
        .get_resource::<RenderMeshInstances>()?
        .render_mesh_queue_data(request.entity)?;
    let material = world
        .get_resource::<RenderMaterialInstances>()?
        .instances
        .get(&request.entity)?;
    if mesh.mesh_asset_id() != request.mesh || material.asset_id != request.material {
        return None;
    }
    let mut ids = Vec::new();
    if let Some(cache) = world.get_resource::<SpecializedMaterialPipelineCache>() {
        ids.extend(
            cache
                .values()
                .filter_map(|view| view.get(&request.entity).copied()),
        );
    }
    if let Some(cache) = world.get_resource::<SpecializedPrepassMaterialPipelineCache>() {
        ids.extend(
            cache
                .values()
                .filter_map(|view| view.get(&request.entity).map(|(_, id, _)| *id)),
        );
    }
    if let Some(cache) = world.get_resource::<SpecializedShadowMaterialPipelineCache>() {
        ids.extend(
            cache
                .values()
                .filter_map(|view| view.get(&request.entity).map(|(id, _)| *id)),
        );
    }
    if ids.is_empty() {
        return Some(0);
    }
    let cache = world.get_resource::<PipelineCache>()?;
    Some(
        if ids
            .iter()
            .all(|id| cache.get_render_pipeline(*id).is_some())
        {
            ids.len()
        } else {
            0
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_render_resources_revoke_readiness_and_last_owner_retires_request() {
        let mut world = World::new();
        let mesh = AssetId::<Mesh>::default();
        let material = AssetId::<bevy_pbr::StandardMaterial>::default().untyped();
        let request = request(&mut world, Entity::PLACEHOLDER, mesh, material);
        let readiness = world.resource::<RasterPipelineReadiness>().clone();
        readiness.publish(mesh, true);
        prepare_pipelines(&mut world);
        assert!(!readiness.ready(mesh));
        assert_eq!(request.compiled.load(Ordering::Acquire), 0);
        let other = request.clone();
        drop(request);
        assert!(readiness.0.requested::<Mesh>().contains(&mesh));
        drop(other);
        assert!(!readiness.0.requested::<Mesh>().contains(&mesh));
    }
}
