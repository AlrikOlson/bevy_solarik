//! Immutable mesh/material products and guarded, exclusive-world publication.
//!
//! CPU preparation is not GPU readiness. Keep the previous representation while
//! polling a replacement, then commit before extraction. This publication epoch
//! is distinct from the scene's actually rendered generation/history. Physical
//! emptiness belongs to the producer and is never inferred from missing assets.
use alloc::sync::Arc;
use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{AssetEvent, Assets, Handle, RenderAssetUsages};
use bevy_camera::{
    primitives::{Aabb, MeshAabb},
    visibility::VisibilityRange,
};
use bevy_ecs::{
    component::Component, entity::Entity, message::MessageReader, resource::Resource,
    schedule::IntoScheduleConfigs, system::Res, world::World,
};
use bevy_material::AlphaMode;
use bevy_mesh::{Mesh, Mesh3d, PrimitiveTopology};
use bevy_pbr::{Lightmap, StandardMaterial};
use bevy_render::{
    Render, RenderApp, RenderSystems, extract_resource::ExtractResourcePlugin,
    scene_readiness::SceneGeometryReadiness,
};
use core::sync::atomic::Ordering;

mod pipelines;
pub use pipelines::RasterPipelineReadiness;

use crate::{material::MaterialRegistration, scene::RaytracingMesh3d};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductState {
    Prepared,
    Uploading,
    GpuReady,
    Active,
    Retiring,
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationError {
    InvalidGeometry,
    MissingMesh,
    MissingMaterial,
    MissingEntity,
    OwnerChanged,
    Stale,
    AlreadyPublished,
    EpochExhausted,
    UnsupportedRasterState,
}
impl core::fmt::Display for PublicationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidGeometry => "publication requires finite, nonempty ray-enabled triangles",
            Self::MissingMesh => "publication mesh asset is missing",
            Self::MissingMaterial => "publication material registration is missing",
            Self::MissingEntity => "publication entity is missing",
            Self::OwnerChanged => "the previous raster/ray/material owner changed",
            Self::Stale => "a newer publication request superseded this replacement",
            Self::AlreadyPublished => "this replacement was already published",
            Self::EpochExhausted => "publication epoch exhausted",
            Self::UnsupportedRasterState => "static publication does not support skin, morph, lightmap or visibility-range variants",
        })
    }
}
impl core::error::Error for PublicationError {}

/// Renderer delivery identity, unrelated to a world revision or simulation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicationEpoch(u64);
impl PublicationEpoch {
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}
#[derive(Resource, Default)]
struct Epochs(u64);
#[derive(Component)]
struct Wanted(PublicationEpoch);

#[derive(Debug)]
pub struct PreparedProduct {
    mesh: Mesh,
    material: MaterialRegistration,
    bounds: Aabb,
}
impl PreparedProduct {
    pub fn new(mesh: Mesh, material: MaterialRegistration) -> Result<Self, PublicationError> {
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|p| p.as_float3())
            .ok_or(PublicationError::InvalidGeometry)?;
        if !mesh.enable_raytracing
            || !mesh.asset_usage.contains(RenderAssetUsages::RENDER_WORLD)
            || mesh.primitive_topology() != PrimitiveTopology::TriangleList
            || positions.is_empty()
            || positions.iter().flatten().any(|v| !v.is_finite())
            || mesh.indices().is_none_or(|indices| {
                indices.len() < 3
                    || !indices.len().is_multiple_of(3)
                    || indices.iter().any(|index| index >= positions.len())
            })
        {
            return Err(PublicationError::InvalidGeometry);
        }
        if mesh.has_morph_targets()
            || mesh.contains_attribute(Mesh::ATTRIBUTE_JOINT_INDEX)
            || mesh.contains_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)
        {
            return Err(PublicationError::UnsupportedRasterState);
        }
        let bounds = mesh
            .compute_aabb()
            .ok_or(PublicationError::InvalidGeometry)?;
        if !bounds.center.is_finite() || !bounds.half_extents.is_finite() {
            return Err(PublicationError::InvalidGeometry);
        }
        Ok(Self {
            mesh,
            material,
            bounds,
        })
    }

    #[must_use]
    pub fn state(&self) -> ProductState {
        ProductState::Prepared
    }

    /// Request an immutable candidate, leaving all current drawing components intact.
    pub fn stage(self, world: &mut World, entity: Entity) -> Result<Replacement, PublicationError> {
        let previous = Previous::capture(world, entity)?;
        supported_entity(world, entity)?;
        let alpha = world
            .get_resource::<Assets<StandardMaterial>>()
            .and_then(|assets| assets.get(self.material.base()))
            .ok_or(PublicationError::MissingMaterial)?
            .alpha_mode;
        world.init_resource::<Epochs>();
        let epoch = PublicationEpoch(
            world
                .resource::<Epochs>()
                .0
                .checked_add(1)
                .ok_or(PublicationError::EpochExhausted)?,
        );
        let retain_cpu = self
            .mesh
            .asset_usage
            .contains(RenderAssetUsages::MAIN_WORLD);
        let mesh = world
            .get_resource_mut::<Assets<Mesh>>()
            .ok_or(PublicationError::MissingMesh)?
            .add(self.mesh);
        world.init_resource::<SceneGeometryReadiness>();
        let readiness = world.resource::<SceneGeometryReadiness>();
        if alpha == AlphaMode::Opaque {
            readiness.request(mesh.id().untyped());
        } else {
            readiness.request_alpha(mesh.id().untyped());
        }
        readiness.request(self.material.base().id().untyped());
        let pipeline = pipelines::request(world, entity, mesh.id(), self.material.raster_id());
        world.resource_mut::<Epochs>().0 = epoch.0;
        world.entity_mut(entity).insert(Wanted(epoch));
        Ok(Replacement {
            entity,
            epoch,
            previous,
            next: Product {
                mesh,
                material: self.material,
                bounds: self.bounds,
                retain_cpu,
                pipeline: Some(pipeline),
            },
            published: false,
        })
    }
}

#[derive(Debug, Clone)]
struct Product {
    mesh: Handle<Mesh>,
    material: MaterialRegistration,
    bounds: Aabb,
    retain_cpu: bool,
    pipeline: Option<Arc<pipelines::Request>>,
}
impl Product {
    fn mesh_available(&self, world: &World) -> bool {
        // Render-only extraction moves the CPU payload; a metadata entry may remain.
        !self.retain_cpu
            || world
                .get_resource::<Assets<Mesh>>()
                .is_some_and(|assets| assets.contains(self.mesh.id()))
    }
    fn is_ready(&self, world: &World) -> bool {
        if !self.mesh_available(world)
            || !world
                .get_resource::<Assets<StandardMaterial>>()
                .is_some_and(|a| a.contains(self.material.base().id()))
        {
            return false;
        }
        world
            .get_resource::<RasterPipelineReadiness>()
            .is_some_and(|ready| ready.ready(self.mesh.id()))
            && world
                .get_resource::<SceneGeometryReadiness>()
                .is_some_and(|ready| {
                    ready.ready(self.mesh.id().untyped())
                        && ready.ready(self.material.base().id().untyped())
                })
    }
}

/// Last committed owner. Scene extraction/history still decides what was rendered.
#[derive(Component, Debug, Clone)]
pub struct ActiveProduct {
    epoch: PublicationEpoch,
    product: Product,
}
impl ActiveProduct {
    /// Actual first-frame specialization: pending, zero if a pipeline was missing,
    /// or the number of compiled variants selected before the first visible draw.
    #[must_use]
    pub fn first_frame_raster_pipelines(&self) -> Option<usize> {
        self.product
            .pipeline
            .as_ref()?
            .first_frame
            .load(Ordering::Acquire)
            .checked_sub(1)
    }
    #[must_use]
    pub fn epoch(&self) -> PublicationEpoch {
        self.epoch
    }
    #[must_use]
    pub fn mesh(&self) -> &Handle<Mesh> {
        &self.product.mesh
    }
}

#[derive(Debug)]
struct Previous {
    epoch: Option<PublicationEpoch>,
    product: Option<Product>,
}
impl Previous {
    fn capture(world: &World, entity: Entity) -> Result<Self, PublicationError> {
        if world.get_entity(entity).is_err() {
            return Err(PublicationError::MissingEntity);
        }
        let epoch = world.get::<ActiveProduct>(entity).map(ActiveProduct::epoch);
        let product = match (
            world.get::<Mesh3d>(entity),
            world.get::<RaytracingMesh3d>(entity),
        ) {
            (None, None) if !crate::material::has_surface(world, entity) && epoch.is_none() => None,
            (Some(raster), Some(ray)) if raster.0 == ray.0 => {
                let material = world
                    .get::<MaterialRegistration>(entity)
                    .filter(|material| material.is_bound(world, entity))
                    .ok_or(PublicationError::OwnerChanged)?
                    .clone();
                let bounds = world
                    .get::<Aabb>(entity)
                    .copied()
                    .or_else(|| {
                        world
                            .get_resource::<Assets<Mesh>>()?
                            .get(&raster.0)?
                            .compute_aabb()
                    })
                    .ok_or(PublicationError::InvalidGeometry)?;
                Some(Product {
                    mesh: raster.0.clone(),
                    material,
                    bounds,
                    pipeline: world
                        .get::<ActiveProduct>(entity)
                        .and_then(|active| active.product.pipeline.clone()),
                    retain_cpu: world.get::<ActiveProduct>(entity).map_or_else(
                        || {
                            world
                                .get_resource::<Assets<Mesh>>()
                                .and_then(|assets| assets.get(&raster.0))
                                .is_some_and(|mesh| {
                                    mesh.asset_usage.contains(RenderAssetUsages::MAIN_WORLD)
                                })
                        },
                        |active| active.product.retain_cpu,
                    ),
                })
            }
            _ => return Err(PublicationError::OwnerChanged),
        };
        Ok(Self { epoch, product })
    }
    fn matches(&self, world: &World, entity: Entity) -> bool {
        if world.get::<ActiveProduct>(entity).map(ActiveProduct::epoch) != self.epoch {
            return false;
        }
        match &self.product {
            Some(old) => {
                world.get::<Mesh3d>(entity).is_some_and(|m| m.0 == old.mesh)
                    && world
                        .get::<RaytracingMesh3d>(entity)
                        .is_some_and(|m| m.0 == old.mesh)
                    && old.material.is_bound(world, entity)
            }
            None => {
                world.get::<Mesh3d>(entity).is_none()
                    && world.get::<RaytracingMesh3d>(entity).is_none()
                    && !crate::material::has_surface(world, entity)
            }
        }
    }
}

/// Dropping/cancelling a staged replacement releases its references, never the active owner.
#[derive(Debug)]
pub struct Replacement {
    entity: Entity,
    epoch: PublicationEpoch,
    previous: Previous,
    next: Product,
    published: bool,
}
impl Replacement {
    #[must_use]
    pub fn mesh(&self) -> &Handle<Mesh> {
        &self.next.mesh
    }
    #[must_use]
    pub fn material(&self) -> &MaterialRegistration {
        &self.next.material
    }
    /// Number of required variants observed as actual compiled GPU pipelines.
    #[must_use]
    pub fn compiled_raster_pipelines(&self) -> usize {
        self.next
            .pipeline
            .as_ref()
            .map_or(0, |p| p.compiled.load(Ordering::Acquire))
    }
    #[must_use]
    pub fn epoch(&self) -> PublicationEpoch {
        self.epoch
    }
    #[must_use]
    pub fn state(&self, world: &World) -> ProductState {
        if self.published {
            return if world
                .get::<ActiveProduct>(self.entity)
                .is_some_and(|p| p.epoch == self.epoch)
            {
                ProductState::Active
            } else {
                ProductState::Retiring
            };
        }
        if !world
            .get::<Wanted>(self.entity)
            .is_some_and(|w| w.0 == self.epoch)
        {
            ProductState::Stale
        } else if self.next.is_ready(world) {
            ProductState::GpuReady
        } else {
            ProductState::Uploading
        }
    }
    pub fn cancel(self) {}

    /// `None` means wait. Every error preserves the previous drawing components.
    pub fn try_publish(
        &mut self,
        world: &mut World,
    ) -> Result<Option<Published>, PublicationError> {
        if self.published {
            return Err(PublicationError::AlreadyPublished);
        }
        if world.get_entity(self.entity).is_err() {
            return Err(PublicationError::MissingEntity);
        }
        if !world
            .get::<Wanted>(self.entity)
            .is_some_and(|w| w.0 == self.epoch)
        {
            return Err(PublicationError::Stale);
        }
        if !self.previous.matches(world, self.entity) {
            return Err(PublicationError::OwnerChanged);
        }
        supported_entity(world, self.entity)?;
        if !self.next.mesh_available(world) {
            return Err(PublicationError::MissingMesh);
        }
        if !world
            .get_resource::<Assets<StandardMaterial>>()
            .is_some_and(|a| a.contains(self.next.material.base().id()))
        {
            return Err(PublicationError::MissingMaterial);
        }
        if !self.next.is_ready(world) {
            return Ok(None);
        }
        // No fallible work follows preflight. Required default raster components
        // introduced by RaytracingMesh3d are replaced by the complete registration.
        world.entity_mut(self.entity).insert((
            Mesh3d(self.next.mesh.clone()),
            RaytracingMesh3d(self.next.mesh.clone()),
            self.next.bounds,
            ActiveProduct {
                epoch: self.epoch,
                product: self.next.clone(),
            },
        ));
        let bound = self.next.material.bind(world, self.entity);
        assert!(bound, "exclusive publication retains its validated entity");
        self.published = true;
        if let Some(pipeline) = &self.next.pipeline {
            pipeline.published.store(true, Ordering::Release);
        }
        Ok(Some(Published {
            epoch: self.epoch,
            retired: self.previous.product.take().map(RetiringProduct),
        }))
    }
}

#[derive(Debug)]
pub struct Published {
    pub epoch: PublicationEpoch,
    pub retired: Option<RetiringProduct>,
}
/// An optional outstanding consumer. Asset/GPU destruction belongs to Bevy/wgpu.
#[derive(Debug)]
pub struct RetiringProduct(Product);
impl RetiringProduct {
    #[must_use]
    pub fn mesh(&self) -> &Handle<Mesh> {
        &self.0.mesh
    }
    #[must_use]
    pub fn state(&self) -> ProductState {
        ProductState::Retiring
    }
}

fn retire_meshes(
    mut events: MessageReader<AssetEvent<Mesh>>,
    readiness: Res<SceneGeometryReadiness>,
) {
    for event in events.read() {
        // Match Bevy's GPU lifetime: Removed also means render-only CPU extraction.
        if let AssetEvent::Unused { id } = event {
            readiness.release(id.untyped());
        }
    }
}
pub struct PublicationPlugin;
impl Plugin for PublicationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SceneGeometryReadiness>()
            .init_resource::<pipelines::Requests>()
            .init_resource::<RasterPipelineReadiness>()
            .add_message::<AssetEvent<Mesh>>()
            .add_systems(PostUpdate, retire_meshes);
    }

    fn finish(&self, app: &mut App) {
        if app.get_sub_app(RenderApp).is_some() {
            app.add_plugins(ExtractResourcePlugin::<pipelines::Requests>::default());
            app.sub_app_mut(RenderApp).add_systems(
                Render,
                pipelines::prepare_pipelines.in_set(RenderSystems::PrepareBindGroups),
            );
        }
    }
}

fn supported_entity(world: &World, entity: Entity) -> Result<(), PublicationError> {
    if world.get::<Lightmap>(entity).is_some()
        || world.get::<VisibilityRange>(entity).is_some()
        || world
            .get::<bevy_mesh::skinning::SkinnedMesh>(entity)
            .is_some()
        || world
            .get::<bevy_mesh::morph::MeshMorphWeights>(entity)
            .is_some()
    {
        return Err(PublicationError::UnsupportedRasterState);
    }
    Ok(())
}
