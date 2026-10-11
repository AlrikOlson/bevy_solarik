//! Shared ownership of a surface's raster material, ray proxy and texture handles.
//!
//! Keep a registration in a cache or bind it to an entity. Clones share ownership;
//! dropping a cache entry never deletes another consumer's assets. Definitions are
//! immutable: use a fresh base handle for a replacement and retain the old
//! registration until that replacement is ready. GPU readiness is a separate gate.
//! The legacy ray registries remain available; do not mutate managed entries.
use alloc::sync::{Arc, Weak};
use bevy_app::{App, Last, Plugin, PostUpdate};
use bevy_asset::{AssetId, Assets, Handle};
use bevy_ecs::{component::Component, entity::Entity, resource::Resource, world::World};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_platform::collections::HashMap;

use crate::{
    gaussian::{GaussianDielectric, GaussianMaterial, GaussianRayMaterials},
    lommel::{LommelMaterial, LommelRayMaterials},
    scene::collimated::RaytracingMaterial3d,
    surface_detail::{DetailedMaterial, DetailedRayMaterials, SurfaceDetail},
};

/// The same optical definition is installed for raster and ray consumers.
pub enum MaterialModel {
    Standard,
    Gaussian(GaussianDielectric),
    Lommel,
    Detailed(Box<SurfaceDetail>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationError {
    MissingBase,
    AlreadyRegistered,
}
impl core::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::MissingBase => "the base material must exist before registration",
            Self::AlreadyRegistered => "share the existing registration or use a fresh base handle",
        })
    }
}
impl core::error::Error for RegistrationError {}

#[derive(Debug)]
enum Raster {
    Standard,
    Gaussian(Handle<GaussianMaterial>),
    Lommel(Handle<LommelMaterial>),
    Detailed(Handle<DetailedMaterial>),
}
#[derive(Debug)]
struct Owned {
    base: Handle<StandardMaterial>,
    raster: Raster,
}

/// Cloneable ownership token, also retained on each bound entity.
#[derive(Component, Debug, Clone)]
pub struct MaterialRegistration(Arc<Owned>);

#[derive(Resource, Default)]
struct Registrations(HashMap<AssetId<StandardMaterial>, Weak<Owned>>);

impl MaterialRegistration {
    pub fn new(
        world: &mut World,
        base: Handle<StandardMaterial>,
        model: MaterialModel,
    ) -> Result<Self, RegistrationError> {
        let value = world
            .get_resource::<Assets<StandardMaterial>>()
            .and_then(|assets| assets.get(&base))
            .cloned()
            .ok_or(RegistrationError::MissingBase)?;
        initialize(world);
        collect(world);
        if Self::get(world, base.id()).is_some()
            || world
                .resource::<GaussianRayMaterials>()
                .0
                .contains_key(&base.id())
            || world
                .resource::<LommelRayMaterials>()
                .0
                .contains(&base.id())
            || world
                .resource::<DetailedRayMaterials>()
                .0
                .contains_key(&base.id())
        {
            return Err(RegistrationError::AlreadyRegistered);
        }
        let raster = create_raster(world, base.id(), value, model);
        let owned = Arc::new(Owned { base, raster });
        world
            .resource_mut::<Registrations>()
            .0
            .insert(owned.base.id(), Arc::downgrade(&owned));
        if let Some(ready) =
            world.get_resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
        {
            ready.request(owned.base.id().untyped());
        }
        Ok(Self(owned))
    }

    /// Acquire another owner without rewriting mappings or invalidating readiness.
    #[must_use]
    pub fn get(world: &World, id: AssetId<StandardMaterial>) -> Option<Self> {
        world
            .get_resource::<Registrations>()?
            .0
            .get(&id)?
            .upgrade()
            .map(Self)
    }

    #[must_use]
    pub fn base(&self) -> &Handle<StandardMaterial> {
        &self.0.base
    }

    #[must_use]
    pub fn detailed(&self) -> Option<&Handle<DetailedMaterial>> {
        match &self.0.raster {
            Raster::Detailed(handle) => Some(handle),
            _ => None,
        }
    }

    /// Replace all known surface shading components as one exclusive-world edit.
    /// Does not change geometry, transforms, visibility or GPU readiness.
    pub fn bind(&self, world: &mut World, entity: Entity) -> bool {
        let Ok(mut entity) = world.get_entity_mut(entity) else {
            return false;
        };
        entity.remove::<(
            MeshMaterial3d<StandardMaterial>,
            MeshMaterial3d<GaussianMaterial>,
            MeshMaterial3d<LommelMaterial>,
            MeshMaterial3d<DetailedMaterial>,
            RaytracingMaterial3d,
        )>();
        match &self.0.raster {
            Raster::Standard => {
                entity.insert(MeshMaterial3d(self.0.base.clone()));
            }
            Raster::Gaussian(handle) => {
                entity.insert(MeshMaterial3d(handle.clone()));
            }
            Raster::Lommel(handle) => {
                entity.insert(MeshMaterial3d(handle.clone()));
            }
            Raster::Detailed(handle) => {
                entity.insert(MeshMaterial3d(handle.clone()));
            }
        }
        if !matches!(self.0.raster, Raster::Standard) {
            entity.insert(RaytracingMaterial3d(self.0.base.clone()));
        }
        entity.insert(self.clone());
        true
    }
}

fn initialize(world: &mut World) {
    world.init_resource::<Registrations>();
    world.init_resource::<GaussianRayMaterials>();
    world.init_resource::<LommelRayMaterials>();
    world.init_resource::<DetailedRayMaterials>();
    world.init_resource::<Assets<GaussianMaterial>>();
    world.init_resource::<Assets<LommelMaterial>>();
    world.init_resource::<Assets<DetailedMaterial>>();
}

fn create_raster(
    world: &mut World,
    id: AssetId<StandardMaterial>,
    base: StandardMaterial,
    model: MaterialModel,
) -> Raster {
    match model {
        MaterialModel::Standard => Raster::Standard,
        MaterialModel::Gaussian(extension) => {
            world
                .resource_mut::<GaussianRayMaterials>()
                .0
                .insert(id, extension.clone());
            Raster::Gaussian(
                world
                    .resource_mut::<Assets<GaussianMaterial>>()
                    .add(GaussianMaterial { base, extension }),
            )
        }
        MaterialModel::Lommel => {
            world.resource_mut::<LommelRayMaterials>().0.insert(id);
            Raster::Lommel(
                world
                    .resource_mut::<Assets<LommelMaterial>>()
                    .add(LommelMaterial {
                        base,
                        extension: Default::default(),
                    }),
            )
        }
        MaterialModel::Detailed(extension) => {
            if extension.lommel.x != 0.0 {
                world.resource_mut::<LommelRayMaterials>().0.insert(id);
            }
            if let Some(mask) = &extension.gaussian_mask {
                world.resource_mut::<GaussianRayMaterials>().0.insert(
                    id,
                    GaussianDielectric {
                        parameters: extension.gaussian_parameters,
                        mask: mask.clone(),
                    },
                );
            }
            world
                .resource_mut::<DetailedRayMaterials>()
                .0
                .insert(id, *extension.clone());
            Raster::Detailed(world.resource_mut::<Assets<DetailedMaterial>>().add(
                DetailedMaterial {
                    base,
                    extension: *extension,
                },
            ))
        }
    }
}

/// Apply streamed `StandardMaterial` selections before render extraction.
pub fn sync(world: &mut World) {
    let pending: Vec<_> = world
        .query::<(Entity, &MeshMaterial3d<StandardMaterial>)>()
        .iter(world)
        .map(|(e, m)| (e, m.id()))
        .collect();
    for (entity, id) in pending {
        if let Some(registration) = MaterialRegistration::get(world, id) {
            // An unchanged standard binding must not produce scene/history deltas.
            if matches!(registration.0.raster, Raster::Standard)
                && world
                    .get::<MaterialRegistration>(entity)
                    .is_some_and(|bound| Arc::ptr_eq(&bound.0, &registration.0))
            {
                continue;
            }
            registration.bind(world, entity);
        } else if world.get::<MaterialRegistration>(entity).is_some() {
            world.entity_mut(entity).remove::<(
                MeshMaterial3d<GaussianMaterial>,
                MeshMaterial3d<LommelMaterial>,
                MeshMaterial3d<DetailedMaterial>,
                RaytracingMaterial3d,
                MaterialRegistration,
            )>();
        }
    }
}

/// Retire mappings after the last cache/entity owner drops. Asset and GPU
/// destruction stay with Bevy/wgpu and their outstanding strong references.
pub fn collect(world: &mut World) {
    let Some(registry) = world.get_resource::<Registrations>() else {
        return;
    };
    let retired: Vec<_> = registry
        .0
        .iter()
        .filter(|(_, owner)| owner.strong_count() == 0)
        .map(|(id, _)| *id)
        .collect();
    if retired.is_empty() {
        return;
    }
    for id in retired {
        world.resource_mut::<Registrations>().0.remove(&id);
        world.resource_mut::<GaussianRayMaterials>().0.remove(&id);
        world.resource_mut::<LommelRayMaterials>().0.remove(&id);
        world.resource_mut::<DetailedRayMaterials>().0.remove(&id);
        if let Some(ready) =
            world.get_resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
        {
            ready.release(id.untyped());
        }
    }
}

pub struct MaterialRegistrationPlugin;
impl Plugin for MaterialRegistrationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Registrations>()
            .add_systems(PostUpdate, sync)
            .add_systems(Last, collect);
    }
}
