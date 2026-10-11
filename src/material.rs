//! Shared ownership of a surface's raster material, ray proxy and texture handles.
//!
//! Keep a registration in a cache or bind it to an entity. Clones share ownership;
//! dropping a cache entry never deletes another consumer's assets. Definitions are
//! immutable: use a fresh base handle for a replacement and retain the old
//! registration until that replacement is ready. GPU readiness is a separate gate.
//! The legacy ray registries remain available; do not mutate managed entries.
use alloc::sync::{Arc, Weak};
use bevy_app::{App, Last, Plugin, PostUpdate};
use bevy_asset::{AssetId, Assets, Handle, UntypedAssetId};
use bevy_ecs::{component::Component, entity::Entity, resource::Resource, world::World};
use bevy_image::Image;
use bevy_pbr::{MeshMaterial3d, RenderMaterialBindings, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{
    RenderApp,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    texture::GpuImage,
};

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
    images: Vec<Handle<Image>>,
}

/// Cloneable ownership token, also retained on each bound entity.
#[derive(Component, Debug, Clone)]
pub struct MaterialRegistration(Arc<Owned>);

#[derive(Resource, Default, Clone, ExtractResource)]
pub(crate) struct Registrations(HashMap<AssetId<StandardMaterial>, Weak<Owned>>);
impl Registrations {
    pub(crate) fn gpu_ready(
        &self,
        id: AssetId<StandardMaterial>,
        bindings: &RenderMaterialBindings,
        images: &RenderAssets<GpuImage>,
    ) -> bool {
        self.0.get(&id).and_then(Weak::upgrade).is_none_or(|owned| {
            MaterialRegistration(owned).dependencies_ready(
                |id| bindings.contains_key(&id),
                |id| images.get(id).is_some(),
            )
        })
    }
}

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
        let images = match &model {
            MaterialModel::Gaussian(extension) => vec![extension.mask.clone()],
            MaterialModel::Detailed(extension) => [
                &extension.colour,
                &extension.detail,
                &extension.coverage0,
                &extension.coverage1,
                &extension.meso_colour,
                &extension.meso_detail,
            ]
            .into_iter()
            .chain(extension.gaussian_mask.iter())
            .cloned()
            .collect(),
            MaterialModel::Standard | MaterialModel::Lommel => Vec::new(),
        };
        let raster = create_raster(world, base.id(), value, model);
        let owned = Arc::new(Owned {
            base,
            raster,
            images,
        });
        world
            .resource_mut::<Registrations>()
            .0
            .insert(owned.base.id(), Arc::downgrade(&owned));
        if let Some(ready) =
            world.get_resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
        {
            ready.request(owned.base.id().untyped());
            ready.revoke(owned.base.id().untyped());
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

    fn raster_id(&self) -> UntypedAssetId {
        match &self.0.raster {
            Raster::Standard => self.0.base.id().untyped(),
            Raster::Gaussian(handle) => handle.id().untyped(),
            Raster::Lommel(handle) => handle.id().untyped(),
            Raster::Detailed(handle) => handle.id().untyped(),
        }
    }
    fn dependencies_ready(
        &self,
        binding: impl Fn(UntypedAssetId) -> bool,
        image: impl Fn(AssetId<Image>) -> bool,
    ) -> bool {
        binding(self.raster_id()) && self.0.images.iter().all(|handle| image(handle.id()))
    }
    pub(crate) fn is_bound(&self, world: &World, entity: Entity) -> bool {
        if !world
            .get::<Self>(entity)
            .is_some_and(|bound| Arc::ptr_eq(&bound.0, &self.0))
        {
            return false;
        }
        let raster_ids = [
            world
                .get::<MeshMaterial3d<StandardMaterial>>(entity)
                .map(|m| m.id().untyped()),
            world
                .get::<MeshMaterial3d<GaussianMaterial>>(entity)
                .map(|m| m.id().untyped()),
            world
                .get::<MeshMaterial3d<LommelMaterial>>(entity)
                .map(|m| m.id().untyped()),
            world
                .get::<MeshMaterial3d<DetailedMaterial>>(entity)
                .map(|m| m.id().untyped()),
        ];
        let raster_matches = raster_ids.into_iter().flatten().eq([self.raster_id()]);
        let proxy = world.get::<RaytracingMaterial3d>(entity).map(|m| m.0.id());
        raster_matches
            && proxy
                == match self.0.raster {
                    Raster::Standard => None,
                    _ => Some(self.0.base.id()),
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

pub(crate) fn has_surface(world: &World, entity: Entity) -> bool {
    world
        .get::<MeshMaterial3d<StandardMaterial>>(entity)
        .is_some()
        || world
            .get::<MeshMaterial3d<GaussianMaterial>>(entity)
            .is_some()
        || world
            .get::<MeshMaterial3d<LommelMaterial>>(entity)
            .is_some()
        || world
            .get::<MeshMaterial3d<DetailedMaterial>>(entity)
            .is_some()
        || world.get::<RaytracingMaterial3d>(entity).is_some()
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
    fn finish(&self, app: &mut App) {
        if app.get_sub_app(RenderApp).is_some() {
            app.add_plugins(ExtractResourcePlugin::<Registrations>::default());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_detail::DetailCoordinates;
    use bevy_math::{DVec3, Vec4};

    #[test]
    fn custom_material_waits_for_its_raster_binding_and_every_optical_image() {
        let mut world = World::new();
        world.init_resource::<Assets<StandardMaterial>>();
        world.init_resource::<Assets<Image>>();
        let maps: [Handle<Image>; 7] =
            core::array::from_fn(|_| world.resource_mut::<Assets<Image>>().add(Image::default()));
        let base = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let base_id = base.id().untyped();
        let registration = MaterialRegistration::new(
            &mut world,
            base,
            MaterialModel::Detailed(Box::new(SurfaceDetail {
                coordinates: DetailCoordinates::new(DVec3::ZERO, [1.; 8]).unwrap(),
                colour: maps[0].clone(),
                detail: maps[1].clone(),
                coverage0: maps[2].clone(),
                coverage1: maps[3].clone(),
                meso_colour: maps[4].clone(),
                meso_detail: maps[5].clone(),
                gaussian_mask: Some(maps[6].clone()),
                gaussian_parameters: Vec4::ONE,
                lommel: Vec4::ZERO,
            })),
        )
        .unwrap();
        assert!(
            !registration.dependencies_ready(|id| id == base_id, |_| true),
            "base ray material alone is not the custom raster material"
        );
        for missing in &maps {
            assert!(
                !registration.dependencies_ready(|_| true, |id| id != missing.id()),
                "each scan, coverage and dielectric mask must have a GPU image"
            );
        }
        assert!(registration.dependencies_ready(|_| true, |_| true));
    }
}
