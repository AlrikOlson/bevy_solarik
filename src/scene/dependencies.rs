//! History depends on assets actually used by the scene, not unrelated arrivals.
use super::{
    binder::RayInstanceInput, blas::BlasManager, extract::StandardMaterialAssets, history::Bounds,
};
use bevy_asset::{AssetId, Handle};
use bevy_image::Image;
use bevy_mesh::Mesh;
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{SamplerId, TextureId},
    texture::GpuImage,
};
use std::hash::{Hash, Hasher};

type ImageState = Option<(TextureId, SamplerId)>;
#[derive(Clone, PartialEq)]
struct MaterialState {
    hash: u64,
    images: Vec<ImageState>,
    emitter: bool,
    ready: bool,
}
#[derive(Clone, Copy, PartialEq)]
struct MeshState {
    revision: u64,
    ready: bool,
    bounds: Option<Bounds>,
}
#[derive(Default)]
pub(super) struct Dependencies {
    meshes: HashMap<AssetId<Mesh>, MeshState>,
    materials: HashMap<AssetId<StandardMaterial>, MaterialState>,
    global_images: Vec<ImageState>,
}
#[derive(Default)]
pub(super) struct Changes {
    pub meshes: HashSet<AssetId<Mesh>>,
    pub materials: HashSet<AssetId<StandardMaterial>>,
    pub global: bool,
    old_bounds: HashMap<AssetId<Mesh>, Option<Bounds>>,
    old_emitters: HashSet<AssetId<StandardMaterial>>,
}
impl Changes {
    pub fn affects(&self, input: &RayInstanceInput) -> bool {
        self.meshes.contains(&input.mesh) || self.materials.contains(&input.material)
    }
    pub fn old_bounds(&self, input: &RayInstanceInput, fallback: Option<Bounds>) -> Option<Bounds> {
        if self.old_emitters.contains(&input.material) {
            return None;
        }
        self.old_bounds
            .get(&input.mesh)
            .copied()
            .unwrap_or(fallback)
    }
}
fn image_state(images: &RenderAssets<GpuImage>, handle: Option<AssetId<Image>>) -> ImageState {
    handle
        .and_then(|id| images.get(id))
        .map(|image| (image.texture.id(), image.sampler.id()))
}
fn material_state(material: &StandardMaterial, images: &RenderAssets<GpuImage>) -> MaterialState {
    // Hash all StandardMaterial fields. Only unique used materials are visited
    // when scene inputs/resources change; this is not a per-instance operation.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    format!("{material:?}").hash(&mut hash);
    let textures = [
        &material.base_color_texture,
        &material.normal_map_texture,
        &material.emissive_texture,
        &material.metallic_roughness_texture,
    ];
    let ready = textures.iter().all(|texture| {
        texture
            .as_ref()
            .is_none_or(|t| images.get(t.id()).is_some())
    });
    MaterialState {
        hash: hash.finish(),
        images: textures
            .into_iter()
            .map(|t| image_state(images, t.as_ref().map(Handle::id)))
            .collect(),
        emitter: material.emissive.red != 0.0
            || material.emissive.green != 0.0
            || material.emissive.blue != 0.0,
        ready,
    }
}
impl Dependencies {
    pub fn observe(
        &mut self,
        inputs: impl Iterator<Item = RayInstanceInput>,
        blas: &BlasManager,
        materials: &StandardMaterialAssets,
        images: &RenderAssets<GpuImage>,
        extensions: (
            &crate::surface_detail::DetailedRayMaterials,
            &crate::gaussian::GaussianRayMaterials,
        ),
        global_images: [Option<AssetId<Image>>; 3],
    ) -> Changes {
        let mut mesh_ids: HashSet<AssetId<Mesh>> = HashSet::default();
        let mut material_ids: HashSet<AssetId<StandardMaterial>> = HashSet::default();
        for input in inputs {
            mesh_ids.insert(input.mesh);
            material_ids.insert(input.material);
        }
        let mut changes = Changes::default();
        let globals = global_images
            .into_iter()
            .map(|id| image_state(images, id))
            .collect::<Vec<_>>();
        changes.global = !self.global_images.is_empty() && globals != self.global_images;
        self.global_images = globals;
        let mut next_meshes = HashMap::default();
        for id in mesh_ids {
            let state = MeshState {
                revision: blas.revision(&id),
                ready: blas.get(&id).is_some(),
                bounds: blas.bounds.get(&id).copied(),
            };
            if let Some(old) = self.meshes.get(&id)
                && *old != state
            {
                changes.meshes.insert(id);
                changes.old_bounds.insert(id, old.bounds);
            }
            next_meshes.insert(id, state);
        }
        // Removed instances still need their old bounds, even after the asset
        // itself was unloaded before the scene publication reached this binder.
        for (&id, state) in &self.meshes {
            changes.old_bounds.entry(id).or_insert(state.bounds);
        }
        let mut next_materials = HashMap::default();
        for id in material_ids {
            let state = materials.get(&id).map(|m| {
                let mut state = material_state(m, images);
                let mut additional = Vec::new();
                if let Some(detail) = extensions.0.0.get(&id) {
                    additional.extend([
                        detail.coverage0.id(),
                        detail.coverage1.id(),
                        detail.colour.id(),
                        detail.detail.id(),
                        detail.meso_colour.id(),
                        detail.meso_detail.id(),
                    ]);
                }
                if let Some(gaussian) = extensions.1.0.get(&id) {
                    additional.push(gaussian.mask.id());
                }
                state.images.extend(
                    additional
                        .into_iter()
                        .map(|id| image_state(images, Some(id))),
                );
                state
            });
            let old = self.materials.get(&id);
            if old.is_some_and(|m| m.emitter) {
                changes.old_emitters.insert(id);
            }
            if state.as_ref() != old {
                changes.materials.insert(id);
                changes.global |=
                    old.is_some_and(|m| m.emitter) || state.as_ref().is_none_or(|s| s.emitter);
            }
            if let Some(state) = state {
                next_materials.insert(id, state);
            }
        }
        for (&id, state) in &self.materials {
            if state.emitter {
                changes.old_emitters.insert(id);
            }
        }
        self.meshes = next_meshes;
        self.materials = next_materials;
        changes
    }
}
