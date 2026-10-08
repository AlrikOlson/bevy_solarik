//! Shared immutable source parts with one scene transform per assembly.
use super::{MeshletMesh, MeshletVisibilityCutout};
use crate::StandardMaterial;
use alloc::sync::Arc;
use bevy_asset::Handle;
use bevy_camera::visibility::Visibility;
use bevy_ecs::component::Component;
use bevy_math::Affine3A;
use bevy_transform::components::Transform;

/// One source surface. Geometry, alpha and material references are shared by all roots.
#[derive(Clone)]
pub struct MeshletAssemblyPart {
    pub mesh: Handle<MeshletMesh>,
    pub material: Handle<StandardMaterial>,
    pub transform: Affine3A,
    pub cutout: Option<MeshletVisibilityCutout>,
    pub double_sided: bool,
}

/// Part ordering is immutable source identity. Replace the Arc to publish a new prototype.
#[derive(Component, Clone)]
#[require(Transform, Visibility)]
pub struct MeshletAssembly3d(pub Arc<[MeshletAssemblyPart]>);

impl MeshletAssembly3d {
    pub fn new(parts: impl Into<Arc<[MeshletAssemblyPart]>>) -> Self {
        let parts = parts.into();
        assert!(
            parts.len() < u32::MAX as usize,
            "assembly part identity overflow"
        );
        Self(parts)
    }
}
