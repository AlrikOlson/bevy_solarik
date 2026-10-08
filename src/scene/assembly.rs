//! Ray geometry shares immutable source parts with one extracted scene root.
use alloc::sync::Arc;
use bevy_asset::Handle;
use bevy_ecs::{
    component::Component,
    lifecycle::RemovedComponents,
    query::{With, Without},
    system::{Commands, Query},
};
use bevy_math::Affine3A;
use bevy_mesh::Mesh;
use bevy_pbr::{PreviousGlobalTransform, StandardMaterial};
use bevy_render::{
    Extract,
    sync_world::{RenderEntity, SyncToRenderWorld},
};
use bevy_transform::components::{GlobalTransform, Transform};

#[derive(Clone)]
pub struct RaytracingAssemblyPart {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
    pub transform: Affine3A,
}
#[derive(Component, Clone)]
#[require(Transform, SyncToRenderWorld)]
pub struct RaytracingAssembly3d(pub Arc<[RaytracingAssemblyPart]>);
impl RaytracingAssembly3d {
    pub fn new(parts: impl Into<Arc<[RaytracingAssemblyPart]>>) -> Self {
        let parts = parts.into();
        assert!(
            parts.len() < u32::MAX as usize,
            "ray assembly part identity overflow"
        );
        Self(parts)
    }
}
pub(super) fn previous_transforms(
    mut commands: Commands,
    new: Query<
        (bevy_ecs::entity::Entity, &GlobalTransform),
        (With<RaytracingAssembly3d>, Without<PreviousGlobalTransform>),
    >,
    mut existing: Query<
        (&GlobalTransform, &mut PreviousGlobalTransform),
        With<RaytracingAssembly3d>,
    >,
) {
    for (entity, transform) in &new {
        commands
            .entity(entity)
            .try_insert(PreviousGlobalTransform(transform.affine()));
    }
    for (transform, mut previous) in &mut existing {
        if previous.0 != transform.affine() {
            previous.0 = transform.affine();
        }
    }
}
pub(super) fn extract(
    roots: Extract<
        Query<(
            RenderEntity,
            &RaytracingAssembly3d,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
        )>,
    >,
    mut removed: Extract<RemovedComponents<RaytracingAssembly3d>>,
    mapping: Extract<Query<RenderEntity>>,
    mut commands: Commands,
    existing: Query<(
        &RaytracingAssembly3d,
        &GlobalTransform,
        Option<&PreviousGlobalTransform>,
    )>,
) {
    for entity in removed.read() {
        if let Ok(render) = mapping.get(entity) {
            commands.entity(render).remove::<RaytracingAssembly3d>();
        }
    }
    for (entity, assembly, transform, previous) in &roots {
        if existing
            .get(entity)
            .is_ok_and(|(old, old_transform, old_previous)| {
                Arc::ptr_eq(&old.0, &assembly.0)
                    && old_transform.affine() == transform.affine()
                    && old_previous.map(|p| p.0) == previous.map(|p| p.0)
            })
        {
            continue;
        }
        let mut root = commands.entity(entity);
        root.insert((assembly.clone(), *transform));
        if let Some(previous) = previous {
            root.insert(previous.clone());
        } else {
            root.remove::<PreviousGlobalTransform>();
        }
    }
}
