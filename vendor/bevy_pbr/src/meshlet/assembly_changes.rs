//! Changed root identities, including removed required and optional components.
use super::super::MeshletAssembly3d;
use crate::PreviousGlobalTransform;
use bevy_camera::visibility::RenderLayers;
use bevy_ecs::{
    entity::Entity,
    lifecycle::RemovedComponents,
    query::{Changed, Or, With},
    system::{Query, SystemParam},
};
use bevy_light::{NotShadowCaster, NotShadowReceiver};
use bevy_platform::collections::HashSet;
use bevy_transform::components::GlobalTransform;

#[derive(SystemParam)]
pub(in crate::meshlet) struct AssemblyChanges<'w, 's> {
    changed: Query<
        'w,
        's,
        Entity,
        (
            With<MeshletAssembly3d>,
            Or<(
                Changed<MeshletAssembly3d>,
                Changed<GlobalTransform>,
                Changed<PreviousGlobalTransform>,
                Changed<RenderLayers>,
                Changed<NotShadowReceiver>,
                Changed<NotShadowCaster>,
            )>,
        ),
    >,
    assembly: RemovedComponents<'w, 's, MeshletAssembly3d>,
    transform: RemovedComponents<'w, 's, GlobalTransform>,
    previous: RemovedComponents<'w, 's, PreviousGlobalTransform>,
    layers: RemovedComponents<'w, 's, RenderLayers>,
    receiver: RemovedComponents<'w, 's, NotShadowReceiver>,
    caster: RemovedComponents<'w, 's, NotShadowCaster>,
}
impl AssemblyChanges<'_, '_> {
    pub(super) fn collect(&mut self) -> HashSet<Entity> {
        self.changed
            .iter()
            .chain(self.assembly.read())
            .chain(self.transform.read())
            .chain(self.previous.read())
            .chain(self.layers.read())
            .chain(self.receiver.read())
            .chain(self.caster.read())
            .collect()
    }
}
