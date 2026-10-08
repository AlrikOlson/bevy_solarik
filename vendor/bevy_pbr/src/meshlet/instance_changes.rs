//! ECS evidence for skipping the large exact scene receipt on unchanged frames.
//! Removals must be observed separately: Changed<T> cannot see a missing T.
use super::{MeshletDoubleSided, MeshletMesh3d, MeshletVisibilityCutout};
use crate::PreviousGlobalTransform;
use bevy_camera::visibility::RenderLayers;
use bevy_ecs::{
    lifecycle::RemovedComponents,
    query::{Changed, Or, With},
    system::{Query, SystemParam},
};
use bevy_light::{NotShadowCaster, NotShadowReceiver};
use bevy_transform::components::GlobalTransform;

type ChangedInputs = Or<(
    Changed<MeshletMesh3d>,
    Changed<GlobalTransform>,
    Changed<PreviousGlobalTransform>,
    Changed<RenderLayers>,
    Changed<NotShadowReceiver>,
    Changed<NotShadowCaster>,
    Changed<MeshletVisibilityCutout>,
    Changed<MeshletDoubleSided>,
)>;

#[derive(SystemParam)]
pub(super) struct InstanceChanges<'w, 's> {
    changed: Query<'w, 's, (), (With<MeshletMesh3d>, With<GlobalTransform>, ChangedInputs)>,
    mesh: RemovedComponents<'w, 's, MeshletMesh3d>,
    transform: RemovedComponents<'w, 's, GlobalTransform>,
    previous: RemovedComponents<'w, 's, PreviousGlobalTransform>,
    layers: RemovedComponents<'w, 's, RenderLayers>,
    receiver: RemovedComponents<'w, 's, NotShadowReceiver>,
    caster: RemovedComponents<'w, 's, NotShadowCaster>,
    cutout: RemovedComponents<'w, 's, MeshletVisibilityCutout>,
    sidedness: RemovedComponents<'w, 's, MeshletDoubleSided>,
}

impl InstanceChanges<'_, '_> {
    pub(super) fn unchanged(&mut self) -> bool {
        // Evaluate and drain every reader, even when an earlier one changed.
        let removed = [
            self.mesh.read().count(),
            self.transform.read().count(),
            self.previous.read().count(),
            self.layers.read().count(),
            self.receiver.read().count(),
            self.caster.read().count(),
            self.cutout.read().count(),
            self.sidedness.read().count(),
        ];
        removed.iter().all(|&count| count == 0) && self.changed.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{system::SystemState, world::World};
    use bevy_math::{Affine3A, Vec3};

    #[test]
    fn actual_ecs_changes_and_removals_cannot_hide_behind_stable_membership() {
        let mut world = World::new();
        let mut state = SystemState::<InstanceChanges>::new(&mut world);
        let entity = world
            .spawn((MeshletMesh3d::default(), GlobalTransform::IDENTITY))
            .id();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        macro_rules! change_and_remove {
            ($value:expr, $ty:ty) => {{
                world.entity_mut(entity).insert($value);
                assert!(!state.get_mut(&mut world).unwrap().unchanged());
                assert!(state.get_mut(&mut world).unwrap().unchanged());
                world.entity_mut(entity).remove::<$ty>();
                assert!(!state.get_mut(&mut world).unwrap().unchanged());
                assert!(state.get_mut(&mut world).unwrap().unchanged());
            }};
        }
        change_and_remove!(
            PreviousGlobalTransform(Affine3A::IDENTITY),
            PreviousGlobalTransform
        );
        change_and_remove!(RenderLayers::layer(3), RenderLayers);
        change_and_remove!(NotShadowReceiver, NotShadowReceiver);
        change_and_remove!(NotShadowCaster, NotShadowCaster);
        change_and_remove!(
            MeshletVisibilityCutout {
                layer: 1,
                cutoff: 0.5
            },
            MeshletVisibilityCutout
        );
        change_and_remove!(MeshletDoubleSided, MeshletDoubleSided);
        world
            .entity_mut(entity)
            .insert(GlobalTransform::from_translation(Vec3::X));
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world
            .entity_mut(entity)
            .insert(PreviousGlobalTransform(Affine3A::from_translation(Vec3::X)));
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        // Mutation through an ECS borrow, not just component insertion.
        world
            .entity_mut(entity)
            .get_mut::<PreviousGlobalTransform>()
            .unwrap()
            .0 = Affine3A::IDENTITY;
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.despawn(entity);
        let replacement = world
            .spawn((MeshletMesh3d::default(), GlobalTransform::IDENTITY))
            .id();
        assert!(
            !state.get_mut(&mut world).unwrap().unchanged(),
            "same-size replacement is not stable"
        );
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world.despawn(replacement);
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
    }

    #[test]
    fn removing_required_inputs_and_multiple_optional_inputs_drains_every_reader() {
        let mut world = World::new();
        let mut state = SystemState::<InstanceChanges>::new(&mut world);
        let entity = world
            .spawn((
                MeshletMesh3d::default(),
                GlobalTransform::IDENTITY,
                NotShadowCaster,
                NotShadowReceiver,
            ))
            .id();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world
            .entity_mut(entity)
            .remove::<(NotShadowCaster, NotShadowReceiver)>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).remove::<GlobalTransform>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).remove::<MeshletMesh3d>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
    }
}
