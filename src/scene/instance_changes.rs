//! Detect renderer input mutations before walking persistent ray scene slots.
use super::RaytracingMesh3d;
use bevy_ecs::{
    lifecycle::RemovedComponents,
    query::{Changed, Or, With},
    system::{Query, SystemParam},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_transform::components::GlobalTransform;

type Material = MeshMaterial3d<StandardMaterial>;
type ChangedInputs = Or<(
    Changed<RaytracingMesh3d>,
    Changed<Material>,
    Changed<GlobalTransform>,
    Changed<PreviousGlobalTransform>,
)>;

#[derive(SystemParam)]
pub(super) struct RayInstanceChanges<'w, 's> {
    changed: Query<
        'w,
        's,
        (),
        (
            With<RaytracingMesh3d>,
            With<Material>,
            With<GlobalTransform>,
            ChangedInputs,
        ),
    >,
    mesh: RemovedComponents<'w, 's, RaytracingMesh3d>,
    material: RemovedComponents<'w, 's, Material>,
    transform: RemovedComponents<'w, 's, GlobalTransform>,
    previous: RemovedComponents<'w, 's, PreviousGlobalTransform>,
}
impl RayInstanceChanges<'_, '_> {
    pub(super) fn unchanged(&mut self) -> bool {
        let removed = [
            self.mesh.read().count(),
            self.material.read().count(),
            self.transform.read().count(),
            self.previous.read().count(),
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
    fn ray_input_gate_observes_transform_history_material_mesh_and_removal() {
        let mut world = World::new();
        let mut state = SystemState::<RayInstanceChanges>::new(&mut world);
        let entity = world
            .spawn((
                RaytracingMesh3d::default(),
                Material::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world
            .entity_mut(entity)
            .insert(GlobalTransform::from_translation(Vec3::X));
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world
            .entity_mut(entity)
            .insert(PreviousGlobalTransform(Affine3A::IDENTITY));
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world
            .entity_mut(entity)
            .get_mut::<PreviousGlobalTransform>()
            .unwrap()
            .0 = Affine3A::from_translation(Vec3::X);
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).remove::<PreviousGlobalTransform>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).insert(Material::default());
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).remove::<Material>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).insert(Material::default());
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).insert(RaytracingMesh3d::default());
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).remove::<GlobalTransform>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).insert(GlobalTransform::IDENTITY);
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).remove::<RaytracingMesh3d>();
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.entity_mut(entity).insert(RaytracingMesh3d::default());
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        world.despawn(entity);
        let replacement = world
            .spawn((
                RaytracingMesh3d::default(),
                Material::default(),
                GlobalTransform::IDENTITY,
            ))
            .id();
        assert!(
            !state.get_mut(&mut world).unwrap().unchanged(),
            "same-count replacement changes identity"
        );
        assert!(state.get_mut(&mut world).unwrap().unchanged());
        world.despawn(replacement);
        assert!(!state.get_mut(&mut world).unwrap().unchanged());
        assert!(state.get_mut(&mut world).unwrap().unchanged());
    }
}
