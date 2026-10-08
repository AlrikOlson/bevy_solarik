//! Detect renderer input mutations before walking persistent ray scene slots.
use super::{RaytracingAssembly3d, RaytracingMesh3d};
use bevy_ecs::{
    entity::Entity,
    lifecycle::RemovedComponents,
    query::{Changed, Or, With},
    system::{Query, SystemParam},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashSet;
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
        Entity,
        (
            With<RaytracingMesh3d>,
            With<Material>,
            With<GlobalTransform>,
            ChangedInputs,
        ),
    >,
    assembly_changed: Query<
        'w,
        's,
        Entity,
        (
            With<RaytracingAssembly3d>,
            Or<(
                Changed<RaytracingAssembly3d>,
                Changed<GlobalTransform>,
                Changed<PreviousGlobalTransform>,
            )>,
        ),
    >,
    assembly_removed: RemovedComponents<'w, 's, RaytracingAssembly3d>,
    mesh: RemovedComponents<'w, 's, RaytracingMesh3d>,
    material: RemovedComponents<'w, 's, Material>,
    transform: RemovedComponents<'w, 's, GlobalTransform>,
    previous: RemovedComponents<'w, 's, PreviousGlobalTransform>,
}
impl RayInstanceChanges<'_, '_> {
    pub(super) fn collect(&mut self) -> HashSet<Entity> {
        let mut roots: HashSet<_> = self
            .changed
            .iter()
            .chain(self.assembly_changed.iter())
            .collect();
        roots.extend(self.assembly_removed.read());
        roots.extend(self.mesh.read());
        roots.extend(self.material.read());
        roots.extend(self.transform.read());
        roots.extend(self.previous.read());
        roots
    }
    #[cfg(test)]
    fn unchanged(&mut self) -> bool {
        self.collect().is_empty()
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
