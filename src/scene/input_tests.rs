use super::*;
use crate::scene::instance_changes::RayInstanceChanges;
use crate::scene::{RaytracingAssembly3d, RaytracingAssemblyPart, RaytracingMesh3d};
use bevy_ecs::{system::SystemState, world::World};
use bevy_pbr::PreviousGlobalTransform;
use bevy_transform::components::GlobalTransform;

fn assembly(offsets: &[f32]) -> RaytracingAssembly3d {
    RaytracingAssembly3d::new(
        offsets
            .iter()
            .map(|x| RaytracingAssemblyPart {
                mesh: Handle::default(),
                material: Handle::default(),
                transform: Affine3A::from_translation(Vec3::X * *x),
            })
            .collect::<Vec<_>>(),
    )
}
fn rows_for(
    world: &mut World,
    state: &mut SystemState<(SceneRows, RayInstanceChanges)>,
    root: Entity,
) -> (Vec<RayInstanceInput>, HashSet<Entity>) {
    let (rows, mut changed) = state.get_mut(world).unwrap();
    let root_rows = rows.for_root(root);
    let full: Vec<_> = rows.iter().filter(|r| r.entity.entity == root).collect();
    assert!(root_rows == full, "delta and full snapshot expansion agree");
    (root_rows, changed.collect())
}
#[test]
fn delta_roots_reorder_shrink_remove_and_preserve_ordinary_owner() {
    let mut world = World::new();
    let root = world
        .spawn((assembly(&[1., 2., 3.]), RaytracingMesh3d::default()))
        .id();
    let unrelated = world.spawn(assembly(&[4., 5.])).id();
    let mut state = SystemState::<(SceneRows, RayInstanceChanges)>::new(&mut world);
    let mut owners = super::super::input_roots::InputRoots::default();
    owners.clear();
    let (initial, changed) = rows_for(&mut world, &mut state, root);
    assert_eq!(changed.len(), 2);
    let other = state.get_mut(&mut world).unwrap().0.for_root(unrelated);
    for row in initial.iter().chain(other.iter()) {
        owners.observe(row);
    }
    assert_eq!(owners.len(), 2);
    assert_eq!(initial.len(), 4);
    world.entity_mut(root).insert(assembly(&[3., 1., 2.]));
    let (reordered, changed) = rows_for(&mut world, &mut state, root);
    assert_eq!(changed, HashSet::from_iter([root]));
    assert!(owners.replace(root, &reordered).is_empty());
    assert_ne!(initial[1].transform, reordered[1].transform);
    world.entity_mut(root).insert(assembly(&[3.]));
    let (shrunk, _) = rows_for(&mut world, &mut state, root);
    assert_eq!(
        owners.replace(root, &shrunk),
        [initial[2].entity, initial[3].entity]
    );
    world.entity_mut(root).remove::<RaytracingAssembly3d>();
    let (single, changed) = rows_for(&mut world, &mut state, root);
    assert_eq!(changed, HashSet::from_iter([root]));
    assert_eq!(owners.replace(root, &single), [initial[1].entity]);
    assert_eq!(single.len(), 1);
    world.despawn(root);
    let (removed, changed) = rows_for(&mut world, &mut state, root);
    assert!(removed.is_empty());
    assert!(changed.contains(&root));
    assert_eq!(owners.replace(root, &removed), [initial[0].entity]);
    assert_eq!(owners.len(), 1, "unrelated root was never removed");
    assert!(owners.replace(unrelated, &other).is_empty());
}
#[test]
fn delta_previous_transform_settles_without_occluder_change_and_dependencies_republish() {
    let mut world = World::new();
    let root = world
        .spawn((
            assembly(&[1., 2.]),
            GlobalTransform::from_translation(Vec3::X),
            PreviousGlobalTransform(Affine3A::IDENTITY),
        ))
        .id();
    let mut state = SystemState::<(SceneRows, RayInstanceChanges)>::new(&mut world);
    let (initial, _) = rows_for(&mut world, &mut state, root);
    let mut cache = SceneCache::default();
    cache.slots.begin();
    let mut dirty = Vec::new();
    for row in initial.iter().cloned() {
        assert!(observe_instance(&mut cache, row, false, &mut dirty));
    }
    let first = cache.slots.get_part(initial[0].entity).unwrap();
    assert!(
        !cache.slots.is_active(first),
        "pending asset retains its identity"
    );
    cache.slots.activate(first);
    cache.slots.deactivate(first);
    world
        .entity_mut(root)
        .insert(PreviousGlobalTransform(Affine3A::from_translation(Vec3::X)));
    let (settled, changed) = rows_for(&mut world, &mut state, root);
    assert_eq!(changed, HashSet::from_iter([root]));
    dirty.clear();
    for (old, new) in initial.iter().zip(settled.iter()) {
        assert!(same_occluder(old, new));
        assert!(!observe_instance(
            &mut cache,
            new.clone(),
            false,
            &mut dirty
        ));
    }
    assert_eq!(dirty.len(), 2, "motion history is still published");
    dirty.clear();
    for row in settled.iter().cloned() {
        observe_instance(&mut cache, row, false, &mut dirty);
    }
    assert!(dirty.is_empty(), "settled rows are retained");
    for row in settled.iter().cloned() {
        observe_instance(&mut cache, row, true, &mut dirty);
    }
    assert_eq!(
        dirty.len(),
        2,
        "asset readiness forces republish even with unchanged transforms"
    );
    assert_eq!(cache.slots.get_part(initial[0].entity), Some(first));
    world.entity_mut(root).remove::<PreviousGlobalTransform>();
    let (no_previous, changed) = rows_for(&mut world, &mut state, root);
    assert_eq!(changed, HashSet::from_iter([root]));
    assert!(no_previous == settled);
}

#[test]
fn shared_source_pairs_match_full_expansion_without_root_transform_work() {
    let mut world = World::new();
    let shared = assembly(&[1., 2., 3.]);
    let root = world.spawn(shared.clone()).id();
    for i in 1..512 {
        world.spawn((
            shared.clone(),
            GlobalTransform::from_translation(Vec3::X * i as f32),
        ));
    }
    world.spawn(RaytracingMesh3d::default());
    let mut materials = bevy_asset::Assets::<StandardMaterial>::default();
    let different = materials.add(StandardMaterial::default());
    world.spawn(RaytracingAssembly3d::new(vec![RaytracingAssemblyPart {
        mesh: Handle::default(),
        material: different,
        transform: Affine3A::IDENTITY,
    }]));
    let mut state = SystemState::<SceneRows>::new(&mut world);
    for replacement in [assembly(&[4.]), assembly(&[]), shared.clone()] {
        world.entity_mut(root).insert(replacement);
        let rows = state.get_mut(&mut world).unwrap();
        let full: HashSet<_> = rows
            .iter()
            .map(|input| (input.mesh, input.material))
            .collect();
        assert_eq!(rows.sources(), full);
    }
    world.entity_mut(root).remove::<RaytracingAssembly3d>();
    let rows = state.get_mut(&mut world).unwrap();
    assert_eq!(
        rows.sources(),
        rows.iter()
            .map(|input| (input.mesh, input.material))
            .collect()
    );
}
