//! Delta publication is compared with a fresh full scene after actual ECS changes.
use super::*;
use crate::StandardMaterial;
use bevy_ecs::world::World;
use bevy_math::Vec3;

type State = SystemState<(
    AssemblyRows<'static, 'static>,
    assembly_changes::AssemblyChanges<'static, 'static>,
)>;

fn assembly(offsets: &[f32]) -> MeshletAssembly3d {
    MeshletAssembly3d::new(
        offsets
            .iter()
            .map(|&x| MeshletAssemblyPart {
                mesh: Default::default(),
                material: Default::default(),
                transform: Affine3A::from_translation(Vec3::new(x, 0.0, 0.0)),
                cutout: None,
                double_sided: x > 1.0,
            })
            .collect::<Vec<_>>(),
    )
}
fn prepared(part: &MeshletAssemblyPart) -> Option<Prepared> {
    let x = part.transform.translation.x as u32;
    Some((
        100 + x,
        MeshletAabb::default(),
        1 + x,
        Vec4::new(0.0, 0.5, f32::from(part.double_sided), 0.0),
    ))
}
fn assert_same(actual: &InstanceManager, expected: &InstanceManager) {
    let active = |manager: &InstanceManager| {
        let mut rows = alloc::collections::BTreeMap::new();
        for (&root, input) in &manager.assembly_inputs {
            for (part, &slot) in input.slots.iter().enumerate() {
                if !manager.slots.is_active(slot) {
                    continue;
                }
                let i = slot.index as usize;
                let uniform = &manager.instance_uniforms.get()[i];
                rows.insert(
                    (root.to_bits(), part),
                    (
                        uniform.world_from_local,
                        uniform.previous_world_from_local,
                        uniform.flags,
                        manager.instance_cutouts.get()[i],
                        manager.instance_material_assets[i],
                        manager.bvh_depths[i],
                        manager.instance_bvh_root_nodes.get()[i],
                        manager.instances[i].clone(),
                    ),
                );
            }
        }
        rows
    };
    assert_eq!(active(actual), active(expected));
    assert_eq!(
        actual.scene_metadata.materials,
        expected.scene_metadata.materials
    );
    assert_eq!(actual.max_bvh_depth, expected.max_bvh_depth);
    assert_eq!(actual.scene_instance_count, expected.scene_instance_count);
    assert_eq!(actual.assembly_part_count, expected.assembly_part_count);
    assert_eq!(actual.assembly_pending, expected.assembly_pending);
}
fn sync(
    world: &mut World,
    state: &mut State,
    manager: &mut InstanceManager,
    force: bool,
    blocked: bool,
) -> (usize, usize, Vec<(SceneInstance, SceneSlot)>) {
    let (rows, mut changes) = state.get_mut(world).unwrap();
    let roots = targets(manager, &rows, changes.collect(), force);
    let bindings = RenderMaterialBindings::default();
    let mut prepare = |part: &MeshletAssemblyPart| if blocked { None } else { prepared(part) };
    manager.slots.begin();
    let mut removed = Vec::new();
    let mut parts = 0;
    for &root in &roots {
        parts += update_root(
            manager,
            root,
            rows.get(root).ok(),
            force,
            &bindings,
            &mut prepare,
            &mut removed,
        )
        .1;
    }
    // CPU equivalence covers requested retirements and live metadata. Actual
    // completion/reuse is exercised by SceneSlots and the GPU storage fixture.
    for &(_, slot) in &removed {
        manager.deactivate_slot(slot);
    }
    manager.refresh_scene_metadata();
    let mut fresh = InstanceManager::new();
    fresh.slots.begin();
    for row in &rows {
        update_root(
            &mut fresh,
            row.0,
            Some(row),
            true,
            &bindings,
            &mut prepare,
            &mut Vec::new(),
        );
    }
    fresh.refresh_scene_metadata();
    assert_same(manager, &fresh);
    (roots.len(), parts, removed)
}
#[test]
fn root_delta_matches_full_snapshot_after_motion_replacement_shrink_and_removal() {
    let mut world = World::new();
    let mut state = State::new(&mut world);
    let prototype = assembly(&[0.0, 1.0, 2.0]);
    let roots: Vec<_> = (0..4)
        .map(|_| {
            world
                .spawn((
                    prototype.clone(),
                    GlobalTransform::IDENTITY,
                    PreviousGlobalTransform(Affine3A::IDENTITY),
                ))
                .id()
        })
        .collect();
    let mut manager = InstanceManager::new();
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).1,
        12
    );
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).0,
        0
    );
    let retained = manager.assembly_inputs[&roots[3]].slots.clone();
    world
        .entity_mut(roots[0])
        .insert(GlobalTransform::from_translation(Vec3::Y));
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).0,
        1
    );
    world
        .entity_mut(roots[0])
        .insert(PreviousGlobalTransform(Affine3A::from_translation(Vec3::Y)));
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).1,
        3
    );
    world
        .entity_mut(roots[0])
        .remove::<PreviousGlobalTransform>();
    sync(&mut world, &mut state, &mut manager, false, false);
    world.entity_mut(roots[1]).insert(assembly(&[2.0]));
    let delta = sync(&mut world, &mut state, &mut manager, false, false);
    assert_eq!((delta.0, delta.1, delta.2.len()), (1, 1, 2));
    assert!(
        delta
            .2
            .iter()
            .all(|(owner, _)| owner.entity == roots[1] && owner.part > 1)
    );
    world
        .entity_mut(roots[2])
        .insert(assembly(&[2.0, 0.0, 1.0]));
    sync(&mut world, &mut state, &mut manager, false, false);
    world.despawn(roots[0]);
    let newcomer = world.spawn((prototype, GlobalTransform::IDENTITY)).id();
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).0,
        2
    );
    world.entity_mut(newcomer).remove::<GlobalTransform>();
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false)
            .2
            .len(),
        3
    );
    assert_eq!(manager.assembly_inputs[&roots[3]].slots, retained);
    assert_eq!(sync(&mut world, &mut state, &mut manager, true, false).1, 7);
}
#[test]
fn pending_roots_optional_removals_and_material_depth_counts_match_full_rebuild() {
    let mut world = World::new();
    let mut state = State::new(&mut world);
    let root = world
        .spawn((assembly(&[0.0, 1.0, 2.0]), GlobalTransform::IDENTITY))
        .id();
    let mut manager = InstanceManager::new();
    sync(&mut world, &mut state, &mut manager, false, true);
    assert_eq!(manager.scene_instance_count, 0);
    let slots = manager.assembly_inputs[&root].slots.clone();
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).1,
        3
    );
    assert_eq!(manager.assembly_inputs[&root].slots, slots);
    world
        .entity_mut(root)
        .insert((RenderLayers::layer(3), NotShadowReceiver, NotShadowCaster));
    sync(&mut world, &mut state, &mut manager, false, false);
    world
        .entity_mut(root)
        .remove::<(RenderLayers, NotShadowReceiver, NotShadowCaster)>();
    sync(&mut world, &mut state, &mut manager, false, false);
    assert_eq!(
        sync(&mut world, &mut state, &mut manager, false, false).0,
        0
    );
    let assets = Assets::<StandardMaterial>::default();
    let material = assets.reserve_handle();
    let mut changed = assembly(&[0.0]);
    Arc::make_mut(&mut changed.0)[0].material = material;
    world.entity_mut(root).insert(changed);
    sync(&mut world, &mut state, &mut manager, false, false);
    assert_eq!(manager.max_bvh_depth, 1);
    sync(&mut world, &mut state, &mut manager, true, true);
    assert_eq!(manager.max_bvh_depth, 0);
    assert!(manager.scene_metadata.materials.is_empty());
    sync(&mut world, &mut state, &mut manager, false, false);
    world.entity_mut(root).remove::<MeshletAssembly3d>();
    sync(&mut world, &mut state, &mut manager, false, false);
    assert!(manager.assembly_pending.is_empty());
}
#[test]
fn assembly_removal_never_retires_an_ordinary_part_on_the_same_entity() {
    let root = Entity::from_bits(41);
    let mut manager = InstanceManager::new();
    manager.slots.begin();
    let ordinary = manager.observe_query(0, root);
    manager.add_material_instance(
        ordinary,
        root.into(),
        11,
        MeshletAabb::default(),
        8,
        &GlobalTransform::IDENTITY,
        None,
        None,
        DUMMY_MESH_MATERIAL.untyped(),
        &RenderMaterialBindings::default(),
        false,
        false,
        Vec4::ZERO,
    );
    let assembly = assembly(&[0.0, 1.0]);
    let mut removed = Vec::new();
    update_root(
        &mut manager,
        root,
        Some((
            root,
            &assembly,
            &GlobalTransform::IDENTITY,
            None,
            None,
            false,
            false,
        )),
        false,
        &RenderMaterialBindings::default(),
        &mut prepared,
        &mut removed,
    );
    manager.slots.begin();
    manager.observe_query(0, root);
    update_root(
        &mut manager,
        root,
        None,
        false,
        &RenderMaterialBindings::default(),
        &mut prepared,
        &mut removed,
    );
    assert_eq!(removed.len(), 2);
    assert!(removed.iter().all(|(owner, _)| owner.part > 0));
    for (_, slot) in removed {
        manager.deactivate_slot(slot);
    }
    assert!(
        manager
            .slots
            .unobserved_from(&manager.query_slots)
            .is_empty()
    );
    assert_eq!(manager.slots.active_indices(), &[ordinary.index]);
    manager.refresh_scene_metadata();
    assert_eq!(manager.max_bvh_depth, 8);
    assert_eq!(manager.scene_instance_count, 1);
}
