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
                        manager.binding_slots[i],
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
    sync_sources(
        world,
        state,
        manager,
        force,
        &HashSet::default(),
        &HashSet::default(),
        &RenderMaterialBindings::default(),
        &mut |part| if blocked { None } else { prepared(part) },
    )
}
fn sync_sources(
    world: &mut World,
    state: &mut State,
    manager: &mut InstanceManager,
    force: bool,
    meshes: &HashSet<AssetId<MeshletMesh>>,
    materials: &HashSet<UntypedAssetId>,
    bindings: &RenderMaterialBindings,
    prepare: &mut impl FnMut(&MeshletAssemblyPart) -> Option<Prepared>,
) -> (usize, usize, Vec<(SceneInstance, SceneSlot)>) {
    let (rows, mut changes) = state.get_mut(world).unwrap();
    let invalidated = invalidated_sources(&rows, meshes, materials);
    let mut changed = changes.collect();
    changed.extend(invalidated.iter().copied());
    let roots = targets(manager, &rows, changed, force);
    manager.slots.begin();
    let mut removed = Vec::new();
    let mut parts = 0;
    for &root in &roots {
        parts += update_root(
            manager,
            root,
            rows.get(root).ok(),
            force || invalidated.contains(&root),
            bindings,
            prepare,
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
    assembly_hierarchy::update(manager, &roots);
    let mut fresh = InstanceManager::new();
    fresh.slots.begin();
    for row in &rows {
        update_root(
            &mut fresh,
            row.0,
            Some(row),
            true,
            bindings,
            prepare,
            &mut Vec::new(),
        );
    }
    fresh.refresh_scene_metadata();
    assert_same(manager, &fresh);
    let mut members = Vec::new();
    for group in manager.assembly_groups.get() {
        let count = group.count & !(1 << 31);
        members.extend_from_slice(
            &manager.assembly_members.get()[group.first as usize..(group.first + count) as usize],
        );
    }
    members.sort_unstable();
    let mut active = manager.slots.active_indices().to_vec();
    active.sort_unstable();
    assert_eq!(
        members, active,
        "hierarchy visits each live part exactly once"
    );
    (roots.len(), parts, removed)
}

#[test]
fn hierarchy_keeps_unchanged_root_addresses_after_neighbor_removal() {
    let mut world = World::new();
    let mut state = State::new(&mut world);
    let prototype = assembly(&[0.0, 1.0, 2.0]);
    let mut roots: Vec<_> = (0..4)
        .map(|_| {
            world
                .spawn((prototype.clone(), GlobalTransform::IDENTITY))
                .id()
        })
        .collect();
    roots.sort_unstable_by_key(|root| root.to_bits());
    let mut manager = InstanceManager::new();
    sync(&mut world, &mut state, &mut manager, false, false);
    let slot = manager.assembly_inputs[&roots[3]].slots[0].index;
    let address = |manager: &InstanceManager| {
        manager
            .assembly_groups
            .get()
            .iter()
            .enumerate()
            .find_map(|(i, group)| {
                (group.count != 0 && manager.assembly_members.get()[group.first as usize] == slot)
                    .then_some((i, group.first))
            })
            .unwrap()
    };
    let original = address(&manager);
    world.despawn(roots[0]);
    sync(&mut world, &mut state, &mut manager, false, false);
    assert_eq!(
        address(&manager),
        original,
        "unchanged root must not be republished"
    );
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
    assembly_hierarchy::update(&mut manager, &[root]);
    assert_eq!(
        manager
            .assembly_groups
            .get()
            .iter()
            .map(|g| g.count & !(1 << 31))
            .sum::<u32>(),
        3
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
    assembly_hierarchy::update(&mut manager, &[root]);
    let live: Vec<_> = manager
        .assembly_groups
        .get()
        .iter()
        .filter(|g| g.count != 0)
        .collect();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].count, 1);
    assert_eq!(
        manager.assembly_members.get()[live[0].first as usize],
        ordinary.index
    );
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

fn distinct_source(
    meshes: &Assets<MeshletMesh>,
    materials: &Assets<StandardMaterial>,
) -> MeshletAssembly3d {
    let mut source = assembly(&[0.0, 1.0]);
    let mesh = meshes.reserve_handle();
    let material = materials.reserve_handle();
    for part in Arc::make_mut(&mut source.0) {
        part.mesh = mesh.clone();
        part.material = material.clone();
    }
    source
}

struct SourceState {
    bindings: RenderMaterialBindings,
    geometry: HashMap<AssetId<MeshletMesh>, Option<u32>>,
}
impl SourceState {
    fn sync(
        &self,
        world: &mut World,
        state: &mut State,
        manager: &mut InstanceManager,
        force: bool,
        meshes: &[AssetId<MeshletMesh>],
        materials: &[UntypedAssetId],
    ) -> (usize, usize, Vec<(SceneInstance, SceneSlot)>) {
        sync_sources(
            world,
            state,
            manager,
            force,
            &meshes.iter().copied().collect(),
            &materials.iter().copied().collect(),
            &self.bindings,
            &mut |part| {
                self.bindings.get(&part.material.id().untyped())?;
                let revision = self.geometry.get(&part.mesh.id()).copied().flatten()?;
                let mut data = prepared(part)?;
                data.0 += revision * 1000;
                Some(data)
            },
        )
    }
}

#[test]
fn source_changes_preserve_unrelated_slots_and_match_full_publication() {
    let mut world = World::new();
    let mut state = State::new(&mut world);
    let first = assembly(&[0.0, 1.0]);
    let meshes = Assets::<MeshletMesh>::default();
    let materials = Assets::<StandardMaterial>::default();
    let second = distinct_source(&meshes, &materials);
    let mesh = first.0[0].mesh.id();
    let other_mesh = second.0[0].mesh.id();
    let material = first.0[0].material.id().untyped();
    let other_material = second.0[0].material.id().untyped();
    let a = world.spawn((first.clone(), GlobalTransform::IDENTITY)).id();
    // Same source IDs in a different allocation must still invalidate.
    let independent = MeshletAssembly3d::new(first.0.to_vec());
    let b = world.spawn((independent, GlobalTransform::IDENTITY)).id();
    let c = world.spawn((second, GlobalTransform::IDENTITY)).id();
    let mut sources = SourceState {
        bindings: RenderMaterialBindings::default(),
        geometry: HashMap::from_iter([(mesh, Some(0)), (other_mesh, Some(0))]),
    };
    sources
        .bindings
        .insert(material, MaterialBindingId::default());
    sources
        .bindings
        .insert(other_material, MaterialBindingId::default());
    let mut manager = InstanceManager::new();
    assert_eq!(
        sources
            .sync(&mut world, &mut state, &mut manager, false, &[], &[])
            .1,
        6
    );
    let retained = manager.assembly_inputs[&c].slots.clone();
    let stable = manager.assembly_inputs[&a].slots.clone();
    let unrelated_mesh = meshes.reserve_handle();
    assert_eq!(
        sources
            .sync(
                &mut world,
                &mut state,
                &mut manager,
                false,
                &[unrelated_mesh.id()],
                &[]
            )
            .0,
        0
    );

    sources.geometry.insert(mesh, Some(1));
    let changed = sources.sync(&mut world, &mut state, &mut manager, false, &[mesh], &[]);
    assert_eq!((changed.0, changed.1), (2, 4));
    assert_eq!(manager.assembly_inputs[&a].slots, stable);
    assert_eq!(manager.assembly_inputs[&c].slots, retained);
    sources.bindings.get_mut(&other_material).unwrap().slot.0 = 7;
    let changed = sources.sync(
        &mut world,
        &mut state,
        &mut manager,
        false,
        &[],
        &[other_material],
    );
    assert_eq!((changed.0, changed.1), (1, 2));

    sources.geometry.insert(mesh, None);
    sources.sync(&mut world, &mut state, &mut manager, false, &[mesh], &[]);
    assert_eq!(manager.assembly_pending, HashSet::from_iter([a, b]));
    assert_eq!(manager.scene_instance_count, 2);
    sources.geometry.insert(mesh, Some(2));
    assert_eq!(
        sources
            .sync(&mut world, &mut state, &mut manager, false, &[], &[])
            .1,
        4
    );
    assert!(manager.assembly_pending.is_empty());
    sources.bindings.remove(&material);
    sources.sync(
        &mut world,
        &mut state,
        &mut manager,
        false,
        &[],
        &[material],
    );
    assert_eq!(manager.scene_instance_count, 2);
    sources
        .bindings
        .insert(material, MaterialBindingId::default());
    assert_eq!(
        sources
            .sync(&mut world, &mut state, &mut manager, false, &[], &[])
            .1,
        4
    );

    world.despawn(a);
    let replacement = world.spawn((first, GlobalTransform::IDENTITY)).id();
    assert_ne!(replacement, a);
    sources.sync(&mut world, &mut state, &mut manager, false, &[mesh], &[]);
    assert!(!manager.assembly_inputs.contains_key(&a));
    assert!(manager.assembly_inputs.contains_key(&replacement));
    assert_eq!(manager.assembly_inputs[&c].slots, retained);
    // Atlas/receipt invalidation must continue to visit every source.
    let full = sources.sync(&mut world, &mut state, &mut manager, true, &[], &[]);
    assert_eq!((full.0, full.1), (3, 6));
}

#[test]
#[ignore = "full forest target-selection diagnostic; run explicitly in release"]
fn full_bank_resource_invalidation_targets_only_source_users() {
    let mut world = World::new();
    let forest = assembly(&[0.0, 1.0, 2.0]);
    let terrain = distinct_source(&Assets::default(), &Assets::default());
    for _ in 0..262_144 {
        world.spawn((forest.clone(), GlobalTransform::IDENTITY));
    }
    let ground: HashSet<_> = (0..16)
        .map(|_| {
            world
                .spawn((terrain.clone(), GlobalTransform::IDENTITY))
                .id()
        })
        .collect();
    let mut state = SystemState::<AssemblyRows>::new(&mut world);
    let rows = state.get(&world).unwrap();
    let mut manager = InstanceManager::new();
    for row in &rows {
        manager
            .assembly_inputs
            .insert(row.0, AssemblyInput::new(row));
    }
    let start = std::time::Instant::now();
    let invalidated = invalidated_sources(
        &rows,
        &HashSet::from_iter([terrain.0[0].mesh.id()]),
        &HashSet::default(),
    );
    assert_eq!(invalidated, ground);
    let selected = targets(&manager, &rows, invalidated, false);
    let elapsed = start.elapsed();
    let full = targets(&manager, &rows, HashSet::default(), true);
    assert_eq!(full.len(), 262_160);
    assert_eq!(selected.len(), 16);
    assert!(selected.len() * 10 < full.len());
    assert_eq!(
        invalidated_sources(
            &rows,
            &HashSet::default(),
            &HashSet::from_iter([terrain.0[0].material.id().untyped()])
        ),
        ground
    );
    println!(
        "forest source invalidation: {} roots retained; {} roots visited; selection {:?}",
        full.len() - selected.len(),
        selected.len(),
        elapsed
    );
}
