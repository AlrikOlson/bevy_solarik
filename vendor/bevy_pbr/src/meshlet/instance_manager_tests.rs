//! Exercise the production receipt against actual ECS mutations and removals.

#[test]
fn changed_material_owners_follow_slots_through_reorder_and_unknown_mutation() {
    let mut world = World::new();
    let a = spawn_meshlet(&mut world);
    let b = spawn_meshlet(&mut world);
    let mut manager = capture(&mut world);
    let sa = manager.slots.get(a).unwrap();
    let sb = manager.slots.get(b).unwrap();
    let mut materials = RenderMaterialInstances::default();
    insert_material(&mut materials, a.into());
    insert_material(&mut materials, b.into());
    manager.material_receipt = materials.instances.receipt();
    insert_material(&mut materials, b.into());
    manager.observe_query(0, b);
    manager.observe_query(1, a);
    assert!(!manager.changed_materials(&materials));
    assert!(!manager.material_changes[sa.index as usize]);
    assert!(manager.material_changes[sb.index as usize]);
    manager.material_receipt = materials.instances.receipt();
    materials.instances.remove(&a.into());
    assert!(!manager.changed_materials(&materials));
    assert!(manager.material_changes[sa.index as usize]);
    assert!(!manager.material_changes[sb.index as usize]);
    manager.material_receipt = materials.instances.receipt();
    assert!(!manager.changed_materials(&materials));
    assert!(!manager.material_changes.iter().any(|changed| *changed));
    materials
        .instances
        .get_mut(&MainEntity::from(b))
        .unwrap()
        .last_change_tick
        .set(12);
    assert!(manager.changed_materials(&materials));
    manager.material_receipt = materials.instances.receipt();
    materials.instances = crate::material_instances::MaterialInstanceMap::default();
    assert!(manager.changed_materials(&materials));
}

#[test]
fn query_order_receipts_follow_real_reorder_without_changing_scene_slots() {
    let mut world = World::new();
    let entities: Vec<_> = (0..8).map(|_| spawn_meshlet(&mut world)).collect();
    let mut manager = capture(&mut world);
    let original_slots = manager.query_slots.clone();
    manager.slots.begin();
    for (position, &entity) in entities.iter().rev().enumerate() {
        let slot = manager.observe_query(position, entity);
        assert_eq!(slot, original_slots[7 - position]);
        assert_eq!(manager.inputs[position].as_ref().unwrap().entity, entity);
    }
    let publication = manager.slots.publication;
    assert!(!matches(&mut world, &manager));
    for (position, &entity) in entities.iter().enumerate() {
        manager.observe_query(position, entity);
    }
    assert!(matches(&mut world, &manager));
    assert_eq!(manager.slots.publication, publication);
    // Inserting a new row ahead of retained receipts must move, not overwrite,
    // the previous first receipt and its reverse slot address.
    let newcomer = spawn_meshlet(&mut world);
    let slot = manager.observe_query(0, newcomer);
    manager.inputs[0] = Some(InstanceInput::from_row(
        world.query::<Data>().get(&world, newcomer).unwrap(),
        slot,
    ));
    for (position, &entity) in entities.iter().enumerate() {
        manager.observe_query(position + 1, entity);
        assert_eq!(
            manager.inputs[position + 1].as_ref().unwrap().entity,
            entity
        );
    }
    assert_eq!(manager.inputs.len(), 9);
}

use super::*;
use bevy_ecs::world::World;
use bevy_math::{Affine3A, Vec3};

type Data = (
    Entity,
    &'static MeshletMesh3d,
    &'static GlobalTransform,
    Option<&'static PreviousGlobalTransform>,
    Option<&'static RenderLayers>,
    Has<NotShadowReceiver>,
    Has<NotShadowCaster>,
    Option<&'static MeshletVisibilityCutout>,
    Has<MeshletDoubleSided>,
);

fn capture(world: &mut World) -> InstanceManager {
    let mut manager = InstanceManager::new();
    manager.slots.begin();
    for (position, row) in world.query::<Data>().iter(world).enumerate() {
        let slot = manager.observe_query(position, row.0);
        manager.inputs[position] = Some(InstanceInput::from_row(row, slot));
        manager.slots.activate(slot);
    }
    manager.query_publication = manager.slots.publication;
    manager
}
fn matches(world: &mut World, manager: &InstanceManager) -> bool {
    manager.query_matches(world.query::<Data>().iter(world))
}

#[test]
fn persistent_raw_scene_receipt_handles_motion_catchup_optional_removals_and_membership() {
    let mut world = World::new();
    let entity = world
        .spawn((MeshletMesh3d::default(), GlobalTransform::IDENTITY))
        .id();
    let mut inputs = capture(&mut world);
    assert!(matches(&mut world, &inputs));
    // Previous-transform extraction can write an identical value every frame.
    world
        .entity_mut(entity)
        .insert(PreviousGlobalTransform(Affine3A::IDENTITY));
    assert!(matches(&mut world, &inputs));
    world
        .entity_mut(entity)
        .insert(GlobalTransform::from_translation(Vec3::X));
    assert!(!matches(&mut world, &inputs));
    inputs = capture(&mut world);
    world
        .entity_mut(entity)
        .insert(PreviousGlobalTransform(Affine3A::from_translation(Vec3::X)));
    assert!(
        !matches(&mut world, &inputs),
        "motion history must catch up after movement"
    );
    inputs = capture(&mut world);
    assert!(matches(&mut world, &inputs));
    world
        .entity_mut(entity)
        .insert(PreviousGlobalTransform(Affine3A::IDENTITY));
    inputs = capture(&mut world);
    world.entity_mut(entity).remove::<PreviousGlobalTransform>();
    assert!(
        !matches(&mut world, &inputs),
        "missing previous transform falls back to current"
    );
    for kind in 0..4 {
        inputs = capture(&mut world);
        match kind {
            0 => {
                world.entity_mut(entity).insert(RenderLayers::layer(3));
            }
            1 => {
                world.entity_mut(entity).insert(NotShadowCaster);
            }
            2 => {
                world.entity_mut(entity).insert(NotShadowReceiver);
            }
            _ => {
                world.entity_mut(entity).insert(MeshletDoubleSided);
            }
        }
        assert!(!matches(&mut world, &inputs));
        inputs = capture(&mut world);
        match kind {
            0 => {
                world.entity_mut(entity).remove::<RenderLayers>();
            }
            1 => {
                world.entity_mut(entity).remove::<NotShadowCaster>();
            }
            2 => {
                world.entity_mut(entity).remove::<NotShadowReceiver>();
            }
            _ => {
                world.entity_mut(entity).remove::<MeshletDoubleSided>();
            }
        }
        assert!(!matches(&mut world, &inputs));
    }
    inputs = capture(&mut world);
    world.entity_mut(entity).insert(MeshletVisibilityCutout {
        layer: 1,
        cutoff: 0.5,
    });
    assert!(!matches(&mut world, &inputs));
    inputs = capture(&mut world);
    world
        .entity_mut(entity)
        .get_mut::<MeshletVisibilityCutout>()
        .unwrap()
        .cutoff = 0.4;
    assert!(!matches(&mut world, &inputs));
    inputs = capture(&mut world);
    world.entity_mut(entity).remove::<MeshletVisibilityCutout>();
    assert!(!matches(&mut world, &inputs));
    inputs = capture(&mut world);
    let other = world
        .spawn((MeshletMesh3d::default(), GlobalTransform::IDENTITY))
        .id();
    assert!(
        !matches(&mut world, &inputs),
        "new row cannot hide at the end"
    );
    inputs = capture(&mut world);
    world.despawn(other);
    assert!(
        !matches(&mut world, &inputs),
        "departure cannot retain stale GPU membership"
    );
    inputs = capture(&mut world);
    let assets = Assets::<MeshletMesh>::default();
    world
        .entity_mut(entity)
        .insert(MeshletMesh3d(assets.reserve_handle()));
    assert!(!matches(&mut world, &inputs));
    inputs = capture(&mut world);
    world.entity_mut(entity).remove::<MeshletMesh3d>();
    assert!(!matches(&mut world, &inputs));
}

fn scene_matches(
    world: &mut World,
    manager: &InstanceManager,
    materials: &RenderMaterialInstances,
    bindings: &RenderMaterialBindings,
) -> bool {
    manager.scene_matches(
        world.query::<Data>().iter(world),
        materials,
        bindings,
        false,
    )
}

fn insert_material(materials: &mut RenderMaterialInstances, entity: MainEntity) {
    materials.instances.insert(
        entity,
        crate::RenderMaterialInstance {
            asset_id: DUMMY_MESH_MATERIAL.untyped(),
            last_change_tick: bevy_ecs::change_detection::Tick::new(1),
        },
    );
}

fn spawn_meshlet(world: &mut World) -> Entity {
    world
        .spawn((MeshletMesh3d::default(), GlobalTransform::IDENTITY))
        .id()
}

#[test]
fn cached_slots_require_readiness_and_material_mutation_receipts() {
    let mut world = World::new();
    world.spawn((MeshletMesh3d::default(), GlobalTransform::IDENTITY));
    let mut manager = capture(&mut world);
    let mut materials = RenderMaterialInstances::default();
    let bindings = RenderMaterialBindings::default();
    manager.material_receipt = materials.instances.receipt();
    assert!(scene_matches(&mut world, &manager, &materials, &bindings));
    let slot = manager.query_slots[0];
    manager.slots.deactivate(slot);
    assert!(!scene_matches(&mut world, &manager, &materials, &bindings));
    manager.slots.activate(slot);
    assert!(scene_matches(&mut world, &manager, &materials, &bindings));
    let entity = manager.slots.entity(slot.index).unwrap().into();
    insert_material(&mut materials, entity);
    assert!(!scene_matches(&mut world, &manager, &materials, &bindings));
    manager.material_receipt = materials.instances.receipt();
    assert!(scene_matches(&mut world, &manager, &materials, &bindings));
    materials.instances = crate::material_instances::MaterialInstanceMap::default();
    assert!(!scene_matches(&mut world, &manager, &materials, &bindings));
}

#[test]
fn cached_query_reordering_repairs_slots_without_publication() {
    let mut world = World::new();
    let a = spawn_meshlet(&mut world);
    let b = spawn_meshlet(&mut world);
    let mut manager = capture(&mut world);
    let old = manager.query_slots.clone();
    manager.query_slots.reverse();
    assert!(!matches(&mut world, &manager));
    let publication = manager.slots.publication;
    manager.slots.begin();
    let first = manager.observe_query(0, a);
    let second = manager.observe_query(1, b);
    assert_eq!([first, second], old.as_slice());
    assert_eq!(manager.slots.publication, publication);
    assert!(matches(&mut world, &manager));
    let mut stale = first;
    stale.generation += 1;
    manager.query_slots[0] = stale;
    assert!(!matches(&mut world, &manager));
    assert_eq!(manager.observe_query(0, a), first);
    assert!(matches(&mut world, &manager));
}

#[test]
fn persistent_binding_receipt_invalidates_slot_changes_additions_and_removals() {
    let id = DUMMY_MESH_MATERIAL.untyped();
    let mut bindings = RenderMaterialBindings::default();
    bindings.insert(id, MaterialBindingId::default());
    let mut slots = HashMap::default();
    slots.insert(id, 0);
    assert!(binding_slots_match(&slots, &bindings));
    bindings.get_mut(&id).unwrap().slot.0 = 7;
    assert!(!binding_slots_match(&slots, &bindings));
    slots.insert(id, 7);
    assert!(binding_slots_match(&slots, &bindings));
    bindings.clear();
    assert!(!binding_slots_match(&slots, &bindings));
    slots.clear();
    assert!(binding_slots_match(&slots, &bindings));
}
