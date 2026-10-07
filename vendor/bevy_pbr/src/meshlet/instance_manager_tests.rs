//! Exercise the production receipt against actual ECS mutations and removals.
use super::*;
use bevy_ecs::world::World;
use bevy_math::Vec3;

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

fn capture(world: &mut World) -> Vec<InstanceInput> {
    world
        .query::<Data>()
        .iter(world)
        .map(InstanceInput::from_row)
        .collect()
}
fn matches(world: &mut World, inputs: &[InstanceInput]) -> bool {
    inputs_match(inputs, world.query::<Data>().iter(world))
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
