//! Real ECS removals through the production material sweep.
use super::*;
use bevy_asset::Handle;
use bevy_ecs::system::RunSystemOnce;
use bevy_render::MainWorld;

#[test]
fn material_sweep_preserves_virtual_owner_but_retires_removed_materials() {
    let mut main = World::new();
    let mut materials = Assets::<StandardMaterial>::default();
    let material_handle = materials.add(StandardMaterial::default());
    let virtual_owner = main
        .spawn((
            Mesh3d(Handle::default()),
            meshlet::MeshletMesh3d(Handle::default()),
            MeshMaterial3d(material_handle.clone()),
        ))
        .id();
    let retired = main.spawn(Mesh3d(Handle::default())).id();
    let replaced = main.spawn(Mesh3d(Handle::default())).id();
    for entity in [virtual_owner, retired, replaced] {
        main.entity_mut(entity).remove::<Mesh3d>();
    }
    let material = material_handle.id().untyped();
    let mut instances = RenderMaterialInstances {
        current_change_tick: Tick::new(1),
        ..Default::default()
    };
    for (entity, tick) in [(virtual_owner, 0), (retired, 0), (replaced, 1)] {
        instances.instances.insert(
            entity.into(),
            RenderMaterialInstance {
                asset_id: material,
                last_change_tick: Tick::new(tick),
            },
        );
    }
    let mut render = World::new();
    let mut main_world = MainWorld::default();
    *main_world = main;
    render.insert_resource(main_world);
    render.insert_resource(instances);
    render
        .run_system_once(late_sweep_material_instances)
        .unwrap();
    let instances = render.resource::<RenderMaterialInstances>();
    assert_eq!(
        instances.mesh_material(virtual_owner.into()),
        material,
        "removing ordinary raster must retain the live virtual mesh material"
    );
    assert!(!instances.instances.contains_key(&MainEntity::from(retired)));
    assert!(
        instances
            .instances
            .contains_key(&MainEntity::from(replaced)),
        "same-frame replacement still owns membership"
    );
    render
        .resource_mut::<MainWorld>()
        .entity_mut(virtual_owner)
        .remove::<MeshMaterial3d<StandardMaterial>>();
    render
        .run_system_once(early_sweep_material_instances::<StandardMaterial>)
        .unwrap();
    assert!(
        !render
            .resource::<RenderMaterialInstances>()
            .instances
            .contains_key(&MainEntity::from(virtual_owner)),
        "retaining a virtual mesh does not retain a removed material"
    );
}
