use super::{RaytracingMesh3d, collimated::RaytracingMaterial3d};
use bevy_asset::{AssetId, Assets};
use bevy_derive::Deref;
use bevy_ecs::{
    lifecycle::RemovedComponents,
    resource::Resource,
    system::{Commands, Query},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{Extract, extract_resource::ExtractResource, sync_world::RenderEntity};
use bevy_transform::components::GlobalTransform;

pub fn extract_raytracing_scene(
    instances: Extract<
        Query<(
            RenderEntity,
            &RaytracingMesh3d,
            Option<&MeshMaterial3d<StandardMaterial>>,
            Option<&RaytracingMaterial3d>,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
        )>,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
    existing: Query<(
        &RaytracingMesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &GlobalTransform,
        Option<&PreviousGlobalTransform>,
    )>,
) {
    let _cpu_profile = bevy_render::diagnostic::profile_scope("scene.extract");
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }

    for (render_entity, mesh, material, override_material, transform, previous_frame_transform) in
        &instances
    {
        let Some(material) = override_material
            .map(|m| MeshMaterial3d(m.0.clone()))
            .or_else(|| material.cloned())
        else {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
            continue;
        };
        if let Ok((old_mesh, old_material, old_transform, old_previous)) =
            existing.get(render_entity)
            && old_mesh.id() == mesh.id()
            && old_material.id() == material.id()
            && old_transform.affine() == transform.affine()
            && old_previous.map(|t| t.0) == previous_frame_transform.map(|t| t.0)
        {
            continue;
        }
        let mut commands = commands.entity(render_entity);
        if previous_frame_transform.is_none() {
            commands.remove::<PreviousGlobalTransform>();
        }

        match previous_frame_transform.cloned() {
            Some(previous_frame_transform) => commands.insert((
                mesh.clone(),
                material.clone(),
                *transform,
                previous_frame_transform,
            )),
            None => commands.insert((mesh.clone(), material.clone(), *transform)),
        };
    }
}

#[derive(Resource, Deref, Default)]
pub struct StandardMaterialAssets(HashMap<AssetId<StandardMaterial>, StandardMaterial>);

impl ExtractResource for StandardMaterialAssets {
    type Source = Assets<StandardMaterial>;

    fn extract_resource(source: &Self::Source) -> Self {
        Self(
            source
                .iter()
                .map(|(asset_id, material)| (asset_id, material.clone()))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{change_detection::DetectChanges, schedule::Schedule, world::World};
    use bevy_math::{Affine3A, Vec3};
    use bevy_render::MainWorld;

    #[test]
    fn extraction_reuses_unchanged_data_and_tracks_motion_and_material_lifetimes() {
        let mut render = World::new();
        let render_entity = render.spawn_empty().id();
        let mut materials = Assets::<StandardMaterial>::default();
        let material = materials.add(StandardMaterial::default());
        let override_material = materials.add(StandardMaterial::default());
        let mut main = MainWorld::default();
        let main_entity = main
            .spawn((
                RenderEntity::from(render_entity),
                RaytracingMesh3d::default(),
                MeshMaterial3d(material.clone()),
                GlobalTransform::IDENTITY,
                PreviousGlobalTransform(Affine3A::IDENTITY),
            ))
            .id();
        render.insert_resource(main);
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_raytracing_scene);
        let extract = |render: &mut World, schedule: &mut Schedule| {
            render.increment_change_tick();
            schedule.run(render);
        };
        extract(&mut render, &mut schedule);
        let first_tick = render
            .entity(render_entity)
            .get_ref::<GlobalTransform>()
            .unwrap()
            .last_changed();
        extract(&mut render, &mut schedule);
        assert_eq!(
            render
                .entity(render_entity)
                .get_ref::<GlobalTransform>()
                .unwrap()
                .last_changed(),
            first_tick,
            "unchanged extraction must not dirty scene transforms"
        );

        let moved = GlobalTransform::from_translation(Vec3::new(4.0, 5.0, 6.0));
        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .insert(moved);
        extract(&mut render, &mut schedule);
        assert_eq!(
            render
                .get::<GlobalTransform>(render_entity)
                .unwrap()
                .affine(),
            moved.affine()
        );
        assert_eq!(
            render
                .get::<PreviousGlobalTransform>(render_entity)
                .unwrap()
                .0,
            Affine3A::IDENTITY
        );
        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .insert(PreviousGlobalTransform(moved.affine()));
        extract(&mut render, &mut schedule);
        assert_eq!(
            render
                .get::<PreviousGlobalTransform>(render_entity)
                .unwrap()
                .0,
            moved.affine()
        );

        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .insert(RaytracingMaterial3d(override_material.clone()));
        extract(&mut render, &mut schedule);
        assert_eq!(
            render
                .get::<MeshMaterial3d<StandardMaterial>>(render_entity)
                .unwrap()
                .id(),
            override_material.id()
        );
        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .remove::<(RaytracingMaterial3d, PreviousGlobalTransform)>();
        extract(&mut render, &mut schedule);
        assert_eq!(
            render
                .get::<MeshMaterial3d<StandardMaterial>>(render_entity)
                .unwrap()
                .id(),
            material.id()
        );
        assert!(
            render
                .get::<PreviousGlobalTransform>(render_entity)
                .is_none()
        );

        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .remove::<MeshMaterial3d<StandardMaterial>>();
        extract(&mut render, &mut schedule);
        assert!(
            render.get::<RaytracingMesh3d>(render_entity).is_none(),
            "missing material must stop stale ray participation"
        );
        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .insert(MeshMaterial3d(material));
        extract(&mut render, &mut schedule);
        assert!(render.get::<RaytracingMesh3d>(render_entity).is_some());
        render
            .resource_mut::<MainWorld>()
            .entity_mut(main_entity)
            .remove::<RaytracingMesh3d>();
        extract(&mut render, &mut schedule);
        assert!(
            render.get::<RaytracingMesh3d>(render_entity).is_none(),
            "removed mesh must stop ray participation"
        );
    }
}
