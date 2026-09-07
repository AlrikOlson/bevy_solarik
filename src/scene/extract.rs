use super::{RaytracingMesh3d, SolarikLightOff, SolarikMaterial3d};
use bevy_asset::{AssetId, Assets};
use bevy_camera::visibility::InheritedVisibility;
use bevy_derive::Deref;
use bevy_ecs::{
    lifecycle::RemovedComponents,
    query::With,
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
            Option<&SolarikMaterial3d>,
            Option<&InheritedVisibility>,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
        )>,
    >,
    mut removed_raytracing_meshes: Extract<RemovedComponents<RaytracingMesh3d>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    for main_entity in removed_raytracing_meshes.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<RaytracingMesh3d>();
        }
    }

    for (
        render_entity,
        mesh,
        standard,
        ray_material,
        visibility,
        transform,
        previous_frame_transform,
    ) in &instances
    {
        let mut commands = commands.entity(render_entity);
        let material = ray_material
            .map(|m| MeshMaterial3d(m.0.clone()))
            .or_else(|| standard.cloned());
        let Some(material) = material.filter(|_| visibility.is_none_or(|v| v.get())) else {
            commands.remove::<RaytracingMesh3d>();
            continue;
        };

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

/// Mirror analytical-light exclusions, including removal, into the render world.
pub fn extract_light_off(
    marked: Extract<Query<RenderEntity, With<SolarikLightOff>>>,
    mut removed: Extract<RemovedComponents<SolarikLightOff>>,
    render_entities: Extract<Query<RenderEntity>>,
    mut commands: Commands,
) {
    for main_entity in removed.read() {
        if let Ok(render_entity) = render_entities.get(main_entity) {
            commands.entity(render_entity).remove::<SolarikLightOff>();
        }
    }
    for render_entity in &marked {
        commands.entity(render_entity).insert(SolarikLightOff);
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
