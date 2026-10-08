//! Compact root queries expand immutable parts only when the ray scene changes.
use super::binder::RayInstanceInput;
use super::{RaytracingAssembly3d, RaytracingMesh3d};
use bevy_ecs::{
    entity::Entity,
    system::{Query, SystemParam},
};
use bevy_pbr::{MeshMaterial3d, PreviousGlobalTransform, StandardMaterial};
use bevy_render::scene_slots::SceneInstance;
use bevy_transform::components::GlobalTransform;

#[derive(SystemParam)]
pub(super) struct SceneRows<'w, 's> {
    single: Query<
        'w,
        's,
        (
            Entity,
            &'static RaytracingMesh3d,
            &'static MeshMaterial3d<StandardMaterial>,
            &'static GlobalTransform,
            Option<&'static PreviousGlobalTransform>,
        ),
    >,
    assemblies: Query<
        'w,
        's,
        (
            Entity,
            &'static RaytracingAssembly3d,
            &'static GlobalTransform,
            Option<&'static PreviousGlobalTransform>,
        ),
    >,
}
impl SceneRows<'_, '_> {
    pub(super) fn len(&self) -> usize {
        self.single.iter().len() + self.assemblies.iter().map(|r| r.1.0.len()).sum::<usize>()
    }
    /// Expand only one changed root; removal naturally returns no rows.
    pub(super) fn for_root(&self, entity: Entity) -> Vec<RayInstanceInput> {
        let mut rows = Vec::new();
        if let Ok((entity, mesh, material, transform, previous)) = self.single.get(entity) {
            rows.push(RayInstanceInput {
                entity: entity.into(),
                mesh: mesh.id(),
                material: material.id(),
                transform: transform.affine(),
                previous: previous.map_or(transform.affine(), |p| p.0),
            });
        }
        if let Ok((entity, assembly, transform, previous)) = self.assemblies.get(entity) {
            let transform = transform.affine();
            let previous = previous.map_or(transform, |p| p.0);
            rows.extend(
                assembly
                    .0
                    .iter()
                    .enumerate()
                    .map(|(part, source)| RayInstanceInput {
                        entity: SceneInstance {
                            entity,
                            part: part as u32 + 1,
                        },
                        mesh: source.mesh.id(),
                        material: source.material.id(),
                        transform: transform * source.transform,
                        previous: previous * source.transform,
                    }),
            );
        }
        rows
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = RayInstanceInput> + '_ {
        let single = self
            .single
            .iter()
            .map(
                |(entity, mesh, material, transform, previous)| RayInstanceInput {
                    entity: entity.into(),
                    mesh: mesh.id(),
                    material: material.id(),
                    transform: transform.affine(),
                    previous: previous.map_or(transform.affine(), |p| p.0),
                },
            );
        single.chain(
            self.assemblies
                .iter()
                .flat_map(|(entity, assembly, transform, previous)| {
                    let transform = transform.affine();
                    let previous = previous.map_or(transform, |p| p.0);
                    assembly
                        .0
                        .iter()
                        .enumerate()
                        .map(move |(part, source)| RayInstanceInput {
                            entity: SceneInstance {
                                entity,
                                part: part as u32 + 1,
                            },
                            mesh: source.mesh.id(),
                            material: source.material.id(),
                            transform: transform * source.transform,
                            previous: previous * source.transform,
                        })
                }),
        )
    }
}
