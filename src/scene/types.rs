use bevy_asset::Handle;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, prelude::ReflectComponent, template::FromTemplate};
use bevy_mesh::Mesh;
use bevy_pbr::StandardMaterial;
use bevy_reflect::{Reflect, prelude::ReflectDefault};
use bevy_render::sync_world::SyncToRenderWorld;
use bevy_transform::components::Transform;
use derive_more::derive::From;

/// A mesh component used for raytracing.
///
/// The mesh used in this component must have [`Mesh::enable_raytracing`] set to true,
/// include the following vertex attributes (additional source streams are preserved): `{POSITION, NORMAL, UV_0, TANGENT}`, use [`bevy_mesh::PrimitiveTopology::TriangleList`],
/// and use [`bevy_mesh::Indices::U32`].
///
/// Use [`bevy_pbr::MeshMaterial3d<StandardMaterial>`] or [`SolarikMaterial3d`] for an extension.
#[derive(
    Component, FromTemplate, Clone, Debug, Default, Deref, DerefMut, Reflect, PartialEq, Eq, From,
)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(Transform, SyncToRenderWorld)]
pub struct RaytracingMesh3d(pub Handle<Mesh>);

/// Material for ray hits without adding a second raster material to the entity.
/// Hosts with material extensions retain their original material and gbuffer.
#[derive(Component, Clone, Debug)]
pub struct SolarikMaterial3d(pub Handle<StandardMaterial>);

/// Exclude an analytical light whose emitting lens mesh supplies its energy.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct SolarikLightOff;
