//! Production publication ownership, independent of the domain producing meshes.
use bevy_asset::Assets;
use bevy_camera::primitives::Aabb;
use bevy_ecs::{entity::Entity, world::World};
use bevy_math::{Vec3A, primitives::Cuboid};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_render::scene_readiness::SceneGeometryReadiness;
use bevy_solarik::{
    material::{MaterialModel, MaterialRegistration},
    prelude::RaytracingMesh3d,
    publication::{
        ActiveProduct, PreparedProduct, ProductState, PublicationError, RasterPipelineReadiness,
        Replacement,
    },
};
use bevy_transform::components::Transform;

fn world() -> World {
    let mut world = World::new();
    world.init_resource::<Assets<Mesh>>();
    world.init_resource::<Assets<StandardMaterial>>();
    world.init_resource::<SceneGeometryReadiness>();
    world
}
fn prepared(world: &mut World, size: f32) -> PreparedProduct {
    let mut mesh = Mesh::from(Cuboid::from_length(size));
    mesh.enable_raytracing = true;
    let base = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let material = MaterialRegistration::new(world, base, MaterialModel::Standard).unwrap();
    PreparedProduct::new(mesh, material).unwrap()
}
fn stage(world: &mut World, entity: Entity, size: f32) -> Replacement {
    prepared(world, size).stage(world, entity).unwrap()
}
fn ready(world: &World, next: &Replacement) {
    let ready = world.resource::<SceneGeometryReadiness>();
    ready.publish(next.mesh().id().untyped(), true);
    ready.publish(next.material().base().id().untyped(), true);
    world
        .resource::<RasterPipelineReadiness>()
        .publish(next.mesh().id(), true);
}
fn initial(world: &mut World) -> Entity {
    let entity = world.spawn(Transform::from_xyz(7., 8., 9.)).id();
    let mut first = stage(world, entity, 2.);
    ready(world, &first);
    assert!(first.try_publish(world).unwrap().is_some());
    entity
}

#[test]
fn incomplete_product_preserves_both_paths_until_one_complete_commit() {
    let mut world = world();
    let entity = initial(&mut world);
    let old = world.get::<Mesh3d>(entity).unwrap().0.clone();
    let old_material = world
        .get::<MeshMaterial3d<StandardMaterial>>(entity)
        .unwrap()
        .0
        .clone();
    let old_epoch = world.get::<ActiveProduct>(entity).unwrap().epoch();
    let transform = *world.get::<Transform>(entity).unwrap();
    let mut next = stage(&mut world, entity, 8.);
    assert_eq!(next.state(&world), ProductState::Uploading);
    assert!(next.try_publish(&mut world).unwrap().is_none());
    let readiness = world.resource::<SceneGeometryReadiness>().clone();
    readiness.publish(next.mesh().id().untyped(), true);
    assert!(next.try_publish(&mut world).unwrap().is_none());
    readiness.publish(next.material().base().id().untyped(), true);
    assert!(
        next.try_publish(&mut world).unwrap().is_none(),
        "asset readiness alone cannot publish an uncompiled pipeline"
    );
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, old);
    assert_eq!(world.get::<RaytracingMesh3d>(entity).unwrap().0, old);
    assert_eq!(
        world
            .get::<MeshMaterial3d<StandardMaterial>>(entity)
            .unwrap()
            .0,
        old_material
    );
    assert_eq!(
        world.get::<ActiveProduct>(entity).unwrap().epoch(),
        old_epoch
    );
    assert_eq!(world.get::<Aabb>(entity).unwrap().half_extents, Vec3A::ONE);
    ready(&world, &next);
    assert_eq!(next.state(&world), ProductState::GpuReady);
    let committed = next.try_publish(&mut world).unwrap().unwrap();
    assert_eq!(next.state(&world), ProductState::Active);
    assert_eq!(
        world.get::<ActiveProduct>(entity).unwrap().epoch(),
        committed.epoch
    );
    assert_ne!(committed.epoch, old_epoch);
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, *next.mesh());
    assert_eq!(
        world.get::<RaytracingMesh3d>(entity).unwrap().0,
        *next.mesh()
    );
    assert_eq!(
        world.get::<Aabb>(entity).unwrap().half_extents,
        Vec3A::splat(4.)
    );
    assert_eq!(*world.get::<Transform>(entity).unwrap(), transform);
    let retired = committed.retired.unwrap();
    assert_eq!(retired.state(), ProductState::Retiring);
    assert_eq!(*retired.mesh(), old);
    assert!(world.resource::<Assets<Mesh>>().contains(old.id()));
    assert!(
        readiness.ready(old.id().untyped()),
        "a retained consumer still owns old readiness"
    );
}

#[test]
fn superseded_and_cancelled_work_never_removes_last_active_product() {
    let mut world = world();
    let entity = initial(&mut world);
    let old = world.get::<Mesh3d>(entity).unwrap().0.clone();
    let epoch = world.get::<ActiveProduct>(entity).unwrap().epoch();
    let mut stale = stage(&mut world, entity, 4.);
    ready(&world, &stale);
    let current = stage(&mut world, entity, 6.);
    assert_eq!(stale.state(&world), ProductState::Stale);
    assert_eq!(
        stale.try_publish(&mut world).unwrap_err(),
        PublicationError::Stale
    );
    current.cancel();
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, old);
    assert_eq!(world.get::<ActiveProduct>(entity).unwrap().epoch(), epoch);
    assert!(world.resource::<Assets<Mesh>>().contains(old.id()));
    let mut later = stage(&mut world, entity, 8.);
    ready(&world, &later);
    assert!(later.try_publish(&mut world).unwrap().is_some());
}

#[test]
fn missing_cpu_asset_and_external_owner_change_fail_without_partial_swap() {
    let mut world = world();
    let entity = initial(&mut world);
    let old = world.get::<Mesh3d>(entity).unwrap().0.clone();
    let mut missing = stage(&mut world, entity, 4.);
    ready(&world, &missing);
    world
        .resource_mut::<Assets<Mesh>>()
        .remove(missing.mesh().id());
    assert_eq!(missing.state(&world), ProductState::Uploading);
    assert_eq!(
        missing.try_publish(&mut world).unwrap_err(),
        PublicationError::MissingMesh
    );
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, old);
    assert_eq!(world.get::<RaytracingMesh3d>(entity).unwrap().0, old);
    let mut conflict = stage(&mut world, entity, 6.);
    ready(&world, &conflict);
    world.entity_mut(entity).remove::<RaytracingMesh3d>();
    assert_eq!(
        conflict.try_publish(&mut world).unwrap_err(),
        PublicationError::OwnerChanged
    );
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, old);
    assert_eq!(world.get::<Aabb>(entity).unwrap().half_extents, Vec3A::ONE);
}

#[test]
fn empty_or_nonfinite_geometry_is_an_error_not_physical_emptiness() {
    let mut world = world();
    let valid = prepared(&mut world, 2.);
    assert_eq!(valid.state(), ProductState::Prepared);
    let mut mesh = Mesh::from(Cuboid::default());
    mesh.enable_raytracing = true;
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[f32::NAN, 0., 0.]; 24]);
    let base = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let material = MaterialRegistration::new(&mut world, base, MaterialModel::Standard).unwrap();
    assert_eq!(
        PreparedProduct::new(mesh, material).unwrap_err(),
        PublicationError::InvalidGeometry
    );
}

#[test]
fn unsupported_static_variant_keeps_the_last_drawable_owner() {
    use bevy_camera::visibility::VisibilityRange;
    let mut world = world();
    let entity = initial(&mut world);
    let old = world.get::<Mesh3d>(entity).unwrap().0.clone();
    let mut next = stage(&mut world, entity, 4.);
    ready(&world, &next);
    world.entity_mut(entity).insert(VisibilityRange::default());
    assert_eq!(
        next.try_publish(&mut world).unwrap_err(),
        PublicationError::UnsupportedRasterState
    );
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, old);
    assert_eq!(world.get::<RaytracingMesh3d>(entity).unwrap().0, old);
    assert_eq!(
        prepared(&mut world, 6.)
            .stage(&mut world, entity)
            .unwrap_err(),
        PublicationError::UnsupportedRasterState
    );
}

#[test]
fn revoked_pipeline_receipt_preserves_old_coverage() {
    let mut world = world();
    let entity = initial(&mut world);
    let old = world.get::<Mesh3d>(entity).unwrap().0.clone();
    let mut next = stage(&mut world, entity, 4.);
    ready(&world, &next);
    world
        .resource::<RasterPipelineReadiness>()
        .publish(next.mesh().id(), false);
    assert_eq!(next.state(&world), ProductState::Uploading);
    assert!(next.try_publish(&mut world).unwrap().is_none());
    assert_eq!(world.get::<Mesh3d>(entity).unwrap().0, old);
    next.cancel();
    assert_eq!(world.get::<RaytracingMesh3d>(entity).unwrap().0, old);
}

#[test]
fn actual_last_asset_owner_retires_mesh_requests_not_one_consumer() {
    use bevy_app::App;
    use bevy_asset::{AssetApp, AssetPlugin};
    use bevy_solarik::{material::MaterialRegistrationPlugin, publication::PublicationPlugin};
    let mut app = App::new();
    app.add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .add_plugins((MaterialRegistrationPlugin, PublicationPlugin));
    let first = initial(app.world_mut());
    let mesh = app.world().get::<Mesh3d>(first).unwrap().0.clone();
    let id = mesh.id();
    let second = app.world_mut().spawn(Mesh3d(mesh)).id();
    app.update();
    app.world_mut().despawn(first);
    for _ in 0..5 {
        app.update();
    }
    assert!(app.world().resource::<Assets<Mesh>>().contains(id));
    assert!(
        app.world()
            .resource::<SceneGeometryReadiness>()
            .ready(id.untyped())
    );
    app.world_mut().despawn(second);
    for _ in 0..5 {
        app.update();
    }
    assert!(!app.world().resource::<Assets<Mesh>>().contains(id));
    assert!(
        !app.world()
            .resource::<SceneGeometryReadiness>()
            .requested::<Mesh>()
            .contains(&id)
    );
}

#[test]
fn adopting_managed_geometry_replaces_material_and_preserves_other_consumers() {
    use bevy_solarik::{lommel::LommelMaterial, scene::collimated::RaytracingMaterial3d};
    let mut world = world();
    let entity = initial(&mut world);
    // A managed surface may predate this adapter (ordinary terrain bank/tile).
    world.entity_mut(entity).remove::<ActiveProduct>();
    let old = world.get::<Mesh3d>(entity).unwrap().0.clone();
    let other = world.spawn(Mesh3d(old.clone())).id();
    let base = world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let material =
        MaterialRegistration::new(&mut world, base.clone(), MaterialModel::Lommel).unwrap();
    let mut mesh = Mesh::from(Cuboid::from_length(4.));
    mesh.enable_raytracing = true;
    let mut replacement = PreparedProduct::new(mesh, material)
        .unwrap()
        .stage(&mut world, entity)
        .unwrap();
    ready(&world, &replacement);
    let committed = replacement.try_publish(&mut world).unwrap().unwrap();
    assert!(
        world
            .get::<MeshMaterial3d<StandardMaterial>>(entity)
            .is_none()
    );
    assert!(
        world
            .get::<MeshMaterial3d<LommelMaterial>>(entity)
            .is_some()
    );
    assert_eq!(world.get::<RaytracingMaterial3d>(entity).unwrap().0, base);
    drop(committed);
    drop(replacement);
    assert_eq!(world.get::<Mesh3d>(other).unwrap().0, old);
    assert!(world.resource::<Assets<Mesh>>().contains(old.id()));
}

#[test]
fn render_only_extraction_waits_for_gpu_and_retains_readiness_until_last_owner() {
    use bevy_app::App;
    use bevy_asset::{AssetApp, AssetPlugin, RenderAssetUsages};
    use bevy_solarik::{material::MaterialRegistrationPlugin, publication::PublicationPlugin};
    let mut app = App::new();
    app.add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .add_plugins((MaterialRegistrationPlugin, PublicationPlugin));
    let entity = initial(app.world_mut());
    let old = app.world().get::<Mesh3d>(entity).unwrap().0.clone();
    let base = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let material =
        MaterialRegistration::new(app.world_mut(), base, MaterialModel::Standard).unwrap();
    let mut mesh = Mesh::from(Cuboid::from_length(8.));
    mesh.enable_raytracing = true;
    mesh.asset_usage = RenderAssetUsages::RENDER_WORLD;
    let mut next = PreparedProduct::new(mesh, material)
        .unwrap()
        .stage(app.world_mut(), entity)
        .unwrap();
    let id = next.mesh().id();
    // Simulate the source removal performed by Bevy extraction, not GPU completion.
    app.world_mut().resource_mut::<Assets<Mesh>>().remove(id);
    app.update();
    assert_eq!(next.state(app.world()), ProductState::Uploading);
    assert!(next.try_publish(app.world_mut()).unwrap().is_none());
    assert_eq!(app.world().get::<Mesh3d>(entity).unwrap().0, old);
    ready(app.world(), &next);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(next.state(app.world()), ProductState::GpuReady);
    assert!(next.try_publish(app.world_mut()).unwrap().is_some());
    assert_eq!(
        app.world().get::<Aabb>(entity).unwrap().half_extents,
        Vec3A::splat(4.)
    );
    // A later generation can adopt this active product without its CPU source.
    let later = stage(app.world_mut(), entity, 10.);
    later.cancel();
    drop(next);
    app.update();
    assert!(
        app.world()
            .resource::<SceneGeometryReadiness>()
            .ready(id.untyped())
    );
    app.world_mut().despawn(entity);
    for _ in 0..5 {
        app.update();
    }
    assert!(
        !app.world()
            .resource::<SceneGeometryReadiness>()
            .requested::<Mesh>()
            .contains(&id)
    );
}
