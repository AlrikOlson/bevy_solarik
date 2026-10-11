//! Ownership and replacement through the production material registration API.
use bevy_asset::{Assets, Handle};
use bevy_ecs::world::World;
use bevy_image::Image;
use bevy_math::{DVec3, Vec4};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_solarik::{
    gaussian::{GaussianDielectric, GaussianMaterial, GaussianRayMaterials},
    lommel::{LommelMaterial, LommelRayMaterials},
    material::{self, MaterialModel, MaterialRegistration, RegistrationError},
    scene::collimated::RaytracingMaterial3d,
    surface_detail::{DetailCoordinates, DetailedMaterial, DetailedRayMaterials, SurfaceDetail},
};

fn world() -> World {
    let mut world = World::new();
    world.init_resource::<Assets<StandardMaterial>>();
    world.init_resource::<Assets<Image>>();
    world
}

fn base(world: &mut World) -> Handle<StandardMaterial> {
    world
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default())
}

fn detail() -> SurfaceDetail {
    SurfaceDetail {
        coordinates: DetailCoordinates::new(DVec3::ZERO, [1.0; 8]).unwrap(),
        colour: Handle::default(),
        detail: Handle::default(),
        coverage0: Handle::default(),
        coverage1: Handle::default(),
        gaussian_parameters: Vec4::ZERO,
        gaussian_mask: None,
        lommel: Vec4::X,
        meso_colour: Handle::default(),
        meso_detail: Handle::default(),
    }
}

#[test]
fn shared_consumers_keep_mapping_and_images_until_last_release() {
    let mut world = world();
    world.init_resource::<bevy_render::scene_readiness::SceneGeometryReadiness>();
    let mask = world.resource_mut::<Assets<Image>>().add(Image::default());
    let base = base(&mut world);
    let optics = GaussianDielectric::new(1.333, 0.2, mask.clone()).unwrap();
    let registration = MaterialRegistration::new(
        &mut world,
        base.clone(),
        MaterialModel::Gaussian(optics.clone()),
    )
    .unwrap();
    let a = world.spawn_empty().id();
    let b = world.spawn_empty().id();
    assert!(registration.bind(&mut world, a));
    assert!(registration.bind(&mut world, b));
    world
        .resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
        .publish(base.id().untyped(), true);
    let raster = world
        .get::<MeshMaterial3d<GaussianMaterial>>(b)
        .unwrap()
        .0
        .clone();
    drop(registration);
    world.despawn(a);
    material::collect(&mut world);
    assert!(
        world
            .resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
            .ready(base.id().untyped())
    );
    assert_eq!(
        world.resource::<GaussianRayMaterials>().0[&base.id()].parameters,
        optics.parameters
    );
    assert!(world.resource::<Assets<Image>>().contains(mask.id()));
    assert!(
        world
            .resource::<Assets<GaussianMaterial>>()
            .contains(raster.id())
    );
    assert_eq!(world.get::<RaytracingMaterial3d>(b).unwrap().0, base);
    world.despawn(b);
    material::collect(&mut world);
    assert!(
        !world
            .resource::<GaussianRayMaterials>()
            .0
            .contains_key(&base.id())
    );
    assert!(
        !world
            .resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
            .ready(base.id().untyped())
    );
    // Strong handles held by another subsystem are never force-deleted.
    assert!(
        world
            .resource::<Assets<StandardMaterial>>()
            .contains(base.id())
    );
    assert!(
        world
            .resource::<Assets<GaussianMaterial>>()
            .contains(raster.id())
    );
    assert!(world.resource::<Assets<Image>>().contains(mask.id()));
}

#[test]
fn last_owner_releases_assets_through_bevy_tracking() {
    use bevy_app::App;
    use bevy_asset::{AssetApp, AssetPlugin};
    let mut app = App::new();
    app.add_plugins(AssetPlugin::default())
        .init_asset::<Image>()
        .init_asset::<StandardMaterial>()
        .init_asset::<GaussianMaterial>()
        .init_asset::<LommelMaterial>()
        .init_asset::<DetailedMaterial>()
        .add_plugins(material::MaterialRegistrationPlugin);
    let world = app.world_mut();
    let mask = world.resource_mut::<Assets<Image>>().add(Image::default());
    let mask_id = mask.id();
    let base = base(world);
    let base_id = base.id();
    let optics = GaussianDielectric::new(1.333, 0.2, mask).unwrap();
    let registration =
        MaterialRegistration::new(world, base, MaterialModel::Gaussian(optics)).unwrap();
    let entity = world.spawn_empty().id();
    registration.bind(world, entity);
    let raster_id = world
        .get::<MeshMaterial3d<GaussianMaterial>>(entity)
        .unwrap()
        .id();
    drop(registration);
    app.update();
    assert!(app.world().resource::<Assets<Image>>().contains(mask_id));
    app.world_mut().despawn(entity);
    // Bevy propagates handle drops across each asset dependency layer.
    for _ in 0..5 {
        app.update();
    }
    assert!(
        !app.world()
            .resource::<Assets<StandardMaterial>>()
            .contains(base_id)
    );
    assert!(
        !app.world()
            .resource::<Assets<GaussianMaterial>>()
            .contains(raster_id)
    );
    assert!(!app.world().resource::<Assets<Image>>().contains(mask_id));
}

#[test]
fn every_optical_transition_has_one_raster_and_matching_ray_proxy() {
    let mut world = world();
    let entity = world.spawn_empty().id();
    let models = [
        MaterialModel::Gaussian(GaussianDielectric::new(1.333, 0.2, Handle::default()).unwrap()),
        MaterialModel::Lommel,
        MaterialModel::Detailed(Box::new(detail())),
        MaterialModel::Standard,
    ];
    let mut old = None;
    for (index, model) in models.into_iter().enumerate() {
        let base = base(&mut world);
        let registration = MaterialRegistration::new(&mut world, base.clone(), model).unwrap();
        world
            .entity_mut(entity)
            .insert(MeshMaterial3d(base.clone()));
        material::sync(&mut world);
        let count = usize::from(
            world
                .get::<MeshMaterial3d<StandardMaterial>>(entity)
                .is_some(),
        ) + usize::from(
            world
                .get::<MeshMaterial3d<GaussianMaterial>>(entity)
                .is_some(),
        ) + usize::from(
            world
                .get::<MeshMaterial3d<LommelMaterial>>(entity)
                .is_some(),
        ) + usize::from(
            world
                .get::<MeshMaterial3d<DetailedMaterial>>(entity)
                .is_some(),
        );
        assert_eq!(count, 1, "transition {index}");
        if index != 3 {
            assert_eq!(world.get::<RaytracingMaterial3d>(entity).unwrap().0, base);
        } else {
            assert!(world.get::<RaytracingMaterial3d>(entity).is_none());
        }
        drop(old.replace(registration));
        material::collect(&mut world);
        assert_eq!(
            world.resource::<GaussianRayMaterials>().0.len(),
            usize::from(index == 0)
        );
        assert_eq!(
            world.resource::<LommelRayMaterials>().0.len(),
            usize::from(index == 1 || index == 2)
        );
        assert_eq!(
            world.resource::<DetailedRayMaterials>().0.len(),
            usize::from(index == 2)
        );
    }
}

#[test]
fn duplicate_and_missing_registration_do_not_overwrite_live_optics() {
    let mut world = world();
    assert_eq!(
        MaterialRegistration::new(&mut world, Handle::default(), MaterialModel::Standard)
            .unwrap_err(),
        RegistrationError::MissingBase
    );
    let base = base(&mut world);
    let first = MaterialRegistration::new(&mut world, base.clone(), MaterialModel::Lommel).unwrap();
    assert_eq!(
        MaterialRegistration::new(&mut world, base.clone(), MaterialModel::Standard).unwrap_err(),
        RegistrationError::AlreadyRegistered
    );
    assert!(
        world
            .resource::<LommelRayMaterials>()
            .0
            .contains(&base.id())
    );
    drop(first);
    material::collect(&mut world);
    let second =
        MaterialRegistration::new(&mut world, base.clone(), MaterialModel::Standard).unwrap();
    assert_eq!(second.base(), &base);
    assert!(
        !world
            .resource::<LommelRayMaterials>()
            .0
            .contains(&base.id())
    );
}

#[test]
fn unregistered_standard_replacement_removes_old_optical_components() {
    let mut world = world();
    let first = base(&mut world);
    let registration = MaterialRegistration::new(&mut world, first, MaterialModel::Lommel).unwrap();
    let entity = world.spawn_empty().id();
    registration.bind(&mut world, entity);
    let plain = base(&mut world);
    world
        .entity_mut(entity)
        .insert(MeshMaterial3d(plain.clone()));
    material::sync(&mut world);
    assert_eq!(
        world
            .get::<MeshMaterial3d<StandardMaterial>>(entity)
            .unwrap()
            .0,
        plain
    );
    assert!(
        world
            .get::<MeshMaterial3d<LommelMaterial>>(entity)
            .is_none()
    );
    assert!(world.get::<RaytracingMaterial3d>(entity).is_none());
    assert!(world.get::<MaterialRegistration>(entity).is_none());
}
