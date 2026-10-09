use super::*;
use crate::{
    gaussian::{GaussianDielectric, GaussianRayMaterials},
    lommel::LommelRayMaterials,
    surface_detail::{DetailCoordinates, DetailedRayMaterials, SurfaceDetail},
};
use bevy_asset::Assets;
use bevy_math::Vec4;
use bevy_render::extract_resource::ExtractResource;

struct Fixture {
    dependencies: Dependencies,
    materials: Assets<StandardMaterial>,
    used: AssetId<StandardMaterial>,
    unused: AssetId<StandardMaterial>,
    detailed: DetailedRayMaterials,
    gaussian: GaussianRayMaterials,
    lommel: LommelRayMaterials,
}
impl Fixture {
    fn new() -> Self {
        let mut materials = Assets::default();
        let used = materials.add(StandardMaterial::default()).id();
        let unused = materials.add(StandardMaterial::default()).id();
        let mut fixture = Self {
            dependencies: Dependencies::default(),
            materials,
            used,
            unused,
            detailed: DetailedRayMaterials::default(),
            gaussian: GaussianRayMaterials::default(),
            lommel: LommelRayMaterials::default(),
        };
        fixture.observe();
        fixture
    }
    fn observe(&mut self) -> Changes {
        self.dependencies.observe(
            [(Handle::<Mesh>::default().id(), self.used)].into_iter(),
            &BlasManager::default(),
            &StandardMaterialAssets::extract_resource(&self.materials),
            &RenderAssets::default(),
            (&self.detailed, &self.gaussian, &self.lommel),
            [None; 3],
        )
    }
    fn unchanged(&mut self) {
        let changes = self.observe();
        assert!(!changes.global);
        assert!(changes.materials.is_empty());
        assert!(changes.meshes.is_empty());
    }
    fn local_change(&mut self) {
        let changes = self.observe();
        assert!(
            !changes.global,
            "nonemissive material retains bounded invalidation"
        );
        assert_eq!(changes.materials, HashSet::from_iter([self.used]));
        self.unchanged();
    }
}
fn detail() -> SurfaceDetail {
    SurfaceDetail {
        coordinates: DetailCoordinates::default(),
        colour: Handle::default(),
        detail: Handle::default(),
        meso_colour: Handle::default(),
        meso_detail: Handle::default(),
        coverage0: Handle::default(),
        coverage1: Handle::default(),
        gaussian_parameters: Vec4::ZERO,
        gaussian_mask: None,
        lommel: Vec4::ZERO,
    }
}
fn water() -> GaussianDielectric {
    GaussianDielectric::new(1.333, 0.1, Handle::default()).unwrap()
}

#[test]
fn unused_material_registry_arrival_and_eviction_preserve_history() {
    let mut f = Fixture::new();
    f.detailed.0.insert(f.unused, detail());
    f.gaussian.0.insert(f.unused, water());
    f.lommel.0.insert(f.unused);
    f.unchanged();
    f.detailed.0.get_mut(&f.unused).unwrap().coordinates.phase[0].x = 0.5;
    f.gaussian.0.get_mut(&f.unused).unwrap().parameters.y = 0.7;
    f.unchanged();
    f.detailed.0.clear();
    f.gaussian.0.clear();
    f.lommel.0.clear();
    f.unchanged();
}

#[test]
fn used_detail_water_and_disk_parameters_invalidate_only_their_material() {
    let mut f = Fixture::new();
    f.detailed.0.insert(f.used, detail());
    f.local_change();
    assert!(
        !f.dependencies.materials[&f.used].ready,
        "unready extension textures are tracked"
    );
    for edit in [
        |d: &mut SurfaceDetail| d.coordinates.phase[0].x = 0.25,
        |d: &mut SurfaceDetail| d.coordinates.cell[0].x = 42,
        |d: &mut SurfaceDetail| d.coordinates.meso_phase[1].w = 0.125,
        |d: &mut SurfaceDetail| d.coordinates.meso_cell[2].z = -17,
        |d: &mut SurfaceDetail| d.gaussian_parameters = Vec4::ONE,
        |d: &mut SurfaceDetail| d.gaussian_mask = Some(Handle::default()),
        |d: &mut SurfaceDetail| d.lommel = Vec4::X,
    ] {
        edit(f.detailed.0.get_mut(&f.used).unwrap());
        f.local_change();
    }
    let mut images = Assets::<Image>::default();
    f.detailed.0.get_mut(&f.used).unwrap().gaussian_mask = Some(images.add(Image::default()));
    f.local_change();
    f.gaussian.0.insert(f.used, water());
    f.local_change();
    f.gaussian.0.get_mut(&f.used).unwrap().parameters.y = 0.6;
    f.local_change();
    f.gaussian.0.get_mut(&f.used).unwrap().mask = images.add(Image::default());
    f.local_change();
    f.lommel.0.insert(f.used);
    f.local_change();
    f.lommel.0.remove(&f.used);
    f.local_change();
    f.detailed.0.remove(&f.used);
    f.local_change();
    f.gaussian.0.remove(&f.used);
    f.local_change();
    assert!(f.dependencies.materials[&f.used].ready);
}

#[test]
fn emitter_changes_and_unknown_materials_still_invalidate_globally() {
    let mut f = Fixture::new();
    f.materials.get_mut(f.used).unwrap().emissive.red = 1.0;
    assert!(f.observe().global);
    f.gaussian.0.insert(f.used, water());
    assert!(f.observe().global);
    f.materials.get_mut(f.used).unwrap().emissive.red = 0.0;
    assert!(
        f.observe().global,
        "the previous emitter remains a global dependency"
    );
    f.unchanged();
    f.materials.remove(f.used);
    assert!(
        f.observe().global,
        "an unknown used material must fail closed"
    );
}
