//! Physically sized scanned detail, independent of mesh UV resolution.
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetId, Handle, embedded_asset};
use bevy_ecs::resource::Resource;
use bevy_image::Image;
use bevy_math::{DVec3, IVec4, UVec4, Vec3, Vec4};
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_reflect::TypePath;
use bevy_render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy_render::render_resource::{AsBindGroup, ShaderType};
use bevy_shader::{ShaderRef, load_shader_library};

/// Split integer scan-cell and fractional coordinates preserve small detail
/// without passing astronomical positions as floats. Local positions are metres.
/// The fine octave (`phase`, `cell`) holds the metre-scale scans; the
/// mesoscale octave (`meso_phase`, `meso_cell`) holds the same layers at tens
/// of metres. A zero inverse size (`w`) disables that octave for the layer.
#[derive(Clone, Debug, Default, ShaderType)]
pub struct DetailCoordinates {
    pub phase: [Vec4; 8],
    pub cell: [IVec4; 8],
    pub meso_phase: [Vec4; 8],
    pub meso_cell: [IVec4; 8],
}

impl DetailCoordinates {
    /// Origin in a stable material frame, and eight strictly positive scan sizes
    /// in metres. Integer coordinates must fit i32 (about 2 billion patches).
    /// The mesoscale octave is disabled.
    pub fn new(origin: DVec3, sizes: [f64; 8]) -> Option<Self> {
        let (phase, cell) = Self::octave(origin, sizes)?;
        Some(Self {
            phase,
            cell,
            meso_phase: [Vec4::ZERO; 8],
            meso_cell: [IVec4::ZERO; 8],
        })
    }

    /// As [`Self::new`], with a second octave of eight strictly positive
    /// mesoscale sizes in metres for the same layers.
    pub fn with_mesoscale(origin: DVec3, sizes: [f64; 8], meso_sizes: [f64; 8]) -> Option<Self> {
        let (phase, cell) = Self::octave(origin, sizes)?;
        let (meso_phase, meso_cell) = Self::octave(origin, meso_sizes)?;
        Some(Self {
            phase,
            cell,
            meso_phase,
            meso_cell,
        })
    }

    fn octave(origin: DVec3, sizes: [f64; 8]) -> Option<([Vec4; 8], [IVec4; 8])> {
        let mut phase = [Vec4::ZERO; 8];
        let mut cell = [IVec4::ZERO; 8];
        for (i, size) in sizes.into_iter().enumerate() {
            if !origin.is_finite() || !size.is_finite() || size <= 0.0 {
                return None;
            }
            let inverse = (1.0 / size) as f32;
            if !inverse.is_finite() || inverse <= 0.0 {
                return None;
            }
            let p = origin / size;
            if p.abs().max_element() >= f64::from(i32::MAX) {
                return None;
            }
            let floor = p.floor();
            cell[i] = floor.as_ivec3().extend(0);
            phase[i] = (p - floor).as_vec3().extend(inverse);
        }
        Some((phase, cell))
    }

    /// CPU reconstruction for numerical coordinate acceptance/readback.
    #[must_use]
    pub fn reconstruct(&self, layer: usize, local: Vec3) -> DVec3 {
        Self::reconstruct_octave(&self.phase, &self.cell, layer, local)
    }

    /// CPU reconstruction of the mesoscale octave.
    #[must_use]
    pub fn reconstruct_mesoscale(&self, layer: usize, local: Vec3) -> DVec3 {
        Self::reconstruct_octave(&self.meso_phase, &self.meso_cell, layer, local)
    }

    fn reconstruct_octave(
        phase: &[Vec4; 8],
        cell: &[IVec4; 8],
        layer: usize,
        local: Vec3,
    ) -> DVec3 {
        cell[layer].truncate().as_dvec3()
            + (local * phase[layer].w + phase[layer].truncate()).as_dvec3()
    }
}

/// Eight linear coverage channels and two octaves of shared scan arrays.
/// Colour is sRGB encoded around linear mean 0.5; detail xy stores tangent
/// normals and z stores roughness around mean 0.5. The base material supplies
/// macroscopic means. The mesoscale arrays hold the same eight layers at the
/// sizes `coordinates.meso_phase` records; a disabled octave still needs a
/// bound array (one neutral texel per layer is enough).
/// Mesh transforms must be rigid and local positions in metres.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct SurfaceDetail {
    #[uniform(100)]
    pub coordinates: DetailCoordinates,
    #[texture(101, dimension = "2d_array")]
    #[sampler(102)]
    pub colour: Handle<Image>,
    #[texture(103, dimension = "2d_array")]
    pub detail: Handle<Image>,
    #[texture(104)]
    #[sampler(105)]
    pub coverage0: Handle<Image>,
    #[texture(106)]
    pub coverage1: Handle<Image>,
    #[uniform(107)]
    pub gaussian_parameters: Vec4,
    #[texture(108)]
    #[sampler(109)]
    pub gaussian_mask: Option<Handle<Image>>,
    /// x enables the particulate disk law instead of dielectric optics.
    #[uniform(110)]
    pub lommel: Vec4,
    #[texture(111, dimension = "2d_array")]
    pub meso_colour: Handle<Image>,
    #[texture(112, dimension = "2d_array")]
    pub meso_detail: Handle<Image>,
}

impl MaterialExtension for SurfaceDetail {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_solarik/surface_detail.wgsl".into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        Self::fragment_shader()
    }
}

pub type DetailedMaterial = ExtendedMaterial<StandardMaterial, SurfaceDetail>;

/// Map a custom raster material's ray proxy to the same scan parameters.
/// The caller owns handle lifetime and removes entries when evicting materials.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct DetailedRayMaterials(pub HashMap<AssetId<StandardMaterial>, SurfaceDetail>);

#[derive(Clone, Default, ShaderType)]
pub(crate) struct GpuSurfaceDetail {
    pub coordinates: DetailCoordinates,
    /// Coverage IDs in the ordinary texture array, colour/detail in scan array.
    /// x == `u32::MAX` disables detail for an ordinary `StandardMaterial`.
    pub textures: UVec4,
    /// Mesoscale colour (x) and detail (y) in the scan array.
    pub meso: UVec4,
}

/// Registers filtered scanned detail for Bevy's forward and deferred paths.
pub struct SurfaceDetailPlugin;
impl Plugin for SurfaceDetailPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "surface_detail.wgsl");
        load_shader_library!(app, "detail_sampling.wgsl");
        app.init_resource::<DetailedRayMaterials>();
        app.add_plugins(ExtractResourcePlugin::<DetailedRayMaterials>::default());
        app.add_plugins(MaterialPlugin::<DetailedMaterial>::default());
    }
}
