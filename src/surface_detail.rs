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
#[derive(Clone, Debug, Default, ShaderType)]
pub struct DetailCoordinates {
    pub phase: [Vec4; 8],
    pub cell: [IVec4; 8],
}

impl DetailCoordinates {
    /// Origin in a stable material frame, and eight strictly positive scan sizes
    /// in metres. Integer coordinates must fit i32 (about 2 billion patches).
    pub fn new(origin: DVec3, sizes: [f64; 8]) -> Option<Self> {
        let mut result = Self {
            phase: [Vec4::ZERO; 8],
            cell: [IVec4::ZERO; 8],
        };
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
            result.cell[i] = floor.as_ivec3().extend(0);
            result.phase[i] = (p - floor).as_vec3().extend(inverse);
        }
        Some(result)
    }

    /// CPU reconstruction for numerical coordinate acceptance/readback.
    #[must_use]
    pub fn reconstruct(&self, layer: usize, local: Vec3) -> DVec3 {
        self.cell[layer].truncate().as_dvec3()
            + (local * self.phase[layer].w + self.phase[layer].truncate()).as_dvec3()
    }
}

/// Eight linear coverage channels and two shared scan arrays. Colour is sRGB
/// encoded around linear mean 0.5; detail xy stores tangent normals and z stores
/// roughness around mean 0.5. The base material supplies macroscopic means.
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
