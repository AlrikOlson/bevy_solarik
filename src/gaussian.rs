//! Isotropic Gaussian slope microfacets for measured dielectric surfaces.
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetId, Handle, embedded_asset};
use bevy_ecs::resource::Resource;
use bevy_image::Image;
use bevy_math::Vec4;
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_reflect::TypePath;
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::AsBindGroup,
};
use bevy_shader::{ShaderRef, load_shader_library};

/// Mask alpha selects Gaussian optics; RGB may hold an ordinary normal map.
/// This extension targets Solarik deferred lighting. Forward Bevy lighting
/// retains its GGX fallback. Clearcoat is unavailable on these materials.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct GaussianDielectric {
    /// Bevy reflectance, perceptual roughness, enabled, reserved.
    #[uniform(100)]
    pub parameters: Vec4,
    #[texture(101)]
    #[sampler(102)]
    pub mask: Handle<Image>,
}

impl GaussianDielectric {
    /// RMS slope is sqrt(E[sx²+sy²]), not the standard deviation of one axis.
    /// The supported range keeps the lobe above the renderer's mirror cutoff.
    pub fn new(ior: f32, rms_slope: f32, mask: Handle<Image>) -> Option<Self> {
        if !ior.is_finite()
            || !(1.0..=7.0 / 3.0).contains(&ior)
            || !rms_slope.is_finite()
            || !(0.025..=1.0).contains(&rms_slope)
        {
            return None;
        }
        Some(Self {
            parameters: Vec4::new(
                (ior - 1.0) / (0.4 * (ior + 1.0)),
                rms_slope.sqrt(),
                1.0,
                0.0,
            ),
            mask,
        })
    }
}
impl MaterialExtension for GaussianDielectric {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_solarik/gaussian_material.wgsl".into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        Self::fragment_shader()
    }
}
pub type GaussianMaterial = ExtendedMaterial<StandardMaterial, GaussianDielectric>;

/// Parameters for the `StandardMaterial` ray proxy; caller owns entry lifetime.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct GaussianRayMaterials(pub HashMap<AssetId<StandardMaterial>, GaussianDielectric>);

pub struct GaussianDielectricPlugin;
impl Plugin for GaussianDielectricPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "gaussian_material.wgsl");
        load_shader_library!(app, "gaussian_math.wgsl");
        app.init_resource::<GaussianRayMaterials>();
        app.add_plugins(ExtractResourcePlugin::<GaussianRayMaterials>::default());
        app.add_plugins(MaterialPlugin::<GaussianMaterial>::default());
    }
}

/// Exact unpolarized Fresnel for air incident on a nonabsorbing dielectric.
#[must_use]
pub fn dielectric_fresnel(cosine: f64, ior: f64) -> f64 {
    if ior == 1.0 {
        return 0.0;
    }
    let c = cosine.clamp(0.0, 1.0);
    let t = (1.0 - (1.0 - c * c) / (ior * ior)).max(0.0).sqrt();
    let rs = (c - ior * t) / (c + ior * t);
    let rp = (ior * c - t) / (ior * c + t);
    0.5 * (rs * rs + rp * rp)
}
