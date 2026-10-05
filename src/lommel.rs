//! Reciprocal Lommel-Seeliger disk scattering for dark particulate surfaces.
//! Base colour means normal full-phase radiance factor, bounded to 0..0.25.
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetId, embedded_asset};
use bevy_ecs::resource::Resource;
use bevy_math::Vec4;
use bevy_pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy_platform::collections::HashSet;
use bevy_reflect::TypePath;
use bevy_render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::AsBindGroup,
};
use bevy_shader::{ShaderRef, load_shader_library};

/// A diffuse particulate surface, with no separate smooth-interface lobe.
/// This is a disk law, not a fitted opposition or wavelength phase function.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct LommelSeeliger {
    #[uniform(100)]
    pub enabled: Vec4,
}
impl Default for LommelSeeliger {
    fn default() -> Self {
        Self { enabled: Vec4::X }
    }
}
impl MaterialExtension for LommelSeeliger {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_solarik/lommel_material.wgsl".into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        Self::fragment_shader()
    }
}
pub type LommelMaterial = ExtendedMaterial<StandardMaterial, LommelSeeliger>;
/// Ray proxies using the same disk law. Caller owns entry lifetime.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct LommelRayMaterials(pub HashSet<AssetId<StandardMaterial>>);
pub struct LommelSeeligerPlugin;
impl Plugin for LommelSeeligerPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "lommel_material.wgsl");
        load_shader_library!(app, "lommel_math.wgsl");
        app.init_resource::<LommelRayMaterials>();
        app.add_plugins(ExtractResourcePlugin::<LommelRayMaterials>::default());
        app.add_plugins(MaterialPlugin::<LommelMaterial>::default());
    }
}
/// BRDF (sr^-1), without incoming cosine. Clamping rho to 1/4 guarantees
/// directional-hemispherical reflectance <= 1 for every incidence.
#[must_use]
pub fn lommel_seeliger(rho: f64, incoming: f64, outgoing: f64) -> f64 {
    if incoming <= 0.0 || outgoing <= 0.0 {
        return 0.0;
    }
    2.0 * rho.clamp(0.0, 0.25) / (core::f64::consts::PI * (incoming + outgoing))
}
