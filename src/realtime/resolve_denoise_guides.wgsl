// Optional guide textures for host denoisers such as MetalFX.
// that runs on the lit frame after `solari_lighting` — MetalFX's
// temporal denoised scaler on the M5 Max. The same resolve as
// `resolve_dlss_rr_textures.wgsl` (diffuse albedo, the split-sum
// specular albedo), with the normal and the roughness written APART
// (the scaler takes two textures) and the specular hit distance
// zeroed here so the sky keeps a defined value — `specular_gi.wgsl`
// overwrites it for every traced pixel in the same compute pass.
enable wgpu_ray_query;

#import bevy_pbr::pbr_functions::{calculate_diffuse_color, calculate_F0}
#import bevy_render::view::View
#import bevy_solarik::gbuffer_utils::gpixel_resolve
#import bevy_solarik::realtime_bindings::{gbuffer, depth_buffer, view, guide_diffuse_albedo, guide_specular_albedo, guide_normal, guide_roughness, guide_specular_hit_distance}

@compute @workgroup_size(8, 8, 1)
fn resolve_denoise_guides(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    if any(pixel_id >= vec2u(view.main_pass_viewport.zw)) { return; }

    textureStore(guide_specular_hit_distance, pixel_id, vec4(0.0));

    let depth = textureLoad(depth_buffer, pixel_id, 0);
    if depth == 0.0 {
        // The sky: no surface. A black diffuse albedo and a mid specular
        // albedo keep the demodulation finite; the normal faces the
        // camera's up so the scaler sees a valid unit vector.
        textureStore(guide_diffuse_albedo, pixel_id, vec4(0.0));
        textureStore(guide_specular_albedo, pixel_id, vec4(0.5));
        textureStore(guide_normal, pixel_id, vec4(0.0, 1.0, 0.0, 0.0));
        textureStore(guide_roughness, pixel_id, vec4(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let surface = gpixel_resolve(textureLoad(gbuffer, pixel_id, 0), depth, pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
    let F0 = calculate_F0(surface.material.base_color, surface.material.metallic, vec3(surface.material.reflectance));
    let wo = normalize(view.world_position - surface.world_position);

    textureStore(guide_diffuse_albedo, pixel_id, vec4(calculate_diffuse_color(surface.material.base_color, surface.material.metallic, 0.0, 0.0), 0.0));
    textureStore(guide_specular_albedo, pixel_id, vec4(env_brdf_approx2(F0, surface.material.roughness, surface.world_normal, wo), 0.0));
    textureStore(guide_normal, pixel_id, vec4(surface.world_normal, 0.0));
    textureStore(guide_roughness, pixel_id, vec4(surface.material.perceptual_roughness, 0.0, 0.0, 0.0));
}

// The split-sum environment BRDF fit (the same function as
// resolve_dlss_rr_textures.wgsl's — kept local so this module does not
// pull that module's DLSS-only bindings through an import).
fn env_brdf_approx2(specular_color: vec3<f32>, alpha: f32, N: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    let NoV = abs(dot(N, V));

    var X: vec4<f32>;
    X.x = 1.0;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    var Y: vec4<f32>;
    Y.x = 1.0;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    let M1 = mat2x2<f32>(0.99044, 1.29678, -1.28514, -0.755907);
    let M2 = mat3x3<f32>(1.0, 20.3225, 121.563, 2.92338, -27.0302, 626.13, 59.4188, 222.592, 316.627);
    let M3 = mat2x2<f32>(0.0365463, 9.0632, 3.32707, -9.04756);
    let M4 = mat3x3<f32>(1.0, 9.04401, 5.56589, 3.59685, -16.3174, 19.7886, -1.36772, 9.22949, -20.2123);

    var bias = dot(M1 * X.xy, Y.xy) / dot(M2 * X.xyw, Y.xyw);
    let scale = dot(M3 * X.xy, Y.xy) / dot(M4 * X.xzw, Y.xyw);

    bias *= saturate(specular_color.g * 50.0);

    return fma(specular_color, vec3(max(0.0, scale)), vec3(max(0.0, bias)));
}
