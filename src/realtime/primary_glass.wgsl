enable wgpu_ray_query;
#define_import_path bevy_solarik::primary_glass

#import bevy_solarik::thin_glass::{thin_glass_weights, offset_thin_glass_ray}
#import bevy_solarik::gbuffer_utils::{reconstruct_world_position, ResolvedGPixel}
#import bevy_solarik::scene_bindings::{trace_glass_ray, materials, instance_material_id, MATERIAL_FLAG_ALPHA_BLEND, MATERIAL_FLAG_DIFFUSE_BLEND, resolve_material_alpha, resolve_ray_hit_full, RAY_T_MIN, RAY_T_MAX}
#import bevy_solarik::specular_gi::trace_glossy_path
#import bevy_solarik::surface_path::shade_surface_path
#import bevy_solarik::realtime_bindings::{view_output, depth_buffer, view, constants}
#ifdef DLSS_RR_GUIDE_BUFFERS
#import bevy_solarik::resolve_dlss_rr_textures::resolve_background_guides
#endif

@compute @workgroup_size(8, 8, 1)
fn primary_glass(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }
    let origin = reconstruct_world_position(id.xy, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
    let along_ray = reconstruct_world_position(id.xy, 0.5, view.main_pass_viewport.zw, view.world_from_clip);
    let direction = normalize(along_ray - origin);
    let depth = textureLoad(depth_buffer, id.xy, 0);
    var distance = RAY_T_MAX;
    if depth > 0.0 {
        let background_position = reconstruct_world_position(id.xy, depth, view.main_pass_viewport.zw, view.world_from_clip);
        distance = max(0.0, dot(background_position - origin, direction) - RAY_T_MIN);
    }
    var rng = id.x + id.y * u32(view.main_pass_viewport.z) + constants.frame_index;
    let background = textureLoad(view_output, id.xy);
    let radiance = composite_primary_glass(id.xy, origin, direction, distance, background.rgb / view.exposure, &rng);
    textureStore(view_output, id.xy, vec4(radiance.rgb * view.exposure, background.a));
#ifdef DLSS_RR_GUIDE_BUFFERS
    if radiance.a > 0.0 { resolve_background_guides(id.xy); }
#endif
}

// Front-to-back camera panes split both lobes deterministically. Reflected
// paths retain the existing bounded glossy estimator; transmission reuses
// the already shaded opaque/sky pixel. Inputs and output are scene radiance.
fn composite_primary_glass(pixel_id: vec2u, initial_origin: vec3f, direction: vec3f, max_distance: f32, background: vec3f, rng: ptr<function, u32>) -> vec4f {
    var origin = initial_origin;
    var transmission = vec3(1.0);
    var radiance = vec3(0.0);
    var has_glass = 0.0;
    for (var i = 0u; i <= 32u; i += 1u) {
        let remaining = max_distance - dot(origin - initial_origin, direction);
        if remaining <= RAY_T_MIN { return vec4(radiance + transmission * background, has_glass); }
        let ray = trace_glass_ray(origin, direction, RAY_T_MIN, remaining);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE { return vec4(radiance + transmission * background, has_glass); }
        let material = materials[instance_material_id(ray.instance_index)];
        if (material.flags & (MATERIAL_FLAG_ALPHA_BLEND | MATERIAL_FLAG_DIFFUSE_BLEND)) == 0u {
            // Raster depth bounds this query. An earlier opaque hit was omitted
            // by raster (e.g. a building's back-facing wall), so sky/background
            // cannot stand in for its radiance. Preserve nonglass pixels.
            if has_glass == 0.0 { return vec4(background, 0.0); }
            let hidden = resolve_ray_hit_full(ray);
            var opaque_radiance = vec3(0.0);
            for (var sample = 0u; sample < 4u; sample += 1u) {
                opaque_radiance += shade_surface_path(hidden, -direction, rng);
            }
            return vec4(radiance + transmission * (opaque_radiance / 4.0), has_glass);
        }
        if i == 32u { break; }
        has_glass = 1.0;
        let hit = resolve_ray_hit_full(ray);
        let alpha = clamp(resolve_material_alpha(material, hit.uv), 0.0, 1.0);
        if (material.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u {
            var surface_radiance = vec3(0.0);
            if alpha > 0.0 {
                for (var sample = 0u; sample < 4u; sample += 1u) {
                    surface_radiance += shade_surface_path(hit, -direction, rng);
                }
            }
            radiance += transmission * alpha * (surface_radiance / 4.0);
            transmission *= 1.0 - alpha;
            if all(transmission <= vec3(0.0)) { return vec4(radiance, has_glass); }
            origin = offset_thin_glass_ray(hit.world_position, hit.geometric_world_normal, direction, RAY_T_MIN);
            continue;
        }
        let weights = thin_glass_weights(-direction, hit.geometric_world_normal, hit.material.base_color, alpha, hit.material.reflectance);
        radiance += transmission * alpha * hit.material.emissive;
        if weights.a > 0.0 {
            let wi = reflect(direction, hit.geometric_world_normal);
            let normal = faceForward(hit.geometric_world_normal, direction, hit.geometric_world_normal);
            var surface = ResolvedGPixel(hit.world_position, normal, hit.material);
            // Smooth dielectric delta: owns reflected emission and never PSR.
            surface.material.roughness = 0.0;
            surface.material.metallic = 0.0;
            let reflected = trace_glossy_path(pixel_id, surface, length(hit.world_position - initial_origin), wi, 1.0, rng);
            radiance += transmission * weights.a * reflected;
        }
        transmission *= weights.rgb;
        if all(transmission <= vec3(0.0)) { return vec4(radiance, has_glass); }
        origin = offset_thin_glass_ray(hit.world_position, hit.geometric_world_normal, direction, RAY_T_MIN);
    }
    // A bounded chain cannot assume unvisited panes are transparent.
    return vec4(radiance, has_glass);
}
