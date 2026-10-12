import bevy_solarik::filtered_canopy_scene::forest_set_seed;
import bevy_solarik::scene::scene_hit::scene_hit_is_miss;
import bevy_solarik::scene::raytracing_scene_bindings::scene_hit_material;
import bevy_solarik::scene::raytracing_scene_bindings::scene_hit_alpha;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::primary_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::advance_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::resolve_ray_hit_filtered;
import bevy_solarik::scene::thin_glass::thin_glass_weights;
import bevy_solarik::scene::thin_glass::offset_thin_glass_ray;
import bevy_solarik::realtime::gbuffer_utils::reconstruct_world_position;
import bevy_solarik::realtime::gbuffer_utils::ResolvedGPixel;
import bevy_solarik::scene::raytracing_scene_bindings::emitted_radiance;
import bevy_solarik::scene::raytracing_scene_bindings::trace_glass_ray;
import bevy_solarik::scene::raytracing_scene_bindings::materials;
import bevy_solarik::scene::raytracing_scene_bindings::material_ids;
import bevy_solarik::scene::raytracing_scene_bindings::MATERIAL_FLAG_ALPHA_BLEND;
import bevy_solarik::scene::raytracing_scene_bindings::MATERIAL_FLAG_DIFFUSE_BLEND;
import bevy_solarik::scene::raytracing_scene_bindings::resolve_material_alpha;
import bevy_solarik::scene::raytracing_scene_bindings::resolve_ray_hit_full;
import bevy_solarik::scene::raytracing_scene_bindings::RAY_T_MIN;
import bevy_solarik::scene::raytracing_scene_bindings::ray_max_distance;
import bevy_solarik::realtime::specular_gi::trace_glossy_path;
import bevy_solarik::realtime::surface_path::shade_surface_path;
import bevy_solarik::realtime::realtime_bindings::view_output;
import bevy_solarik::realtime::realtime_bindings::depth_buffer;
import bevy_solarik::realtime::realtime_bindings::view;
import bevy_solarik::realtime::realtime_bindings::constants;
@if((DLSS_RR_GUIDE_BUFFERS)) import bevy_solarik::realtime::resolve_dlss_rr_textures::resolve_background_guides;

enable wgpu_ray_query;




@compute @workgroup_size(8, 8, 1)
fn primary_glass(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }
    let origin = reconstruct_world_position(id.xy, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
    let along_ray = reconstruct_world_position(id.xy, 0.5, view.main_pass_viewport.zw, view.world_from_clip);
    let direction = normalize(along_ray - origin);
    let depth = textureLoad(depth_buffer, id.xy, 0);
    var distance = ray_max_distance();
    if depth > 0.0 {
        let background_position = reconstruct_world_position(id.xy, depth, view.main_pass_viewport.zw, view.world_from_clip);
        distance = max(0.0, dot(background_position - origin, direction) - RAY_T_MIN);
    }
    var rng = id.x + id.y * u32(view.main_pass_viewport.z) + constants.frame_index;
    forest_set_seed(vec2u(id.x+id.y*u32(view.main_pass_viewport.z),constants.frame_index^0x71478d15u));
    let background = textureLoad(view_output, id.xy);
    let radiance = composite_primary_glass(id.xy, origin, direction, distance, background.rgb / view.exposure, &rng);
    textureStore(view_output, id.xy, vec4(radiance.rgb * view.exposure, background.a));
@if(DLSS_RR_GUIDE_BUFFERS) {
    if radiance.a > 0.0 { resolve_background_guides(id.xy); }
}
}

// Front-to-back camera panes split both lobes deterministically. Reflected
// paths retain the existing bounded glossy estimator; transmission reuses
// the already shaded opaque/sky pixel. Inputs and output are scene radiance.
fn composite_primary_glass(pixel_id: vec2u, initial_origin: vec3f, direction: vec3f, max_distance: f32, background: vec3f, rng: ptr<function, u32>) -> vec4f {
    var origin = initial_origin;
@if(RAY_MATERIAL_FOOTPRINTS) {
    var cone = primary_ray_cone(view.clip_from_view, view.main_pass_viewport.zw, distance(initial_origin, view.world_position));
}
    var transmission = vec3(1.0);
    var radiance = vec3(0.0);
    var has_glass = 0.0;
    for (var i = 0u; i <= 32u; i += 1u) {
        let remaining = max_distance - dot(origin - initial_origin, direction);
        if remaining <= RAY_T_MIN { return vec4(radiance + transmission * background, has_glass); }
        let ray = trace_glass_ray(origin, direction, RAY_T_MIN, remaining);
        if scene_hit_is_miss(ray) { return vec4(radiance + transmission * background, has_glass); }
        let material = scene_hit_material(ray);
@if(RAY_MATERIAL_FOOTPRINTS) {
        cone = advance_ray_cone(cone, ray.t);
}
        if (material.flags & (MATERIAL_FLAG_ALPHA_BLEND | MATERIAL_FLAG_DIFFUSE_BLEND)) == 0u {
            // Raster depth bounds this query. An earlier opaque hit was omitted
            // by raster (e.g. a building's back-facing wall), so sky/background
            // cannot stand in for its radiance. Preserve nonglass pixels.
            if has_glass == 0.0 { return vec4(background, 0.0); }
@if(RAY_MATERIAL_FOOTPRINTS) {
            let hidden = resolve_ray_hit_filtered(ray, direction, cone);
}
@if(!(RAY_MATERIAL_FOOTPRINTS)) {
            let hidden = resolve_ray_hit_full(ray);
}
            var opaque_radiance = vec3(0.0);
            for (var sample = 0u; sample < 4u; sample += 1u) {
                opaque_radiance += shade_surface_path(hidden, -direction, rng);
            }
            return vec4(radiance + transmission * (opaque_radiance / 4.0), has_glass);
        }
        if i == 32u { break; }
        has_glass = 1.0;
@if(RAY_MATERIAL_FOOTPRINTS) {
        let hit = resolve_ray_hit_filtered(ray, direction, cone);
}
@if(!(RAY_MATERIAL_FOOTPRINTS)) {
        let hit = resolve_ray_hit_full(ray);
}
        let alpha = clamp(scene_hit_alpha(ray, hit.uv), 0.0, 1.0);
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
        radiance += transmission * alpha * emitted_radiance(hit.material, -direction);
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
