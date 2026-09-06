enable wgpu_ray_query;

#import bevy_solarik::foliage_math::foliage_depth_matches
#import bevy_solarik::surface_path::shade_surface_path
#import bevy_solarik::scene_bindings::{trace_ray, resolve_ray_hit_full}
#import bevy_solarik::gbuffer_utils::{gpixel_resolve, reconstruct_world_position}
#import bevy_solarik::realtime_bindings::{view_output, gbuffer, depth_buffer, view, constants}
#ifdef DLSS_RR_GUIDE_BUFFERS
#import bevy_solarik::resolve_dlss_rr_textures::resolve_background_guides
#endif

@compute @workgroup_size(8, 8, 1)
fn primary_foliage(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }
    let depth = textureLoad(depth_buffer, id.xy, 0);
    if depth == 0.0 { return; }
    let raster = gpixel_resolve(textureLoad(gbuffer, id.xy, 0), depth, id.xy, view.main_pass_viewport.zw, view.world_from_clip);
    let near = reconstruct_world_position(id.xy, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
    let direction = normalize(raster.world_position - near);
    let distance_to_surface = distance(near, raster.world_position);
    let ray = trace_ray(near, direction, 0.0, distance_to_surface + max(0.002, distance_to_surface * 0.00001), RAY_FLAG_NONE);
    if ray.kind == RAY_QUERY_INTERSECTION_NONE { return; }
    let hit = resolve_ray_hit_full(ray);
    if hit.material.diffuse_transmission <= 0.0 { return; }
    if !foliage_depth_matches(raster.world_position, hit.world_position, raster.world_normal, hit.world_normal, distance_to_surface) { return; }
    var rng = id.x + id.y * u32(view.main_pass_viewport.z) + constants.frame_index;
    var radiance = vec3(0.0);
    for (var sample = 0u; sample < 4u; sample += 1u) {
        radiance += shade_surface_path(hit, -direction, &rng);
    }
    textureStore(view_output, id.xy, vec4(radiance * (view.exposure / 4.0), 1.0));
#ifdef DLSS_RR_GUIDE_BUFFERS
    resolve_background_guides(id.xy);
#endif
}

