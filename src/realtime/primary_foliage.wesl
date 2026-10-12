import bevy_solarik::scene::scene_hit::scene_hit_is_miss;
import bevy_solarik::scene::scene_hit::scene_hit_is_coarse;
import bevy_solarik::scene::scene_hit::SceneHit;
import bevy_solarik::scene::scene_hit::scene_coarse;
import bevy_solarik::filtered_canopy_scene::forest_set_seed;
import bevy_solarik::filtered_canopy_scene::forest_primary_seed;
import bevy_solarik::filtered_canopy_scene::forest_background_guides;
import bevy_solarik::filtered_canopy_scene::forest_conditioning_enabled;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::primary_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::advance_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::resolve_ray_hit_filtered;
import bevy_solarik::realtime::foliage_math::foliage_depth_matches;
import bevy_solarik::realtime::surface_path::shade_surface_path;
import bevy_solarik::scene::raytracing_scene_bindings::trace_primary_ray;
import bevy_solarik::scene::raytracing_scene_bindings::trace_primary_conditioned;
import bevy_solarik::scene::raytracing_scene_bindings::resolve_ray_hit_full;
import bevy_solarik::scene::raytracing_scene_bindings::ResolvedRayHitFull;
import bevy_solarik::scene::raytracing_scene_bindings::ray_max_distance;
import bevy_solarik::realtime::gbuffer_utils::gpixel_resolve;
import bevy_solarik::realtime::gbuffer_utils::reconstruct_world_position;
import bevy_solarik::realtime::realtime_bindings::view_output;
import bevy_solarik::realtime::realtime_bindings::gbuffer;
import bevy_solarik::realtime::realtime_bindings::depth_buffer;
import bevy_solarik::realtime::realtime_bindings::view;
import bevy_solarik::realtime::realtime_bindings::constants;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::realtime::primary_light_trace::light_trace_reset;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::realtime::primary_light_trace::light_trace_set;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::realtime::primary_light_trace::light_trace_set_u;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::realtime::primary_light_trace::light_trace_commit;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::filtered_canopy_scene::forest_scene;
@if((DLSS_RR_GUIDE_BUFFERS)) import bevy_solarik::realtime::resolve_dlss_rr_textures::resolve_background_guides;
@if((PRIMARY_CANOPY_QUERY_ONLY)) import bevy_solarik::filtered_canopy_scene::forest_scene;

enable wgpu_ray_query;




@if((PRIMARY_CANOPY_QUERY_ONLY))
@group(3) @binding(0) var<storage,read_write> primary_query_capture:array<vec4u>;

@compute @workgroup_size(8, 8, 1)
fn primary_foliage(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2u(view.main_pass_viewport.zw)) { return; }
@if(PRIMARY_CANOPY_QUERY_ONLY) {
    if constants.reset!=0u && all(id.xy==vec2u(0u)) {
        primary_query_capture[0]=forest_scene[0];primary_query_capture[1]=forest_scene[1];
    }
}
    let depth = textureLoad(depth_buffer, id.xy, 0);
    if depth == 0.0 && !forest_background_guides() { return; }
    let near = reconstruct_world_position(id.xy, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
    let direction = normalize(reconstruct_world_position(id.xy, 0.0000001, view.main_pass_viewport.zw, view.world_from_clip) - near);
    let distance_to_surface = primary_surface_distance(id.xy, depth, near);
    forest_set_seed(forest_primary_seed(id.xy,u32(view.main_pass_viewport.z)));
@if(PRIMARY_CANOPY_QUERY_ONLY) {
    diagnose_primary_query(near,direction,distance_to_surface,id.xy);
}
@if(!(PRIMARY_CANOPY_QUERY_ONLY)) {
    if forest_conditioning_enabled() {
        shade_conditioned_primary(near,direction,distance_to_surface,depth,id.xy);
        return;
    }
    let ray = trace_primary_ray(near, direction, 0.0, distance_to_surface + max(0.002, distance_to_surface * 0.00001), RAY_FLAG_NONE);
    if scene_hit_is_miss(ray) { return; }
    let hit=primary_resolve(ray,direction,near);
    if !primary_surface_matches(hit, scene_hit_is_coarse(ray), id.xy, depth, distance_to_surface) { return; }
    shade_primary_foliage(hit, direction, id.xy);
}
}

// Explicit graphics_debug diagnostic. This is not a radiance estimator.
// A checksum consumes the resolved payload too, keeping event work live.
@if((PRIMARY_CANOPY_QUERY_ONLY))
fn diagnose_primary_query(near:vec3f,direction:vec3f,distance_to_surface:f32,pixel:vec2u) {
    let maximum=distance_to_surface+max(0.002,distance_to_surface*0.00001);
    let query=trace_primary_conditioned(near,direction,0.0,maximum,RAY_FLAG_NONE);
    let event=query.canopy.event.event;
    let sum=dot(event.position_distance,vec4f(0.001))+dot(event.normal_hit,vec4f(0.13))
        +dot(event.color_coverage,vec4f(0.17))+dot(event.surface,vec4f(0.19))
        +dot(vec4f(event.source),vec4f(0.0000001))
        +dot(query.canopy.event.previous_position,vec3f(0.002))
        +query.canopy.weights.y+query.canopy.proposal_ratio+f32(query.canopy.fallback)
        +query.background.t*0.001+f32(query.background.kind)
        +dot(query.background.triangle.barycentrics,vec2f(0.23))
        +f32(query.background.triangle.slot+query.background.triangle.primitive)*0.0000001;
    let encoded=vec4f(query.canopy.coverage,
        query.canopy.weights.x/(1.0+query.canopy.weights.x),fract(abs(sum)),1.0);
    textureStore(view_output,pixel,encoded);
    let size=vec2u(view.main_pass_viewport.zw);
    let corner=vec2u(size.x/2u-256u,size.y/2u-128u);
    if constants.reset==0u || any(pixel<corner) || any(pixel>=corner+vec2u(512u,256u)) {return;}
    let xy=pixel-corner;
    let index=2u+((xy.x/256u)*256u*256u+xy.y*256u+xy.x%256u)*16u;
    primary_query_capture[index]=bitcast<vec4u>(vec4f(near,maximum));
    primary_query_capture[index+1u]=bitcast<vec4u>(vec4f(direction,query.maximum));
    primary_query_capture[index+2u]=bitcast<vec4u>(vec4f(query.canopy.coverage,query.canopy.weights,query.canopy.proposal_ratio));
    primary_query_capture[index+3u]=bitcast<vec4u>(event.position_distance);
    primary_query_capture[index+4u]=bitcast<vec4u>(event.normal_hit);
    primary_query_capture[index+5u]=event.source;
    primary_query_capture[index+6u]=bitcast<vec4u>(encoded);
    primary_query_capture[index+7u]=vec4u(bitcast<u32>(query.background.t),query.background.kind,query.canopy.fallback,1u);
}

fn primary_resolve(ray:SceneHit,direction:vec3f,near:vec3f)->ResolvedRayHitFull {
@if(RAY_MATERIAL_FOOTPRINTS) {
    let cone = primary_ray_cone(view.clip_from_view, view.main_pass_viewport.zw, ray.t + distance(near, view.world_position));
    return resolve_ray_hit_filtered(ray, direction, cone);
}
@if(!(RAY_MATERIAL_FOOTPRINTS)) {
    return resolve_ray_hit_full(ray);
}
}

fn primary_surface_distance(pixel: vec2u, depth: f32, near: vec3f) -> f32 {
    if depth == 0.0 { return ray_max_distance(); }
    return distance(near, reconstruct_world_position(pixel, depth, view.main_pass_viewport.zw, view.world_from_clip));
}

fn primary_surface_matches(hit: ResolvedRayHitFull, coarse: bool, pixel: vec2u, depth: f32, distance_to_surface: f32) -> bool {
    // The guide control intentionally leaves opaque depth behind the sampled canopy.
    // The primary query still clips that event against the actual opaque surface.
    if coarse && forest_background_guides() { return true; }
    if depth == 0.0 || (!coarse && hit.material.diffuse_transmission <= 0.0) { return false; }
    let raster = gpixel_resolve(textureLoad(gbuffer, pixel, 0), depth, pixel, view.main_pass_viewport.zw, view.world_from_clip);
    return foliage_depth_matches(raster.world_position, hit.world_position, raster.world_normal, hit.world_normal, distance_to_surface);
}

fn shade_primary_foliage(hit: ResolvedRayHitFull, direction: vec3f, pixel: vec2u) {
    textureStore(view_output,pixel,vec4f(primary_foliage_radiance(hit,direction,pixel,4u),1.0));
@if(DLSS_RR_GUIDE_BUFFERS) {
    resolve_background_guides(pixel);
}
}

fn primary_foliage_radiance(hit:ResolvedRayHitFull,direction:vec3f,pixel:vec2u,samples:u32)->vec3f {
    var rng = pixel.x + pixel.y * u32(view.main_pass_viewport.z) + constants.frame_index;
    var radiance = vec3(0.0);
    for (var sample = 0u; sample < samples; sample += 1u) {
        radiance += shade_surface_path(hit, -direction, &rng);
    }
    return radiance*(view.exposure/f32(samples));
}

fn shade_conditioned_primary(near:vec3f,direction:vec3f,distance_to_surface:f32,depth:f32,pixel:vec2u) {
    let maximum=distance_to_surface+max(0.002,distance_to_surface*0.00001);
    let query=trace_primary_conditioned(near,direction,0.0,maximum,RAY_FLAG_NONE);
    if !scene_hit_is_miss(query.background) {
        let hit=primary_resolve(query.background,direction,near);
        if primary_surface_matches(hit,false,pixel,depth,distance_to_surface) {
            shade_primary_foliage(hit,direction,pixel);
        }
    }
    if query.canopy.coverage<=0.0 { return; }
    let background=textureLoad(view_output,pixel).rgb;
@if(PRIMARY_CANOPY_DIRECT_ONLY) {
    light_trace_reset();
}
    var foreground=vec3f(0.0);
    if query.canopy.event.event.normal_hit.w>0.0 {
        let hit=primary_resolve(scene_coarse(query.canopy.event),direction,near);
        foreground=primary_foliage_radiance(hit,direction,pixel,1u);
    }
    textureStore(view_output,pixel,vec4f(foreground*query.canopy.weights.x+background*query.canopy.weights.y,1.0));
@if(PRIMARY_CANOPY_DIRECT_ONLY) {
    light_trace_set(1u,vec4f(background,view.exposure));
    light_trace_set(2u,vec4f(foreground,query.canopy.weights.x));
    let result=foreground*query.canopy.weights.x+background*query.canopy.weights.y;
    light_trace_set(3u,vec4f(result,query.canopy.weights.y));
    light_trace_set(4u,vec4f(near,maximum));
    light_trace_set(5u,vec4f(direction,query.maximum));
    light_trace_set(6u,query.canopy.event.event.position_distance);
    light_trace_set(7u,query.canopy.event.event.normal_hit);
    light_trace_set(17u,query.canopy.event.event.surface);
    light_trace_set_u(18u,query.canopy.event.event.source);
    light_trace_commit(pixel,forest_scene[1].x,constants.frame_index,result);
}
@if(DLSS_RR_GUIDE_BUFFERS) {
    resolve_background_guides(pixel);
}
}

