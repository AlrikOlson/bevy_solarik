enable wgpu_ray_query;

#import bevy_solarik::foliage_math::{foliage_depth_matches, foliage_solid_angle_pdf}
#import bevy_solarik::brdf::{evaluate_brdf, evaluate_and_sample_brdf, evaluate_brdf_pdf}
#import bevy_solarik::sampling::{generate_random_light_sample, calculate_resolved_light_contribution, trace_light_visibility, random_emissive_light_pdf, power_heuristic}
#import bevy_solarik::scene_bindings::{trace_ray, resolve_ray_hit_full, sample_sky, light_sources, ResolvedRayHitFull, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solarik::thin_glass::offset_thin_glass_ray
#import bevy_solarik::gbuffer_utils::{gpixel_resolve, reconstruct_world_position}
#import bevy_solarik::realtime_bindings::{view_output, gbuffer, depth_buffer, view, constants}
#ifdef DLSS_RR_GUIDE_BUFFERS
#import bevy_solarik::resolve_dlss_rr_textures::resolve_background_guides
#endif

// A separate estimator avoids feeding two-sided samples to one-sided caches.
// Four paths, four scattering events each; the final vertex uses NEE alone.
fn shade_foliage_path(initial: ResolvedRayHitFull, initial_wo: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var hit = initial;
    var wo = initial_wo;
    var throughput = vec3(1.0);
    var radiance = hit.material.emissive;
    for (var bounce = 0u; bounce < 4u; bounce += 1u) {
        let last = bounce == 3u;
        let mirror = hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && hit.material.metallic > 0.9999;
        if !mirror && arrayLength(&light_sources) > 0u {
            let sample = generate_random_light_sample(rng).resolved_light_sample;
            let light = calculate_resolved_light_contribution(sample, hit.world_position, hit.world_normal);
            var weight = 1.0;
            if light.brdf_rays_can_hit && !last {
                let delta = sample.world_position.xyz - hit.world_position;
                let p_light = foliage_solid_angle_pdf(1.0 / light.inverse_pdf, dot(delta, delta), dot(-light.wi, sample.world_normal));
                weight = power_heuristic(p_light, evaluate_brdf_pdf(wo, light.wi, hit.world_normal, hit.material));
            }
            let origin = offset_thin_glass_ray(hit.world_position, hit.geometric_world_normal, light.wi, RAY_T_MIN);
            let visible = trace_light_visibility(origin, sample.world_position);
            radiance += throughput * light.radiance * light.inverse_pdf * weight * visible
                * evaluate_brdf(wo, light.wi, hit.world_normal, hit.material);
        }
        if last { break; }
        let next = evaluate_and_sample_brdf(wo, hit.world_normal, hit.material, rng);
        if next.pdf <= 0.0 { break; }
        throughput *= next.throughput;
        if all(throughput <= vec3(0.0)) { break; }
        let origin = offset_thin_glass_ray(hit.world_position, hit.geometric_world_normal, next.wi, RAY_T_MIN);
        let ray = trace_ray(origin, next.wi, RAY_T_MIN, RAY_T_MAX, RAY_FLAG_NONE);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE {
            radiance += throughput * sample_sky(next.wi);
            break;
        }
        let previous_position = hit.world_position;
        hit = resolve_ray_hit_full(ray);
        wo = -next.wi;
        var emission_weight = 1.0;
        if !mirror {
            let delta = hit.world_position - previous_position;
            let p_light = foliage_solid_angle_pdf(random_emissive_light_pdf(hit), dot(delta, delta), dot(wo, hit.geometric_world_normal));
            emission_weight = power_heuristic(next.pdf, p_light);
        }
        radiance += throughput * hit.material.emissive * emission_weight;
    }
    return radiance;
}

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
        radiance += shade_foliage_path(hit, -direction, &rng);
    }
    textureStore(view_output, id.xy, vec4(radiance * (view.exposure / 4.0), 1.0));
#ifdef DLSS_RR_GUIDE_BUFFERS
    resolve_background_guides(id.xy);
#endif
}

