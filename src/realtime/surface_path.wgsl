enable wgpu_ray_query;
#define_import_path bevy_solarik::surface_path

#import bevy_solarik::brdf::{evaluate_brdf, evaluate_and_sample_brdf, evaluate_brdf_pdf}
#import bevy_solarik::sampling::{generate_random_light_sample, calculate_resolved_light_contribution, trace_light_visibility, random_emissive_light_solid_angle_pdf, power_heuristic}
#import bevy_solarik::scene_bindings::{trace_ray, resolve_ray_hit_full, sample_sky, light_sources, ResolvedRayHitFull, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solarik::thin_glass::offset_thin_glass_ray

// A separate estimator avoids feeding two-sided samples to one-sided caches.
// Four scattering events per path; the final vertex uses NEE alone.
fn shade_surface_path(initial: ResolvedRayHitFull, initial_wo: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
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
                let p_light = light.solid_angle_pdf;
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
            let p_light = random_emissive_light_solid_angle_pdf(hit, previous_position);
            emission_weight = power_heuristic(next.pdf, p_light);
        }
        radiance += throughput * hit.material.emissive * emission_weight;
    }
    return radiance;
}
