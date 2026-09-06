enable wgpu_ray_query;
#define_import_path bevy_solarik::surface_path

#import bevy_pbr::utils::rand_f
#import bevy_solarik::brdf::{evaluate_brdf, evaluate_and_sample_brdf, evaluate_brdf_pdf}
#import bevy_solarik::sampling::{analytic_light_radiance, sample_random_light_transmitted, random_emissive_light_solid_angle_pdf, power_heuristic}
#import bevy_solarik::scene_bindings::{trace_glass_ray, resolve_ray_hit_full, sample_sky, light_sources, ResolvedRayHitFull, materials, material_ids, resolve_material_alpha, MATERIAL_FLAG_ALPHA_BLEND, MATERIAL_FLAG_DIFFUSE_BLEND, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}
#import bevy_solarik::thin_glass::{thin_glass_weights, sample_thin_glass, offset_thin_glass_ray}

// A separate estimator avoids feeding two-sided samples to one-sided caches.
// Four ordinary scattering events; final vertex uses NEE alone. Thin panes
// have a separate 32-event budget between ordinary vertices.
fn shade_surface_path(initial: ResolvedRayHitFull, initial_wo: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var hit = initial;
    var wo = initial_wo;
    var throughput = vec3(1.0);
    var radiance = hit.material.emissive;
    for (var bounce = 0u; bounce < 4u; bounce += 1u) {
        let last = bounce == 3u;
        let mirror = hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && hit.material.metallic > 0.9999;
        if !mirror && arrayLength(&light_sources) > 0u {
            let shadow = sample_random_light_transmitted(hit.world_position,
                hit.world_normal, hit.geometric_world_normal, rng);
            let light = shadow.light;
            var weight = 1.0;
            if light.brdf_rays_can_hit && !last {
                weight = power_heuristic(light.solid_angle_pdf,
                    evaluate_brdf_pdf(wo, light.wi, hit.world_normal, hit.material) * shadow.continuation_probability);
            }
            radiance += throughput * light.radiance * light.inverse_pdf * weight
                * evaluate_brdf(wo, light.wi, hit.world_normal, hit.material);
        }
        if last { break; }
        let next = evaluate_and_sample_brdf(wo, hit.world_normal, hit.material, rng);
        if next.pdf <= 0.0 { break; }
        throughput *= next.throughput;
        if all(throughput <= vec3(0.0)) { break; }
        var previous_position = hit.world_position;
        var wi = next.wi;
        var path_pdf = next.pdf;
        var delta = mirror;
        var origin = offset_thin_glass_ray(hit.world_position, hit.geometric_world_normal, wi, RAY_T_MIN);
        for (var panes = 0u; panes <= 32u; panes += 1u) {
            let ray = trace_glass_ray(origin, wi, RAY_T_MIN, RAY_T_MAX);
            radiance += throughput * analytic_light_radiance(origin, wi,
                select(ray.t, RAY_T_MAX, ray.kind == RAY_QUERY_INTERSECTION_NONE), delta, previous_position);
            if ray.kind == RAY_QUERY_INTERSECTION_NONE {
                return radiance + throughput * sample_sky(wi);
            }
            let candidate = resolve_ray_hit_full(ray);
            let raw = materials[material_ids[ray.instance_index]];
            let glass = (raw.flags & MATERIAL_FLAG_ALPHA_BLEND) != 0u;
            let coverage = (raw.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u;
            if (glass || coverage) && panes == 32u { return radiance; }
            let alpha = clamp(resolve_material_alpha(raw, candidate.uv), 0.0, 1.0);
            if coverage && rand_f(rng) >= alpha {
                path_pdf *= 1.0 - alpha;
                origin = offset_thin_glass_ray(candidate.world_position, candidate.geometric_world_normal, wi, RAY_T_MIN);
                continue;
            }
            var emission_weight = 1.0;
            if !delta {
                emission_weight = power_heuristic(path_pdf,
                    random_emissive_light_solid_angle_pdf(candidate, previous_position));
            }
            radiance += throughput * candidate.material.emissive * emission_weight * select(1.0, alpha, glass);
            if glass {
                let weights = thin_glass_weights(-wi, candidate.geometric_world_normal,
                    candidate.material.base_color, alpha, candidate.material.reflectance);
                let branch = sample_thin_glass(-wi, candidate.geometric_world_normal,
                    candidate.material.base_color, alpha, candidate.material.reflectance, rand_f(rng));
                throughput *= branch.throughput;
                if all(throughput <= vec3(0.0)) { return radiance; }
                if !branch.reflected { path_pdf *= 1.0 - weights.a; }
                delta = delta || branch.reflected;
                if branch.reflected { previous_position = candidate.world_position; }
                wi = branch.wi;
                origin = offset_thin_glass_ray(candidate.world_position, candidate.geometric_world_normal, wi, RAY_T_MIN);
                continue;
            }
            hit = candidate;
            wo = -wi;
            break;
        }
    }
    return radiance;
}
