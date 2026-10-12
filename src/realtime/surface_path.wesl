import bevy_solarik::scene::scene_hit::scene_hit_is_miss;
import bevy_solarik::scene::raytracing_scene_bindings::scene_hit_material;
import bevy_solarik::scene::raytracing_scene_bindings::scene_hit_alpha;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::primary_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::advance_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::scatter_ray_cone;
@if((RAY_MATERIAL_FOOTPRINTS)) import bevy_solarik::scene::raytracing_scene_bindings::resolve_ray_hit_filtered;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::realtime::primary_light_trace::light_trace_set;
@if((PRIMARY_CANOPY_DIRECT_ONLY)) import bevy_solarik::realtime::primary_light_trace::light_trace_set_u;
import bevy_pbr::render::utils::rand_f;
import bevy_solarik::scene::brdf::evaluate_brdf;
import bevy_solarik::scene::brdf::evaluate_and_sample_brdf;
import bevy_solarik::scene::brdf::evaluate_brdf_pdf;
import bevy_solarik::scene::sampling::analytic_light_radiance;
import bevy_solarik::scene::sampling::sample_random_light_transmitted;
import bevy_solarik::scene::sampling::random_emissive_light_solid_angle_pdf;
import bevy_solarik::scene::sampling::power_heuristic;
import bevy_solarik::scene::raytracing_scene_bindings::emitted_radiance;
import bevy_solarik::scene::raytracing_scene_bindings::trace_glass_ray;
import bevy_solarik::scene::raytracing_scene_bindings::resolve_ray_hit_full;
import bevy_solarik::scene::raytracing_scene_bindings::sample_sky;
import bevy_solarik::scene::raytracing_scene_bindings::light_sources;
import bevy_solarik::scene::raytracing_scene_bindings::ResolvedRayHitFull;
import bevy_solarik::scene::raytracing_scene_bindings::materials;
import bevy_solarik::scene::raytracing_scene_bindings::material_ids;
import bevy_solarik::scene::raytracing_scene_bindings::resolve_material_alpha;
import bevy_solarik::scene::raytracing_scene_bindings::MATERIAL_FLAG_ALPHA_BLEND;
import bevy_solarik::scene::raytracing_scene_bindings::MATERIAL_FLAG_DIFFUSE_BLEND;
import bevy_solarik::scene::raytracing_scene_bindings::RAY_T_MIN;
import bevy_solarik::scene::raytracing_scene_bindings::ray_max_distance;
import bevy_solarik::scene::raytracing_scene_bindings::MIRROR_ROUGHNESS_THRESHOLD;
import bevy_solarik::scene::thin_glass::thin_glass_weights;
import bevy_solarik::scene::thin_glass::sample_thin_glass;
import bevy_solarik::scene::thin_glass::offset_thin_glass_ray;

enable wgpu_ray_query;




// A separate estimator avoids feeding two-sided samples to one-sided caches.
// Four ordinary scattering events; final vertex uses NEE alone. Thin panes
// have a separate 32-event budget between ordinary vertices.
fn shade_surface_path(initial: ResolvedRayHitFull, initial_wo: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
@if(PRIMARY_CANOPY_DIRECT_ONLY) {
    light_trace_set(8u,vec4f(initial.material.base_color,initial.material.roughness));
    light_trace_set(9u,vec4f(initial.material.metallic,initial.material.reflectance,initial.material.diffuse_transmission,initial.material.perceptual_roughness));
    light_trace_set_u(16u,vec4u(bitcast<vec3u>(emitted_radiance(initial.material,initial_wo)),*rng));
}
    return emitted_radiance(initial.material, initial_wo) + shade_surface_scattering(initial, initial_wo, rng);
}

// The incoming path owns initial emission and its MIS weight. This entry
// point adds only scattering, so a secondary handoff cannot count it twice.
fn shade_surface_scattering(initial: ResolvedRayHitFull, initial_wo: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    var hit = initial;
@if(RAY_MATERIAL_FOOTPRINTS) {
    var cone = initial.ray_cone;
}
    var wo = initial_wo;
    var throughput = vec3(1.0);
    var radiance = vec3(0.0);
@if(PRIMARY_CANOPY_DIRECT_ONLY) {
    // Incomplete lighting ablation, enabled only on the debug primary consumer.
    let vertex_count = 1u;
}
@if(!(PRIMARY_CANOPY_DIRECT_ONLY)) {
    let vertex_count = 4u;
}
    for (var bounce = 0u; bounce < vertex_count; bounce += 1u) {
        let last = bounce + 1u == vertex_count;
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
@if(PRIMARY_CANOPY_DIRECT_ONLY) {
            light_trace_set(15u,vec4f(evaluate_brdf(wo,light.wi,hit.world_normal,hit.material),weight));
}
        }
        if last { break; }
        let next = evaluate_and_sample_brdf(wo, hit.world_normal, hit.material, rng);
        if next.pdf <= 0.0 { break; }
        throughput *= next.throughput;
        if all(throughput <= vec3(0.0)) { break; }
@if(RAY_MATERIAL_FOOTPRINTS) {
        cone = scatter_ray_cone(cone, hit.normal_spread, next.sampled_roughness);
}
        var previous_position = hit.world_position;
        var wi = next.wi;
        var path_pdf = next.pdf;
        var delta = mirror;
        var origin = offset_thin_glass_ray(hit.world_position, hit.geometric_world_normal, wi, RAY_T_MIN);
        for (var panes = 0u; panes <= 32u; panes += 1u) {
            let ray = trace_glass_ray(origin, wi, RAY_T_MIN, ray_max_distance());
            radiance += throughput * analytic_light_radiance(origin, wi,
                select(ray.t, ray_max_distance(), scene_hit_is_miss(ray)), delta, previous_position);
            if scene_hit_is_miss(ray) {
                return radiance + throughput * sample_sky(wi);
            }
@if(RAY_MATERIAL_FOOTPRINTS) {
            cone = advance_ray_cone(cone, ray.t);
            let candidate = resolve_ray_hit_filtered(ray, wi, cone);
}
@if(!(RAY_MATERIAL_FOOTPRINTS)) {
            let candidate = resolve_ray_hit_full(ray);
}
            let raw = scene_hit_material(ray);
            let glass = (raw.flags & MATERIAL_FLAG_ALPHA_BLEND) != 0u;
            let coverage = (raw.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u;
            if (glass || coverage) && panes == 32u { return radiance; }
            let alpha = clamp(scene_hit_alpha(ray, candidate.uv), 0.0, 1.0);
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
            radiance += throughput * emitted_radiance(candidate.material, -wi) * emission_weight * select(1.0, alpha, glass);
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
@if(RAY_MATERIAL_FOOTPRINTS) {
                if branch.reflected { cone = scatter_ray_cone(cone, candidate.normal_spread, 0.0); }
}
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
