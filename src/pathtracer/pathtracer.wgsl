enable wgpu_ray_query;

#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::pbr_functions::calculate_F0
#import bevy_pbr::utils::{rand_f, rand_vec2f}
#import bevy_render::view::View
#import bevy_solarik::brdf::{evaluate_brdf, evaluate_and_sample_brdf, evaluate_brdf_pdf}
#import bevy_solarik::sampling::{sample_random_light_transmitted, random_emissive_light_solid_angle_pdf, ggx_vndf_pdf, power_heuristic}
#import bevy_solarik::scene_bindings::{trace_glass_ray, materials, material_ids, MATERIAL_FLAG_ALPHA_BLEND, MATERIAL_FLAG_DIFFUSE_BLEND, resolve_material_alpha, resolve_ray_hit_full, sample_sky, ResolvedRayHitFull, RAY_T_MIN, RAY_T_MAX, MIRROR_ROUGHNESS_THRESHOLD}

#import bevy_solarik::thin_glass::{thin_glass_weights, sample_thin_glass, offset_thin_glass_ray}

@group(1) @binding(0) var accumulation_texture: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(1) var view_output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var<uniform> view: View;

@compute @workgroup_size(8, 8, 1)
fn pathtrace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2u(view.viewport.zw)) {
        return;
    }

    let old_color = textureLoad(accumulation_texture, global_id.xy);

    // Setup RNG
    let pixel_index = global_id.x + global_id.y * u32(view.viewport.z);
    let frame_index = u32(old_color.a) * 5782582u;
    var rng = pixel_index + frame_index;

    // Shoot the first ray from the camera
    let pixel_center = vec2<f32>(global_id.xy) + 0.5;
    let jitter = rand_vec2f(&rng) - 0.5;
    let pixel_uv = (pixel_center + jitter) / view.viewport.zw;
    let pixel_ndc = (pixel_uv * 2.0) - 1.0;
    let primary_ray_target = view.world_from_clip * vec4(pixel_ndc.x, -pixel_ndc.y, 1.0, 1.0);
    var ray_origin = view.world_position;
    var ray_direction = normalize((primary_ray_target.xyz / primary_ray_target.w) - ray_origin);
    var ray_t_min = 0.0;

    // Path trace
    var radiance = vec3(0.0);
    var throughput = vec3(1.0);
    var p_bounce = 0.0;
    var previous_scatter_position = ray_origin;
    var glass_interactions = 0u;
    loop {
        let ray = trace_glass_ray(ray_origin, ray_direction, ray_t_min, RAY_T_MAX);
        if ray.kind != RAY_QUERY_INTERSECTION_NONE {
            let ray_hit = resolve_ray_hit_full(ray);
            let wo = -ray_direction;
            let material = materials[material_ids[ray.instance_index]];
            if (material.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u {
                let alpha = clamp(resolve_material_alpha(material, ray_hit.uv), 0.0, 1.0);
                if rand_f(&rng) >= alpha {
                    if glass_interactions >= 32u { break; }
                    glass_interactions += 1u;
                    p_bounce *= 1.0 - alpha;
                    ray_origin = offset_thin_glass_ray(ray_hit.world_position,
                        ray_hit.geometric_world_normal, ray_direction, RAY_T_MIN);
                    ray_t_min = RAY_T_MIN;
                    continue;
                }
                // Accepted coverage uses the ordinary surface BRDF and emission.
            }
            if (material.flags & MATERIAL_FLAG_ALPHA_BLEND) != 0u {
                // Bound chains of parallel panes/mirrors without consuming the
                // opaque BRDF path. Truncation loses only the remaining energy.
                if glass_interactions >= 32u { break; }
                glass_interactions += 1u;
                let alpha = clamp(resolve_material_alpha(material, ray_hit.uv), 0.0, 1.0);
                var emission_weight = 1.0;
                if p_bounce != 0.0 {
                    emission_weight = power_heuristic(p_bounce, random_emissive_light_solid_angle_pdf(ray_hit, previous_scatter_position));
                }
                radiance += emission_weight * throughput * alpha * ray_hit.material.emissive;
                // Geometric normal avoids normal maps bending transmission or
                // reflecting a ray through the wrong side of a thin interface.
                let weights = thin_glass_weights(wo, ray_hit.geometric_world_normal,
                    ray_hit.material.base_color, alpha, ray_hit.material.reflectance);
                let next = sample_thin_glass(wo, ray_hit.geometric_world_normal,
                    ray_hit.material.base_color, alpha, ray_hit.material.reflectance, rand_f(&rng));
                throughput *= next.throughput;
                if all(throughput <= vec3(0.0)) { break; }
                ray_direction = next.wi;
                ray_origin = offset_thin_glass_ray(ray_hit.world_position,
                    ray_hit.geometric_world_normal, ray_direction, RAY_T_MIN);
                ray_t_min = RAY_T_MIN;
                // A delta reflection cannot compete with the prior NEE ray.
                // Straight-through transmission keeps that competition alive.
                if next.reflected { p_bounce = 0.0; }
                else { p_bounce *= 1.0 - weights.a; }
                continue;
            }
            glass_interactions = 0u;

            // Emissive contribution
            var mis_weight = 1.0;
            if p_bounce != 0.0 { // Not first bounce
                let p_light = random_emissive_light_solid_angle_pdf(ray_hit, previous_scatter_position);
                mis_weight = power_heuristic(p_bounce, p_light);
            }
            radiance += mis_weight * throughput * ray_hit.material.emissive;

            // Sample direct lighting, but only if the surface is not mirror-like
            // TODO: randomly choose to use NEE or not with probability proportional to roughness and metallicness
            let is_perfectly_specular = ray_hit.material.roughness <= MIRROR_ROUGHNESS_THRESHOLD && ray_hit.material.metallic > 0.9999;
            if !is_perfectly_specular {
                let shadow_sample = sample_random_light_transmitted(ray_hit.world_position,
                    ray_hit.world_normal, ray_hit.geometric_world_normal, &rng);
                let direct_lighting = shadow_sample.light;

                mis_weight = 1.0;
                if direct_lighting.brdf_rays_can_hit {
                    let pdf_of_bounce = evaluate_brdf_pdf(wo, direct_lighting.wi, ray_hit.world_normal, ray_hit.material);
                    mis_weight = power_heuristic(direct_lighting.solid_angle_pdf, pdf_of_bounce * shadow_sample.continuation_probability);
                }

                let direct_lighting_brdf = evaluate_brdf(wo, direct_lighting.wi, ray_hit.world_normal, ray_hit.material);
                radiance += mis_weight * throughput * direct_lighting.radiance * direct_lighting.inverse_pdf * direct_lighting_brdf;
            }

            // Sample new ray direction from the material BRDF for next bounce and apply BRDF
            let next_bounce = evaluate_and_sample_brdf(wo, ray_hit.world_normal, ray_hit.material, &rng);
            if next_bounce.pdf == 0.0 { break; }
            ray_direction = next_bounce.wi;
            ray_origin = offset_thin_glass_ray(ray_hit.world_position, ray_hit.geometric_world_normal, ray_direction, RAY_T_MIN);
            ray_t_min = RAY_T_MIN;
            previous_scatter_position = ray_hit.world_position;
            p_bounce = select(next_bounce.pdf, 0.0, is_perfectly_specular);
            throughput *= next_bounce.throughput;

            // Russian roulette for early termination
            let p = luminance(throughput);
            if rand_f(&rng) > p { break; }
            throughput /= p;
        } else {
            // The ray left the scene: it sees the sky. The sky is not in the
            // light list, so BRDF sampling is the only way to it and the
            // contribution is taken whole (the camera ray draws the sky).
            radiance += throughput * sample_sky(ray_direction);
            break;
        }
    }

    // Camera exposure
    radiance *= view.exposure;

    // Accumulation over time via running average
    let new_color = mix(old_color.rgb, radiance, 1.0 / (old_color.a + 1.0));
    textureStore(accumulation_texture, global_id.xy, vec4(new_color, old_color.a + 1.0));
    textureStore(view_output, global_id.xy, vec4(new_color, 1.0));
#ifdef PATHTRACER_DEBUG_SAMPLE_COUNT
    textureStore(view_output, global_id.xy, vec4(vec3((old_color.a + 1.0) / 512.0) / view.exposure, 1.0));
#endif
}

