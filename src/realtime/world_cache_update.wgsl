enable wgpu_ray_query;
#import bevy_solarik::scene_hit::scene_hit_is_miss

#ifdef RAY_MATERIAL_FOOTPRINTS
#import bevy_solarik::scene_bindings::{primary_ray_cone, advance_ray_cone, resolve_ray_hit_filtered}
#endif


#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_pbr::utils::{rand_f, rand_range_u, sample_cosine_hemisphere}
#import bevy_render::maths::PI
#import bevy_render::view::View
#import bevy_solarik::presample_light_tiles::{ResolvedLightSamplePacked, unpack_resolved_light_sample}
#import bevy_solarik::sampling::{calculate_resolved_light_contribution, trace_light_transmission_with_support, trace_shadow_transmission_impl}
#import bevy_solarik::scene_bindings::{trace_ray, resolve_ray_hit_full, sample_sky, RAY_T_MIN, ray_max_distance}
#import bevy_solarik::world_cache_priority::{selected_world_cache_cell_count, PRIORITY_UPDATED_COUNT}
#import bevy_solarik::world_cache::{
    WORLD_CACHE_MAX_TEMPORAL_SAMPLES,
    WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT,
    WORLD_CACHE_MAX_GI_RAY_DISTANCE,
    WORLD_CACHE_CELL_UPDATES_SOFT_CAP,
    query_two_sided_world_cache_with_support,
}
#import bevy_solarik::realtime_bindings::{
    light_tile_resolved_samples,
    view,
    constants,
    world_cache_active_cells_count,
    world_cache_a,
    world_cache_b,
    world_cache_life,
    world_cache_geometry_data,
    world_cache_radiance,
    world_cache_luminance_deltas,
    world_cache_active_cells_new_radiance,
}

@compute @workgroup_size(64, 1, 1)
fn sample_di(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= selected_world_cache_cell_count() { return; }

    let cell_index = world_cache_a[active_cell_id.x];
    let geometry_data = world_cache_geometry_data[cell_index];
    var rng = cell_index + constants.frame_index;

    let new_radiance = sample_random_light_ris(geometry_data.world_position, geometry_data.world_normal, workgroup_id.xy, &rng);

    world_cache_active_cells_new_radiance[active_cell_id.x] = new_radiance;
}

@compute @workgroup_size(64, 1, 1)
fn sample_gi(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    if active_cell_id.x >= selected_world_cache_cell_count() { return; }

    let cell_index = world_cache_a[active_cell_id.x];
    let geometry_data = world_cache_geometry_data[cell_index];
    var rng = cell_index + constants.frame_index;

    let ray_direction = sample_cosine_hemisphere(geometry_data.world_normal, &rng);
    // Traced to the end of the world so that geometry beyond the GI ray
    // distance is not mistaken for sky; such far hits contribute nothing, as
    // upstream's shorter ray had them.
    let ray = trace_ray(geometry_data.world_position + (geometry_data.world_normal * RAY_T_MIN), ray_direction, RAY_T_MIN, ray_max_distance(), RAY_FLAG_NONE);
    // trace_ray skips thin panes when finding the endpoint. Apply their energy
    // to this new cache sample once, excluding the endpoint itself.
    let connection_distance = select(ray.t - RAY_T_MIN, ray_max_distance(), scene_hit_is_miss(ray));
    var pending = world_cache_active_cells_new_radiance[active_cell_id.x];
    pending.w = max(pending.w, connection_distance + 2.0 * RAY_T_MIN);
    let transmission = trace_shadow_transmission_impl(geometry_data.world_position + geometry_data.world_normal * RAY_T_MIN, ray_direction, connection_distance, false).rgb;
    if scene_hit_is_miss(ray) {
        // Escaped to the sky: cosine sampling turns radiance into irradiance
        // with a factor of pi, the same units the cached direct light is in.
        pending = vec4(pending.rgb + transmission * PI * sample_sky(ray_direction), pending.w);
    } else if ray.t <= WORLD_CACHE_MAX_GI_RAY_DISTANCE {
#ifdef RAY_MATERIAL_FOOTPRINTS
        let cone = primary_ray_cone(view.clip_from_view, view.main_pass_viewport.zw, distance(geometry_data.world_position, view.world_position) + ray.t);
        let ray_hit = resolve_ray_hit_filtered(ray, ray_direction, cone);
#else
        let ray_hit = resolve_ray_hit_full(ray);
#endif
        let cell_life = atomicLoad(&world_cache_life[cell_index]);
        let radiance = query_two_sided_world_cache_with_support(ray_hit.world_position, ray_hit.geometric_world_normal, ray_hit.material.diffuse_transmission, view.world_position, ray.t, cell_life, &rng);
        pending = vec4(pending.rgb + transmission * ray_hit.material.base_color * radiance.rgb,
            max(pending.w, distance(geometry_data.world_position, ray_hit.world_position) + radiance.w));
    }
    world_cache_active_cells_new_radiance[active_cell_id.x] = pending;
}

@compute @workgroup_size(64, 1, 1)
fn blend_new_samples(@builtin(global_invocation_id) active_cell_id: vec3<u32>) {
    let selected_count = selected_world_cache_cell_count();
    if active_cell_id.x >= selected_count { return; }

    let cell_index = world_cache_a[active_cell_id.x];

    let old_radiance = world_cache_radiance[cell_index];
    let pending = world_cache_active_cells_new_radiance[active_cell_id.x];
    let new_radiance = pending.rgb;
    let luminance_delta = world_cache_luminance_deltas[cell_index];

    // https://bsky.app/profile/gboisse.bsky.social/post/3m5blga3ftk2a
    let sample_count = min(old_radiance.a + 1.0, WORLD_CACHE_MAX_TEMPORAL_SAMPLES);
    let update_probability = min(1.0, f32(WORLD_CACHE_CELL_UPDATES_SOFT_CAP) / f32(world_cache_active_cells_count));
    var blend_amount = cache_blend_amount(sample_count, luminance(old_radiance.rgb), luminance_delta, update_probability);
    if (constants.reset & 1u) != 0u {
        blend_amount = 1.0;
    }

    let blended_radiance = mix(old_radiance.rgb, new_radiance, blend_amount);
    let blended_luminance_delta = select(mix(luminance_delta, luminance(blended_radiance) - luminance(old_radiance.rgb), 1.0 / 8.0), 0.0, (constants.reset & 1u) != 0u);

    // Exponential history retains old contributions unless fully replaced.
    // Round support outward; nonfinite support deliberately fails local reuse.
    let new_support = pending.w * 1.00001 + 0.01;
    let old_support = world_cache_geometry_data[cell_index].support_radius;
    world_cache_geometry_data[cell_index].support_radius = select(
        max(old_support, new_support), new_support, blend_amount == 1.0);
    world_cache_radiance[cell_index] = vec4(blended_radiance, sample_count);
    world_cache_luminance_deltas[cell_index] = blended_luminance_delta;
    world_cache_geometry_data[cell_index].last_traced_frame = constants.render_frame;
    if active_cell_id.x % 64u == 0u {
        atomicAdd(&world_cache_b[PRIORITY_UPDATED_COUNT], min(64u, selected_count - active_cell_id.x));
    }
}

// Convert the history horizon from rendered frames to successful cell updates.
// Otherwise the update budget silently stretches history as the cache grows.
fn cache_blend_amount(sample_count: f32, old_luminance: f32, luminance_delta: f32, update_probability: f32) -> f32 {
    let alpha = abs(luminance_delta) / max(old_luminance, 0.001);
    let max_sample_count = mix(WORLD_CACHE_MAX_TEMPORAL_SAMPLES, 1.0, pow(saturate(alpha), 1.0 / 8.0));
    let frame_limited_samples = max(1.0, 16.0 * clamp(update_probability, 0.0, 1.0));
    return 1.0 / min(sample_count, min(max_sample_count, frame_limited_samples));
}

fn sample_random_light_ris(world_position: vec3<f32>, world_normal: vec3<f32>, workgroup_id: vec2<u32>, rng: ptr<function, u32>) -> vec4<f32> {
    var workgroup_rng = (workgroup_id.x * 5782582u) + workgroup_id.y;
    let light_tile_start = rand_range_u(128u, &workgroup_rng) * 1024u;

    var weight_sum = 0.0;
    var selected_sample_radiance = vec3(0.0);
    var selected_sample_target_function = 0.0;
    var selected_sample_world_position = vec4(0.0);
    let mis_weight = 1.0 / f32(WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT);
    for (var i = 0u; i < WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT; i++) {
        let tile_sample = light_tile_start + rand_range_u(1024u, rng);
        let resolved_light_sample = unpack_resolved_light_sample(light_tile_resolved_samples[tile_sample], view.exposure);
        let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, world_position, world_normal);

        let contribution = light_contribution.radiance * saturate(dot(light_contribution.wi, world_normal));
        let target_function = luminance(contribution);
        let resampling_weight = mis_weight * (target_function * light_contribution.inverse_pdf);

        weight_sum += resampling_weight;

        if rand_f(rng) < resampling_weight / weight_sum {
            selected_sample_radiance = contribution;
            selected_sample_target_function = target_function;
            selected_sample_world_position = resolved_light_sample.world_position;
        }
    }

    var support_radius = 0.0;
    var unbiased_contribution_weight = 0.0;
    if any(selected_sample_radiance > vec3(0.0)) {
        let inverse_target_function = select(0.0, 1.0 / selected_sample_target_function, selected_sample_target_function > 0.0);
        unbiased_contribution_weight = weight_sum * inverse_target_function;

        let shadow = trace_light_transmission_with_support(world_position + (world_normal * RAY_T_MIN), selected_sample_world_position);
        selected_sample_radiance *= shadow.energy.rgb;
        support_radius = shadow.distance + RAY_T_MIN;
    }

    return vec4(selected_sample_radiance * unbiased_contribution_weight, support_radius);
}
