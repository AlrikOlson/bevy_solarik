enable wgpu_ray_query;

#define_import_path bevy_solarik::sampling

#import bevy_pbr::lighting::D_GGX
#import bevy_pbr::utils::{rand_f, rand_vec2f, rand_u, rand_range_u}
#import bevy_render::maths::{PI_2, orthonormalize}
#import bevy_solarik::thin_glass::thin_glass_weights
#import bevy_solarik::scene_bindings::{trace_glass_ray, resolve_ray_hit_full, MATERIAL_FLAG_ALPHA_BLEND, trace_ray, RAY_T_MIN, RAY_T_MAX, light_sources, directional_lights, local_lights, LightSource, LIGHT_SOURCE_KIND_DIRECTIONAL, light_source_is_emissive_mesh, resolve_triangle_data_full, materials, material_ids, resolve_material_alpha, MATERIAL_FLAG_DIFFUSE_BLEND, ResolvedRayHitFull, MIRROR_ROUGHNESS_THRESHOLD}

fn power_heuristic(f: f32, g: f32) -> f32 {
    return balance_heuristic(f * f, g * g);
}

fn balance_heuristic(f: f32, g: f32) -> f32 {
    // Need to guard against NaNs since ReSTIR reservoirs can have UCW=0
    if f == 0.0 {
        return 0.0;
    }
    return max(0.0, 1.0 / (1.0 + (g / f)));
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
// Result is invalid when output.z <= 0.0, and must be discarded
fn sample_ggx_vndf(wi_tangent: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(vec3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(vec2(i.xy));
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = select(i_std.z, k * i_std.z, i.z > 0.0);
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

fn ggx_vndf_sample_invalid(ray_tangent: vec3<f32>) -> bool {
    return !(ray_tangent.z > 0.0);
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
fn ggx_vndf_pdf(wi_tangent: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32) -> f32 {
    // Mirror BRDF case
    if roughness <= MIRROR_ROUGHNESS_THRESHOLD {
        let mirror_wo = vec3(-wi_tangent.xy, wi_tangent.z);
        if all(abs(mirror_wo - wo_tangent) < vec3(0.0001)) {
            return bitcast<f32>(0x7F800000u); // INF
        } else {
            return 0.0;
        }
    }

    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    var pdf: f32;
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        pdf = ndf / (2.0 * (k * i.z + t));
    } else {
        pdf = ndf * (t - i.z) / (2.0 * len2);
    }

    return select(pdf, 0.0, isnan(pdf));
}

fn isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

const NULL_LIGHT_ID = 0xFFFFFFFFu;

struct LightSample {
    light_id: u32,
    seed: u32,
}

// What kind of light a resolved sample came from, in world_position.w.
// (Typed: the shader composer only exports typed constants across modules.)
const LIGHT_SAMPLE_DIRECTIONAL: f32 = 0.0;   // xyz is the direction to the light
const LIGHT_SAMPLE_EMISSIVE_MESH: f32 = 1.0; // xyz is a point on a triangle
const LIGHT_SAMPLE_LOCAL: f32 = 2.0;         // xyz is a point on a point/spot light's sphere

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
    // Point and spot lights only: the cone axis, cos(outer) and cos(inner)
    // (-1 = no cone) and the raster path's range. Receiver-dependent, so
    // they are applied in calculate_resolved_light_contribution, not here.
    spot_direction: vec3<f32>,
    spot_cos: vec2<f32>,
    range: f32,
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
    brdf_rays_can_hit: bool,
    // Solid-angle density for MIS only; inverse_pdf remains in area measure.
    solid_angle_pdf: f32,
}

struct LightContributionNoPdf {
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

struct GenerateRandomLightSampleResult {
    light_sample: LightSample,
    resolved_light_sample: ResolvedLightSample,
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    light_contribution.radiance *= trace_light_visibility(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

// A thin diffuse receiver can see lights on either side. Evaluate geometry
// at the true surface; offset only the visibility ray toward the sampled side.
fn sample_random_light_two_sided(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, geometric_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    var contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    let side = select(-1.0, 1.0, dot(geometric_normal, contribution.wi) >= 0.0);
    let origin = ray_origin + geometric_normal * (side * RAY_T_MIN);
    contribution.radiance *= trace_light_visibility(origin, sample.resolved_light_sample.world_position);
    return contribution;
}

// Convert area density to solid angle at the previous scattering vertex.
fn area_to_solid_angle_pdf(area_pdf: f32, distance_squared: f32, light_cosine: f32) -> f32 {
    if area_pdf <= 0.0 || distance_squared <= 0.0 { return 0.0; }
    return area_pdf * distance_squared / max(abs(light_cosine), 0.000001);
}

fn random_emissive_light_solid_angle_pdf(hit: ResolvedRayHitFull, previous_position: vec3<f32>) -> f32 {
    let delta = hit.world_position - previous_position;
    let distance_squared = dot(delta, delta);
    if distance_squared <= 0.0 { return 0.0; }
    let cosine = dot(-delta / sqrt(distance_squared), hit.triangle_world_normal);
    return area_to_solid_angle_pdf(random_emissive_light_pdf(hit), distance_squared, cosine);
}

fn random_emissive_light_pdf(hit: ResolvedRayHitFull) -> f32 {
    let area = f32(hit.triangle_count) * hit.triangle_area;
    if area <= 0.0 { return 0.0; }
    return hit.light_probability / area;
}

fn generate_random_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let light_count = arrayLength(&light_sources);
    let bin_id = rand_range_u(light_count, rng);
    let bin = light_sources[bin_id];
    let light_id = select(bin.alias_index, bin_id, rand_f(rng) < bin.alias_threshold);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if light_source_is_emissive_mesh(light_source) {
        let triangle_count = light_source.kind >> 1u;
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf /= light_source.selection_probability;

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    if light_source.kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let directional_light = directional_lights[light_source.id];

#ifndef NO_DIRECTIONAL_LIGHT_SOFT_SHADOWS
        // Sample a random direction within a cone whose base is the sun approximated as a disk
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(&rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        let x = cos(phi) * sin_theta;
        let y = sin(phi) * sin_theta;
        var direction_to_light = vec3(x, y, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with the light direction
        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;
#else
        let direction_to_light = directional_light.direction_to_light;
#endif

        return ResolvedLightSample(
            vec4(direction_to_light, LIGHT_SAMPLE_DIRECTIONAL),
            -direction_to_light,
            directional_light.luminance,
            directional_light.inverse_pdf,
            vec3(0.0, 0.0, 1.0),
            vec2(-1.0),
            0.0,
        );
    } else if !light_source_is_emissive_mesh(light_source) {
        // A point or spot light: a point uniformly on its sphere, with the
        // sphere's outward normal, surface radiance and area. The sample is
        // resolved without a receiver (the tiles presample it, and the
        // resampling re-resolves it at other pixels), so the whole sphere is
        // sampled rather than the cap a receiver would see; the far half has
        // cos_theta_light <= 0 and falls out of the resampling at no cost.
        // Expectation at a facing receiver: radiance × π r² / d², the light's
        // intensity over d², which is what the raster path computes.
        let light = local_lights[light_source.id];
        let direction = uniform_sphere_direction(light_sample.seed);

        return ResolvedLightSample(
            vec4(light.position + direction * light.radius, LIGHT_SAMPLE_LOCAL),
            direction,
            light.radiance,
            light.inverse_pdf,
            light.direction,
            vec2(light.cos_outer, light.cos_inner),
            light.range,
        );
    } else {
        let triangle_count = light_source.kind >> 1u;
        let triangle_id = light_sample.light_id & 0xFFFFu;
        let barycentrics = triangle_barycentrics(light_sample.seed);
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        let raw_material = materials[material_ids[light_source.id]];
        var emission = triangle_data.material.emissive.rgb;
        if (raw_material.flags & (MATERIAL_FLAG_DIFFUSE_BLEND | MATERIAL_FLAG_ALPHA_BLEND)) != 0u {
            emission *= clamp(resolve_material_alpha(raw_material, triangle_data.uv), 0.0, 1.0);
        }
        return ResolvedLightSample(
            vec4(triangle_data.world_position, LIGHT_SAMPLE_EMISSIVE_MESH),
            triangle_data.triangle_world_normal,
            emission,
            f32(triangle_count) * triangle_data.triangle_area,
            vec3(0.0, 0.0, 1.0),
            vec2(-1.0),
            0.0,
        );
    }
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let positional = resolved_light_sample.world_position.w != LIGHT_SAMPLE_DIRECTIONAL;
    let ray = resolved_light_sample.world_position.xyz - select(vec3(0.0), ray_origin, positional);
    let light_distance = length(ray);
    let wi = ray / light_distance;

    let cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
    let light_distance_squared = light_distance * light_distance;

    var radiance = resolved_light_sample.radiance * (cos_theta_light / light_distance_squared);
    if resolved_light_sample.world_position.w == LIGHT_SAMPLE_LOCAL {
        radiance *= local_light_attenuation(resolved_light_sample, wi, light_distance_squared);
    }

    // Only emissive meshes are geometry a BRDF ray can hit; the others are
    // reached by next-event estimation alone, so their MIS weight is 1.
    let brdf_rays_can_hit = resolved_light_sample.world_position.w == LIGHT_SAMPLE_EMISSIVE_MESH;
    var solid_angle_pdf = 0.0;
    if brdf_rays_can_hit && resolved_light_sample.inverse_pdf > 0.0 {
        solid_angle_pdf = area_to_solid_angle_pdf(1.0 / resolved_light_sample.inverse_pdf,
            light_distance_squared, dot(-wi, resolved_light_sample.world_normal));
    }
    return LightContribution(radiance, resolved_light_sample.inverse_pdf, wi, brdf_rays_can_hit, solid_angle_pdf);
}

// The raster path's cone and range on a point or spot light sample
// (bevy_pbr's pbr_lighting.wgsl: Filament's smooth cone and range window),
// so both paths compute the same irradiance from the same lamp. `wi` points
// from the receiver to the sample.
fn local_light_attenuation(sample: ResolvedLightSample, wi: vec3<f32>, light_distance_squared: f32) -> f32 {
    var attenuation = 1.0;
    let cos_outer = sample.spot_cos.x;
    if cos_outer > -1.0 {
        let cd = dot(sample.spot_direction, -wi);
        let t = saturate((cd - cos_outer) / max(sample.spot_cos.y - cos_outer, 1e-4));
        attenuation = t * t;
    }
    let factor = light_distance_squared / (sample.range * sample.range);
    let window = saturate(1.0 - factor * factor);
    return attenuation * window * window;
}

// A uniformly distributed direction on the unit sphere from a seed.
fn uniform_sphere_direction(seed: u32) -> vec3<f32> {
    var rng = seed;
    let random = rand_vec2f(&rng);
    let z = 1.0 - 2.0 * random.x;
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = PI_2 * random.y;
    return vec3(r * cos(phi), r * sin(phi), z);
}

fn resolve_and_calculate_light_contribution(light_sample: LightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContributionNoPdf {
    let resolved_light_sample = resolve_light_sample(light_sample, light_sources[light_sample.light_id >> 16u]);
    let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, origin_world_normal);
    return LightContributionNoPdf(light_contribution.radiance, light_contribution.wi);
}

struct TransmittedLightContribution {
    light: LightContribution,
    continuation_probability: f32,
}

// RGB transport is intentionally separate from scalar ReSTIR visibility.
fn sample_random_light_transmitted(ray_origin: vec3<f32>, world_normal: vec3<f32>, geometric_normal: vec3<f32>, rng: ptr<function, u32>) -> TransmittedLightContribution {
    let sample = generate_random_light_sample(rng).resolved_light_sample;
    var light = calculate_resolved_light_contribution(sample, ray_origin, world_normal);
    let side = select(-1.0, 1.0, dot(geometric_normal, light.wi) >= 0.0);
    let origin = ray_origin + geometric_normal * (side * RAY_T_MIN);
    let transmission = trace_light_transmission(origin, sample.world_position);
    light.radiance *= transmission.rgb;
    return TransmittedLightContribution(light, transmission.a);
}

// A GI reservoir stores outgoing endpoint radiance, never receiver transmission.
// Both diffuse and rough-specular consumers evaluate the current connection.
fn shade_gi_connection(world_position: vec3<f32>, world_normal: vec3<f32>, endpoint: vec3<f32>, radiance: vec3<f32>, weight: f32) -> vec3<f32> {
    if weight <= 0.0 || all(radiance <= vec3(0.0)) { return vec3(0.0); }
    let origin = world_position + world_normal * RAY_T_MIN;
    if trace_point_visibility(origin, endpoint) == 0.0 { return vec3(0.0); }
    let delta = endpoint - origin;
    let distance = length(delta);
    if distance <= RAY_T_MIN { return vec3(0.0); }
    let transmission = trace_shadow_transmission_impl(origin, delta / distance, distance - RAY_T_MIN, false);
    return radiance * weight * transmission.rgb;
}

// Connection energy and competing straight-through BSDF probability.
fn trace_light_transmission(origin: vec3<f32>, light_position: vec4<f32>) -> vec4<f32> {
    var direction = light_position.xyz;
    var distance = RAY_T_MAX;
    if light_position.w != LIGHT_SAMPLE_DIRECTIONAL {
        let delta = light_position.xyz - origin;
        distance = length(delta);
        if distance <= RAY_T_MIN { return vec4(0.0); }
        direction = delta / distance;
    }
    return trace_shadow_transmission(origin, direction, distance - RAY_T_MIN);
}

// RGB straight-through energy plus probability of the competing BSDF path.
// Preserve one ray origin and advance t_min, so panes cannot extend past the light.
fn trace_shadow_transmission(origin: vec3<f32>, direction: vec3<f32>, ray_t_max: f32) -> vec4<f32> {
    return trace_shadow_transmission_impl(origin, direction, ray_t_max, true);
}

// GI endpoint/visibility tracing already samples diffuse alpha coverage.
fn trace_shadow_transmission_impl(origin: vec3<f32>, direction: vec3<f32>, ray_t_max: f32, include_coverage: bool) -> vec4<f32> {
    var transmission = vec4(1.0);
    var ray_t_min = RAY_T_MIN;
    for (var panes = 0u; panes <= 32u; panes += 1u) {
        if ray_t_max < ray_t_min { return transmission; }
        let ray = trace_glass_ray(origin, direction, ray_t_min, ray_t_max);
        if ray.kind == RAY_QUERY_INTERSECTION_NONE { return transmission; }
        if panes == 32u { return vec4(0.0); }
        let raw_material = materials[material_ids[ray.instance_index]];
        let hit = resolve_ray_hit_full(ray);
        let alpha = clamp(resolve_material_alpha(raw_material, hit.uv), 0.0, 1.0);
        if (raw_material.flags & MATERIAL_FLAG_ALPHA_BLEND) != 0u {
            let weights = thin_glass_weights(-direction, hit.geometric_world_normal,
                hit.material.base_color, alpha, hit.material.reflectance);
            transmission *= vec4(weights.rgb, 1.0 - weights.a);
        } else if (raw_material.flags & MATERIAL_FLAG_DIFFUSE_BLEND) != 0u {
            if include_coverage { transmission *= 1.0 - alpha; }
        } else {
            return vec4(0.0);
        }
        if all(transmission.rgb <= vec3(0.0)) { return vec4(0.0); }
        ray_t_min = ray.t + RAY_T_MIN;
    }
    return vec4(0.0);
}
// End shadow transport

fn trace_light_visibility(ray_origin: vec3<f32>, light_sample_world_position: vec4<f32>) -> f32 {
    var ray_direction = light_sample_world_position.xyz;
    var ray_t_max = RAY_T_MAX;

    if light_sample_world_position.w != LIGHT_SAMPLE_DIRECTIONAL {
        let ray = ray_direction - ray_origin;
        let dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN;
    }

    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn trace_point_visibility(ray_origin: vec3<f32>, point: vec3<f32>) -> f32 {
    let ray = point - ray_origin;
    let dist = length(ray);
    let ray_direction = ray / dist;

    let ray_t_max = dist - RAY_T_MIN;
    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}
