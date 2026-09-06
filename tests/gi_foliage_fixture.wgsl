// Deterministic scene I/O; all endpoint generation, cache mixing and reuse are production WGSL.
@group(0) @binding(0) var<storage, read> config: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
const PI = 3.141592653589793;
const RAY_T_MIN = 0.001;
const RAY_T_MAX = 100000.0;
const RAY_FLAG_NONE = 0u;
const RAY_QUERY_INTERSECTION_NONE = 0u;
const SKY_SAMPLE_DISTANCE = 10000.0;
const WORLD_CACHE_CELL_LIFETIME = 10u;
const CONFIDENCE_WEIGHT_CAP = 8.0;
const MAX_GI_SAMPLE_AGE = 16.0;
struct Reservoir {
    sample_point_world_position: vec3<f32>, confidence_weight: f32,
    sample_point_world_normal: vec3<f32>, unbiased_contribution_weight: f32,
    radiance: vec3<f32>, sample_age: f32,
}
struct ReservoirMergeResult { merged_reservoir: Reservoir, selected_sample_radiance: vec3<f32>, wi: vec3<f32> }
struct Material { base_color: vec3<f32>, emissive: vec3<f32>, diffuse_transmission: f32 }
struct Hit { world_position: vec3<f32>, world_normal: vec3<f32>, geometric_world_normal: vec3<f32>, triangle_world_normal: vec3<f32>, material: Material }
struct Ray { kind: u32, t: f32 }
struct Direction { direction: vec3<f32>, inverse_pdf: f32 }
struct View { world_position: vec3<f32> }
var<private> view: View;
var<private> queries: u32;
var<private> errors: u32;
fn rand_f(rng: ptr<function, u32>) -> f32 { *rng += 1u; return 0.25; }
fn luminance(v: vec3<f32>) -> f32 { return dot(v, vec3(0.2126, 0.7152, 0.0722)); }
fn sample_sky_mixture(n: vec3<f32>, rng: ptr<function, u32>) -> Direction {
    let unused = rand_f(rng);
    return Direction(select(n, -n, config[0].z == 1.0), 4.0);
}
fn trace_ray(p: vec3<f32>, d: vec3<f32>, lo: f32, hi: f32, flags: u32) -> Ray {
    if distance(p, vec3(0.0, 0.0, config[0].y * RAY_T_MIN)) > 0.000001
        || lo != RAY_T_MIN || hi != RAY_T_MAX || flags != RAY_FLAG_NONE { errors += 1u; }
    return Ray(select(1u, 0u, config[0].z == 2.0), 2.0);
}
fn resolve_ray_hit_full(ray: Ray) -> Hit {
    let n = vec3(0.0, 0.0, -config[0].y);
    return Hit(-2.0 * n, normalize(n + vec3(config[0].w, 0.0, 0.0)), n,
        vec3(0.0, 0.0, 1.0), Material(vec3(0.2, 0.5, 0.8),
        select(vec3(0.0), vec3(1.0), config[0].z == 3.0), config[0].x));
}
fn sample_sky(d: vec3<f32>) -> vec3<f32> { return vec3(0.3, 0.6, 0.9); }
fn query_world_cache(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, t: f32, life: u32, rng: ptr<function, u32>) -> vec3<f32> {
    queries += 1u;
    let unused = rand_f(rng);
    if distance(p, vec3(0.0, 0.0, 2.0 * config[0].y)) > 0.000001
        || t != 2.0 || life != WORLD_CACHE_CELL_LIFETIME { errors += 1u; }
    return select(config[3].rgb, config[2].rgb, n.z > 0.0);
}
struct LightSample { world_position: vec4<f32> }
struct LightDraw { resolved_light_sample: LightSample }
struct LightContribution { radiance: vec3<f32>, inverse_pdf: f32, wi: vec3<f32> }
struct TransmittedLightContribution { light: LightContribution, continuation_probability: f32 }
fn light_direction() -> vec3<f32> {
    return vec3(sqrt(1.0 - config[4].y * config[4].y), 0.0, config[4].x * config[4].y);
}
fn generate_random_light_sample(rng: ptr<function, u32>) -> LightDraw {
    let unused = rand_f(rng);
    return LightDraw(LightSample(vec4(light_direction(), 0.0)));
}
fn calculate_resolved_light_contribution(s: LightSample, p: vec3<f32>, n: vec3<f32>) -> LightContribution {
    return LightContribution(config[2].rgb, 3.0, light_direction());
}
fn trace_light_transmission(p: vec3<f32>, light: vec4<f32>) -> vec4<f32> {
    let expected = vec3(0.0, 0.0, 2.0 * config[0].y + config[4].x * RAY_T_MIN);
    if distance(p, expected) > 0.000001 { errors += 1u; }
    return vec4(config[5].rgb, 1.0);
}
fn trace_point_visibility(p: vec3<f32>, endpoint: vec3<f32>) -> f32 { return 1.0; }
fn trace_shadow_transmission_impl(p: vec3<f32>, d: vec3<f32>, t: f32, coverage: bool) -> vec4<f32> {
    if coverage || abs(t - (2.0 - 2.0 * RAY_T_MIN)) > 0.000001 { errors += 1u; }
    return vec4(config[1].rgb, 1.0);
}
@compute @workgroup_size(1)
fn probe() {
    var rng = 0u;
    let p = vec3(0.0);
    let n = vec3(0.0, 0.0, config[0].y);
    var r = generate_initial_reservoir(p, n, &rng);
    output[0] = vec4(r.radiance, r.unbiased_contribution_weight);
    output[1] = vec4(r.sample_point_world_normal, r.confidence_weight);
    output[2] = vec4(0.0, 0.0, 0.0, f32(rng));
    output[5] = vec4(0.0, 0.0, f32(queries), f32(errors));
    if config[0].z != 0.0 { return; }
    output[2] = vec4(shade_gi_connection(p, n, r.sample_point_world_position, r.radiance, r.unbiased_contribution_weight), f32(rng));
    r.sample_age = 7.0;
    for (var i = 0u; i < 8u; i++) {
        r = merge_reservoirs(r, p, n, vec3(1.0), r, p, n, vec3(1.0), &rng).merged_reservoir;
        r.confidence_weight = min(r.confidence_weight, CONFIDENCE_WEIGHT_CAP);
    }
    output[3] = vec4(r.radiance, r.unbiased_contribution_weight);
    let beyond = vec3(5.0, 0.0, 3.0 * config[0].y);
    let beyond_normal = normalize(r.sample_point_world_position - beyond);
    let opposite = merge_reservoirs(empty_reservoir(), beyond, beyond_normal, vec3(1.0), r, p, n, vec3(1.0), &rng).merged_reservoir;
    output[4] = vec4(opposite.radiance * opposite.unbiased_contribution_weight, r.sample_age);
    output[5].x = jacobian(vec3(0.2, 0.0, 0.0), p, r.sample_point_world_position, r.sample_point_world_normal);
    output[5].y = jacobian(beyond, p, r.sample_point_world_position, r.sample_point_world_normal);
    output[5].w = f32(errors);
    r.sample_age = 0.0;
    for (var frame = 0u; frame < u32(MAX_GI_SAMPLE_AGE); frame++) {
        r = age_temporal_reservoir(r);
    }
    output[6] = vec4(r.radiance, r.confidence_weight);
}
