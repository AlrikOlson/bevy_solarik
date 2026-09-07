const LIGHT_SAMPLE_DIRECTIONAL: f32 = 0.0;
const LIGHT_SAMPLE_EMISSIVE_MESH: f32 = 1.0;
const LIGHT_SAMPLE_LOCAL: f32 = 2.0;
struct ResolvedRayHitFull {
    world_position: vec3f, triangle_world_normal: vec3f,
    triangle_count: u32, triangle_area: f32, light_probability: f32,
}
fn local_light_attenuation(s: ResolvedLightSample, wi: vec3f, d: f32) -> f32 { return 1.0; }
@group(0) @binding(0) var<storage, read> config: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
    let a = config[0].x;
    let d = config[0].y;
    let pmf = config[0].z;
    let uv = (vec2f(f32(id.x % 256u), f32(id.x / 256u)) + 0.5) / 256.0 * 2.0 - 1.0;
    let position = vec3(uv * a, d);
    let r2 = dot(position, position);
    let cosine = d / sqrt(r2);
    let area = 4.0 * a * a;
    let cone_radius = config[0].w;
    let sample = ResolvedLightSample(vec4(position,1.0), vec3(0.0,0.0,-1.0),
        vec3(1.0), area/pmf, vec3(0.0,0.0,-1.0), vec2(cone_radius), select(0.0, -2.0, cone_radius > 0.0));
    let light = calculate_resolved_light_contribution(sample, vec3(0.0), vec3(0.0,0.0,1.0));
    let hit = ResolvedRayHitFull(position, vec3(0.0,0.0,-1.0), 2u, area/2.0, pmf);
    let hit_pdf = random_emissive_light_solid_angle_pdf(hit, vec3(0.0));
    let brdf_pdf = cosine / 3.141592653589793;
    let w_light = power_heuristic(light.solid_angle_pdf, brdf_pdf);
    let w_bounce = power_heuristic(brdf_pdf, hit_pdf);
    // Quadrature of both strategy expectations, including emitter-selection PMF.
    let radiance = (w_light + w_bounce) * pmf * light.radiance.x * light.inverse_pdf * brdf_pdf;
    output[id.x] = vec4(radiance, hit_pdf, light.solid_angle_pdf, w_light+w_bounce);
}

