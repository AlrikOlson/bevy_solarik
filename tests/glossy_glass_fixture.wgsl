
struct Material { base_color: vec3f, emissive: vec3f, roughness: f32, metallic: f32, reflectance: f32 }
struct ResolvedGPixel { world_position: vec3f, world_normal: vec3f, material: Material }
struct ResolvedRayHitFull { world_position: vec3f, world_normal: vec3f, geometric_world_normal: vec3f, material: Material, uv: vec2f }
struct Ray { kind: u32, instance_index: u32, t: f32 }
struct RawMaterial { flags: u32 }
struct View { world_position: vec3f }
struct Light { wi: vec3f, inverse_pdf: f32, brdf_rays_can_hit: bool, radiance: vec3f }
const RAY_T_MIN: f32 = 0.001;
const RAY_T_MAX: f32 = 10000.0;
const RAY_QUERY_INTERSECTION_NONE: u32 = 0u;
const RAY_FLAG_NONE: u32 = 0u;
const MATERIAL_FLAG_ALPHA_BLEND: u32 = 1u;
const MIRROR_ROUGHNESS_THRESHOLD: f32 = 0.002;
const DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD: f32 = 0.4;
const SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD: f32 = 0.0225;
const WORLD_CACHE_CELL_LIFETIME: u32 = 1u;
const PI: f32 = 3.14159265;
var<private> materials: array<RawMaterial, 2> = array(RawMaterial(1u), RawMaterial(0u));
var<private> material_ids: array<u32, 2> = array(0u, 1u);
var<private> view: View;
var<private> steps: u32;
var<private> replacements: u32;
fn reflection_matrix(n: vec3f) -> mat3x3f {
    return mat3x3f(vec3(1.0,0.0,0.0), vec3(0.0,1.0,0.0), vec3(0.0,0.0,-1.0));
}
fn replace_primary_surface(pixel: vec2u, hit: ResolvedRayHitFull, rotations: mat3x3f, primary: vec3f) {
    replacements += 1u;
}
var<private> direction: vec3f;
@group(0) @binding(0) var<storage> config: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
// Synthetic intersection stream; all transport and branch handling under test
// comes from the production trace_glossy_path, not a reimplementation.
fn trace_ray(origin: vec3f, wi: vec3f, lo: f32, hi: f32, flags: u32) -> Ray {
    direction = wi;
    return Ray(1u, 1u, 1.0); // legacy traversal skips every pane
}
fn trace_glass_ray(origin: vec3f, wi: vec3f, lo: f32, hi: f32) -> Ray {
    direction = wi;
    steps += 1u;
    if wi.z > 0.0 || steps > u32(config[1].z) { return Ray(1u, 1u, 1.0); }
    return Ray(1u, 0u, 1.0);
}
fn resolve_ray_hit_full(ray: Ray) -> ResolvedRayHitFull {
    var m = Material(vec3(0.0), select(vec3(1.0), vec3(1.0, 0.0, 0.0), direction.z > 0.0), 1.0, 0.0, 0.5);
    if ray.instance_index == 0u { m = Material(config[0].rgb, vec3(0.0), 1.0, 0.0, config[1].x); }
    return ResolvedRayHitFull(vec3(0.0, 0.0, -f32(steps)), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), m, vec2(0.0));
}
fn resolve_material_alpha(m: RawMaterial, uv: vec2f) -> f32 { return config[0].a; }
fn rand_f(rng: ptr<function, u32>) -> f32 { return config[1].y; }
fn sample_sky(wi: vec3f) -> vec3f { return vec3(0.0); }
fn orthonormalize(n: vec3f) -> mat3x3f { return mat3x3f(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),n); }
fn get_cell_size(p: vec3f, eye: vec3f, rng: ptr<function,u32>) -> f32 { return 100.0; }
fn query_world_cache(p: vec3f, n: vec3f, eye: vec3f, t: f32, life: u32, rng: ptr<function,u32>) -> vec3f { return vec3(0.0); }
fn sample_random_light(p: vec3f, n: vec3f, rng: ptr<function,u32>) -> Light { return Light(n,1.0,false,vec3(0.0)); }
fn evaluate_brdf(wo: vec3f, wi: vec3f, n: vec3f, m: Material) -> vec3f { return vec3(0.0); }
fn sample_ggx_vndf(wo: vec3f, r: f32, rng: ptr<function,u32>) -> vec3f { return vec3(0.0); }
fn ggx_vndf_sample_invalid(wi: vec3f) -> bool { return true; }
fn ggx_vndf_pdf(wo: vec3f, wi: vec3f, r: f32) -> f32 { return 0.5; }
fn random_emissive_light_pdf(hit: ResolvedRayHitFull) -> f32 { return 0.5; }
fn power_heuristic(a: f32, b: f32) -> f32 { return a*a/(a*a+b*b); }
fn luminance(c: vec3f) -> f32 { return dot(c, vec3(0.2126,0.7152,0.0722)); }
@compute @workgroup_size(1)
fn probe() {
    var rng = 0u;
    let primary = ResolvedGPixel(vec3(0.0),vec3(0.0,0.0,1.0),Material(vec3(1.0),vec3(0.0),config[1].w,1.0,0.5));
    let result = trace_glossy_path(vec2u(0u), primary, 1.0, vec3(0.0,0.0,-1.0), 0.5, &rng);
    output[0] = vec4(result, f32(steps));
    output[1] = vec4(f32(replacements));
}
