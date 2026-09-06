// Synthetic intersections and light I/O; transport, diffuse BSDF and PDFs are production WGSL.
const PI: f32 = 3.141592653589793;
const RAY_T_MIN: f32 = 0.001;
const RAY_T_MAX: f32 = 10000.0;
const RAY_QUERY_INTERSECTION_NONE: u32 = 0u;
const MATERIAL_FLAG_ALPHA_BLEND: u32 = 1u;
const MATERIAL_FLAG_DIFFUSE_BLEND: u32 = 16u;
const MIRROR_ROUGHNESS_THRESHOLD: f32 = 0.002;
const DIFFUSE_GI_REUSE_ROUGHNESS_THRESHOLD: f32 = 0.4;
const SPECULAR_GI_FOR_DI_ROUGHNESS_THRESHOLD: f32 = 0.0225;
const WORLD_CACHE_CELL_LIFETIME: u32 = 1u;
struct ResolvedMaterial { base_color: vec3f, emissive: vec3f, reflectance: f32, perceptual_roughness: f32, roughness: f32, metallic: f32, diffuse_transmission: f32 }
struct ResolvedGPixel { world_position: vec3f, world_normal: vec3f, material: ResolvedMaterial }
struct ResolvedRayHitFull { world_position: vec3f, world_normal: vec3f, geometric_world_normal: vec3f, material: ResolvedMaterial, uv: vec2f }
struct RawMaterial { flags: u32 }
struct Ray { kind: u32, instance_index: u32, t: f32 }
struct View { world_position: vec3f }
struct Light { wi: vec3f, inverse_pdf: f32, brdf_rays_can_hit: bool, radiance: vec3f, solid_angle_pdf: f32 }
struct ShadowSample { light: Light, continuation_probability: f32 }
@group(0) @binding(0) var<storage> config: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
var<private> materials: array<RawMaterial, 2> = array(RawMaterial(1u), RawMaterial(0u));
var<private> material_ids: array<u32, 2> = array(0u, 1u);
var<private> view: View;
var<private> steps: u32;
var<private> bad_offset: u32;
var<private> cache_queries: u32;
var<private> replacements: u32;
fn reflection_matrix(n: vec3f) -> mat3x3f { return mat3x3f(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,0.0,-1.0)); }
fn replace_primary_surface(pixel: vec2u, hit: ResolvedRayHitFull, rotations: mat3x3f, primary: vec3f) { replacements += 1u; }
fn trace_glass_ray(origin: vec3f, wi: vec3f, lo: f32, hi: f32) -> Ray {
    steps += 1u;
    let pane = config[0].w == 4.0;
    if pane && steps == 1u { return Ray(1u, 0u, 1.0); }
    if steps == select(1u, 2u, pane) { return Ray(1u, 1u, 4.0); }
    if origin.z * wi.z <= 0.0 { bad_offset += 1u; }
    return Ray(0u, 1u, 1.0);
}
fn resolve_ray_hit_full(ray: Ray) -> ResolvedRayHitFull {
    var m = ResolvedMaterial(vec3(0.2,0.5,0.8),vec3(config[0].z),0.0,1.0,1.0,0.0,config[0].x);
    if ray.instance_index == 0u { m = ResolvedMaterial(vec3(0.5),vec3(0.0),0.0,0.0,0.0,0.0,0.0); }
    return ResolvedRayHitFull(vec3(0.0),vec3(0.0,0.0,1.0),vec3(0.0,0.0,1.0),m,vec2(0.0));
}
fn resolve_material_alpha(m: RawMaterial, uv: vec2f) -> f32 { return 1.0; }
fn sample_sky(wi: vec3f) -> vec3f { return vec3(f32(wi.z < 0.0 && config[0].w != 1.0 && config[0].w != 2.0)); }
fn analytic_light_radiance(origin: vec3f, wi: vec3f, limit: f32, owned: bool, scatter: vec3f) -> vec3f { return vec3(0.0); }
fn get_cell_size(p: vec3f, eye: vec3f, rng: ptr<function,u32>) -> f32 { return 0.1; }
fn query_world_cache(p: vec3f, n: vec3f, eye: vec3f, t: f32, life: u32, rng: ptr<function,u32>) -> vec3f { cache_queries += 1u; return vec3(0.0); }
fn sample_random_light_transmitted(p: vec3f, n: vec3f, geo: vec3f, rng: ptr<function,u32>) -> ShadowSample {
    return ShadowSample(Light(-n,1.0,false,vec3(select(0.0,PI,config[0].w == 1.0)),1.0),1.0);
}
fn random_emissive_light_solid_angle_pdf(hit: ResolvedRayHitFull, previous: vec3f) -> f32 { return 0.5; }
fn power_heuristic(a: f32, b: f32) -> f32 { return a*a / max(a*a+b*b, 0.00000001); }
fn luminance(c: vec3f) -> f32 { return dot(c,vec3(0.2126,0.7152,0.0722)); }
fn calculate_F0(c:vec3f,m:f32,r:vec3f)->vec3f { return mix(0.16*r*r,c,m); }
fn calculate_diffuse_color(c:vec3f,m:f32,t:f32,s:f32)->vec3f { return c*(1.0-m); }
// F0=0 at the leaf makes this a diffuse-only energy probe. Capture tests use the full GGX shader.
fn D_GGX(a:f32,b:f32)->f32 { return 0.0; }
fn V_SmithGGXCorrelated(a:f32,b:f32,c:f32)->f32 { return 0.0; }
fn specular_multiscatter(a:f32,b:f32,c:vec3f,d:vec3f,e:vec2f,f:f32)->vec3f { return vec3(0.0); }
fn ggx_vndf_pdf(a:vec3f,b:vec3f,c:f32)->f32 { return 0.0; }
fn sample_ggx_vndf(a:vec3f,b:f32,r:ptr<function,u32>)->vec3f { return vec3(0.0,0.0,1.0); }
fn ggx_vndf_sample_invalid(a:vec3f)->bool { return true; }
fn orthonormalize(n:vec3f)->mat3x3f { return mat3x3f(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),n); }
fn rand_f(r:ptr<function,u32>)->f32 { *r = *r * 1664525u + 1013904223u; return f32(*r >> 8u) / 16777216.0; }
fn sample_cosine_hemisphere(n:vec3f,r:ptr<function,u32>)->vec3f {
    let u=rand_f(r); let phi=2.0*PI*rand_f(r);
    return vec3(sqrt(u)*cos(phi),sqrt(u)*sin(phi),sqrt(1.0-u))*n.z;
}
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3u) {
    var rng = id.x * 747796405u + 2891336453u;
    let m = ResolvedMaterial(vec3(1.0),vec3(0.0),0.5,sqrt(config[0].y),config[0].y,1.0,0.0);
    let primary = ResolvedGPixel(vec3(0.0,0.0,1.0),vec3(0.0,0.0,1.0),m);
    let result = trace_glossy_path(vec2u(0u),primary,1.0,vec3(0.0,0.0,-1.0),0.5,&rng);
    output[id.x*2u] = vec4(result,f32(steps));
    output[id.x*2u+1u] = vec4(f32(bad_offset),f32(cache_queries),f32(replacements),0.0);
}

