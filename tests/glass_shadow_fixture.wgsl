struct RawMaterial { flags: u32 }
struct ResolvedMaterial { base_color: vec3f, reflectance: f32 }
struct ResolvedRayHitFull { world_position: vec3f, geometric_world_normal: vec3f, material: ResolvedMaterial, uv: vec2f }
struct Ray { kind: u32, instance_index: u32, t: f32 }
const MATERIAL_FLAG_ALPHA_BLEND: u32 = 1u;
const MATERIAL_FLAG_DIFFUSE_BLEND: u32 = 2u;
const RAY_QUERY_INTERSECTION_NONE: u32 = 0u;
const RAY_T_MIN: f32 = 0.001;
var<private> materials: array<RawMaterial, 1>;
var<private> material_ids: array<u32, 1>;
@group(0) @binding(0) var<storage> config: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
fn trace_glass_ray(origin: vec3f, direction: vec3f, lo: f32, hi: f32) -> Ray {
    let t = ceil(lo);
    if t > config[1].y || t > hi { return Ray(0u,0u,hi); }
    return Ray(1u,0u,t);
}
fn resolve_ray_hit_full(ray: Ray) -> ResolvedRayHitFull {
    return ResolvedRayHitFull(vec3(0.0,0.0,ray.t), vec3(0.0,0.0,1.0),
        ResolvedMaterial(config[0].rgb,config[1].x),vec2(0.0));
}
fn resolve_material_alpha(material: RawMaterial, uv: vec2f) -> f32 { return config[0].a; }
@compute @workgroup_size(1)
fn probe() {
    materials[0].flags = select(MATERIAL_FLAG_ALPHA_BLEND,MATERIAL_FLAG_DIFFUSE_BLEND,config[1].w>0.0);
    if config[2].x>0.0 { materials[0].flags=0u; }
    let cosine=config[2].y;
    let direction=vec3(sqrt(1.0-cosine*cosine),0.0,cosine);
    output[0]=trace_shadow_transmission_impl(vec3(0.0),direction,config[1].z,config[1].w<2.0);
}

