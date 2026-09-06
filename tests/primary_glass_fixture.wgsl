@group(0) @binding(0) var<storage> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: vec4<f32>;
const RAY_T_MIN = 0.0001;
const RAY_T_MAX = 100000.0;
const RAY_QUERY_INTERSECTION_NONE = 0u;
const MATERIAL_FLAG_ALPHA_BLEND = 4u;
struct Intersection { kind: u32, instance_index: u32, t: f32 }
struct Material { flags: u32 }
var<private> materials: array<Material, 2>;
var<private> material_ids: array<u32, 2>;
struct ResolvedMaterial { base_color: vec3f, emissive: vec3f, reflectance: f32, perceptual_roughness: f32, roughness: f32, metallic: f32 }
struct ResolvedGPixel { world_position: vec3f, world_normal: vec3f, material: ResolvedMaterial }
struct Hit { world_position: vec3f, geometric_world_normal: vec3f, material: ResolvedMaterial, uv: vec2f }
fn trace_glass_ray(origin: vec3f, direction: vec3f, tmin: f32, tmax: f32) -> Intersection {
    let plane = floor(origin.z) + 1.0;
    let t = plane - origin.z;
    let wall_t = inputs[2].x - origin.z;
    if inputs[2].x > 0.0 && wall_t >= tmin && wall_t <= tmax && (plane > inputs[1].x || wall_t < t) {
        return Intersection(1u, 1u, wall_t);
    }
    if plane > inputs[1].x || t > tmax { return Intersection(0u, 0u, 0.0); }
    return Intersection(1u, 0u, t);
}
var<private> current_origin: vec3f;
fn resolve_ray_hit_full(ray: Intersection) -> Hit {
    let position = current_origin + vec3(0.0, 0.0, ray.t);
    current_origin = position + vec3(0.0, 0.0, RAY_T_MIN);
    return Hit(position, vec3(0.0, 0.0, -1.0), ResolvedMaterial(inputs[0].xyz, vec3(inputs[1].w), inputs[1].z, 0.0, 0.0, 0.0), vec2(0.0));
}
fn shade_surface_path(hit: Hit, wo: vec3f, rng: ptr<function, u32>) -> vec3f { return inputs[2].yzw; }
fn resolve_material_alpha(material: Material, uv: vec2f) -> f32 { return inputs[0].w; }
fn trace_glossy_path(pixel: vec2u, surface: ResolvedGPixel, distance: f32, wi: vec3f, pdf: f32, rng: ptr<function, u32>) -> vec3f { return vec3(1.0, 0.0, 0.0); }
@compute @workgroup_size(1)
fn probe() {
    materials[0].flags = MATERIAL_FLAG_ALPHA_BLEND;
    materials[1].flags = 0u;
    material_ids[1] = 1u;
    var rng = 0u;
    output = composite_primary_glass(vec2u(0u), vec3(0.0), vec3(0.0, 0.0, 1.0), inputs[1].y, vec3(1.0), &rng);
}
