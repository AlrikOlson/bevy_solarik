@group(0) @binding(0) var<storage, read> config: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> world_cache_life: array<atomic<u32>>;
const PI = 3.14159265359;
const RAY_T_MIN = 0.001;
const RAY_T_MAX = 100000.0;
const RAY_FLAG_NONE = 0u;
const RAY_QUERY_INTERSECTION_NONE = 0u;
const WORLD_CACHE_MAX_GI_RAY_DISTANCE = 50.0;
const WORLD_CACHE_CELL_UPDATES_SOFT_CAP = 40000u;
struct View { world_position: vec3<f32> }
struct Constants { frame_index: u32 }
struct Geometry { world_position: vec3<f32>, world_normal: vec3<f32> }
struct Ray { kind: u32, t: f32 }
struct Material { base_color: vec3<f32>, diffuse_transmission: f32 }
struct Hit { world_position: vec3<f32>, geometric_world_normal: vec3<f32>, material: Material }
var<private> view = View(vec3(0.0, 0.0, 3.0));
var<private> constants = Constants(0u);
var<private> world_cache_active_cells_count = 1u;
var<private> world_cache_active_cell_indices: array<u32, 1>;
var<private> world_cache_geometry_data: array<Geometry, 1>;
var<private> world_cache_active_cells_new_radiance: array<vec3<f32>, 1>;
var<private> query_count = 0u;
var<private> trace_count = 0u;
var<private> invalid_connection = 0u;
var<private> life_mismatch = 0u;
fn rand_f(rng: ptr<function, u32>) -> f32 { *rng += 1u; return 0.5; }
fn sample_cosine_hemisphere(n: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> { return n; }
fn trace_ray(p: vec3<f32>, d: vec3<f32>, lo: f32, hi: f32, flags: u32) -> Ray {
    trace_count += 1u;
    if distance(p, vec3(0.0, 0.0, RAY_T_MIN)) > 0.00001 || d.z != 1.0 { invalid_connection += 1u; }
    return Ray(select(1u, 0u, config[0].z == 1.0), select(2.0, 51.0, config[0].z == 2.0));
}
fn trace_shadow_transmission_impl(p: vec3<f32>, d: vec3<f32>, t: f32, c: bool) -> vec4<f32> {
    let distance_limit = select(select(2.0, 51.0, config[0].z == 2.0) - RAY_T_MIN, RAY_T_MAX, config[0].z == 1.0);
    if abs(t - distance_limit) > 0.00001 { invalid_connection += 1u; }
    return vec4(config[1].rgb, 1.0);
}
fn resolve_ray_hit_full(ray: Ray) -> Hit {
    return Hit(vec3(0.0, 0.0, 2.0), vec3(0.0, 0.0, config[0].y), Material(vec3(0.2, 0.5, 0.8), config[0].x));
}
fn sample_sky(d: vec3<f32>) -> vec3<f32> { return vec3(0.3, 0.6, 0.9); }
fn query_world_cache(p: vec3<f32>, n: vec3<f32>, eye: vec3<f32>, ray_t: f32, life: u32, rng: ptr<function, u32>) -> vec3<f32> {
    query_count += 1u;
    *rng += 1u;
    if life != 7u || ray_t != 2.0 { life_mismatch += 1u; }
    return select(config[3].rgb, config[2].rgb, n.z > 0.0);
}
@compute @workgroup_size(1)
fn probe() {
    world_cache_geometry_data[0] = Geometry(vec3(0.0), vec3(0.0, 0.0, 1.0));
    world_cache_active_cells_count = select(1u, 100000u, config[0].z == 3.0);
    atomicStore(&world_cache_life[0], 7u);
    sample_gi(vec3(0u), vec3(select(0u, 1u, config[0].z == 4.0), 0u, 0u));
    output[0] = vec4(world_cache_active_cells_new_radiance[0], f32(query_count));
    output[1] = vec4(f32(trace_count), f32(invalid_connection), f32(life_mismatch), 0.0);
}
