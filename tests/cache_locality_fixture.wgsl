enable wgpu_ray_query;
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> cached: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> config: vec4<f32>;
struct SkyLight { material_transport_flags: u32 }
var<private> sky_light: SkyLight;
struct RawMaterial { flags: u32 }
struct Material { base_color: vec3<f32>, reflectance: f32 }
struct Hit { geometric_world_normal: vec3<f32>, material: Material, uv: vec2<f32> }
var<private> materials: array<RawMaterial,1>;
var<private> material_ids: array<u32,1>;
const MATERIAL_FLAG_ALPHA_BLEND=1u;
const MATERIAL_FLAG_DIFFUSE_BLEND=2u;
const RAY_T_MIN=0.001;
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>, lo:f32, hi:f32, flags:u32) -> RayIntersection {
    var query:ray_query;
    rayQueryInitialize(&query, scene, RayDesc(flags,255u,lo,hi,origin,direction));
    while rayQueryProceed(&query) {}
    return rayQueryGetCommittedIntersection(&query);
}
fn trace_glass_ray(origin:vec3<f32>, direction:vec3<f32>, lo:f32, hi:f32)->RayIntersection {
    return trace_ray(origin,direction,lo,hi,0u);
}
// Opaque TLAS uses the production fast branch. Separate pane fixtures
// exercise the ordered material branch with the same production function.
fn resolve_ray_hit_full(ray:RayIntersection)->Hit { return Hit(vec3(0.0,0.0,1.0),Material(vec3(1.0),0.5),vec2(0.0)); }
fn resolve_material_alpha(material:RawMaterial,uv:vec2<f32>)->f32 { return 1.0; }
fn thin_glass_weights(wo:vec3<f32>,n:vec3<f32>,t:vec3<f32>,a:f32,r:f32)->vec4<f32> { return vec4(0.0); }
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    let origin=vec3(f32(id.x)*2.0,0.0,0.0);
    let old=cached[id.x];
    let valid=old.z>0.0 && config.y==0.0 && cache_support_is_valid(origin,old.y,config.x);
    let shadow=trace_shadow_transmission_with_support(origin,vec3(0.0,0.0,1.0),10000.0,true);
    if !valid { cached[id.x]=vec4(shadow.energy.x,shadow.distance+0.01,1.0,0.0); }
    output[id.x]=vec4(f32(valid),cached[id.x].x,shadow.energy.x,shadow.distance);
}
