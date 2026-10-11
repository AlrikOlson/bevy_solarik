struct Request { origin_min:vec4f, direction_max:vec4f, policy:vec4u }
struct Output { a:vec4u,b:vec4u }
@group(0) @binding(0) var tlas:acceleration_structure;
@group(0) @binding(1) var<storage,read> requests:array<Request>;
@group(0) @binding(2) var<storage,read_write> output:array<Output>;
const RAY_NO_CULL=255u;
const MATERIAL_FLAG_DOUBLE_SIDED=1u;
const MATERIAL_FLAG_OPAQUE=2u;
struct ForestEvent {position_distance:vec4f,normal_hit:vec4f,color_coverage:vec4f,surface:vec4f,source:vec4u}
struct ForestSceneEvent {event:ForestEvent,previous_position:vec3f}
var<private> fixture_coarse:bool;
fn forest_next_seed()->vec2u {return vec2u(0u);}
// Controlled event isolates nearest tagged dispatch. Real finite support is
// separately exercised by the renderer-kernel and native forest scene controls.
fn forest_trace_scene_impl(origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,seed:vec2u,primary:bool)->ForestSceneEvent {
    var result:ForestSceneEvent;
    if fixture_coarse && minimum<=1.25 && maximum>1.25 {
        result.event.position_distance=vec4f(origin+direction*1.25,1.25);
        result.event.normal_hit=vec4f(-direction,1.0);
        result.event.color_coverage=vec4f(0.2,0.4,0.1,0.7);
        result.event.surface=vec4f(0.0625,0.0,0.5,0.3);
        result.previous_position=result.event.position_distance.xyz+vec3f(3.0,0.0,0.0);
    }
    return result;
}
struct Sky { relative_ray_min:f32 }
const sky_light=Sky(0.000001);
fn ray_max_distance()->f32 {return 6.0;}
// Synthetic material acceptance isolates the unchanged hardware query from its adapter.
fn candidate_is_solid(hit:RayIntersection,glass:bool,origin:vec3f,direction:vec3f)->bool {
    if hit.instance_custom_data==0u {return origin.x>=0.0;}
    if hit.instance_custom_data==1u {return glass;}
    return true;
}
struct Material {flags:u32,alpha:f32,base_color:vec3f,base_color_alpha:f32,perceptual_roughness:f32,metallic:f32,reflectance:f32}
const materials=array<Material,3>(Material(1u,0.25,vec3f(0.0),0.25,1.0,0.0,0.5),
    Material(0u,0.5,vec3f(0.0),0.5,1.0,0.0,0.5),Material(1u,0.75,vec3f(0.0),0.75,1.0,0.0,0.5));
const material_ids=array<u32,3>(0u,1u,2u);
fn resolve_material_alpha(material:Material,uv:vec2f)->f32 {return material.alpha;}
struct ResolvedMaterial {base_color:vec3f,perceptual_roughness:f32,roughness:f32,metallic:f32,reflectance:f32}
struct ResolvedRayHitFull {world_normal:vec3f,geometric_world_normal:vec3f,world_position:vec3f,
    previous_frame_world_position:vec3f,triangle_world_normal:vec3f,material:ResolvedMaterial}
var<private> resolves:u32;
fn resolve_triangle_data_filtered(slot:u32,primitive:u32,bary:vec3f,direction:vec3f,cone:vec2f)->ResolvedRayHitFull {
    resolves+=1u;
    var hit:ResolvedRayHitFull;hit.world_normal=vec3f(1.0);hit.geometric_world_normal=vec3f(2.0);return hit;
}
fn same(a:SceneHit,b:SceneHit)->bool {
    if a.kind!=b.kind {return false;}
    if scene_hit_is_miss(a) {return true;}
    return a.t==b.t && a.triangle.slot==b.triangle.slot
        && a.triangle.primitive==b.triangle.primitive
        && all(a.triangle.barycentrics==b.triangle.barycentrics)
        && a.triangle.front_face==b.triangle.front_face;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3u) {
    if id.x>=arrayLength(&requests) {return;}
    let r=requests[id.x];
    fixture_coarse=false;
    let raw=trace_triangle_ray_impl(r.origin_min.xyz,r.direction_max.xyz,r.origin_min.w,r.direction_max.w,r.policy.x,r.policy.y!=0u);
    let hit=scene_hit_from_triangle(raw);
    let query=trace_ray_impl(r.origin_min.xyz,r.direction_max.xyz,r.origin_min.w,r.direction_max.w,r.policy.x,r.policy.y!=0u);
    var mismatch=u32(!same(hit,query));
    if raw.kind==RAY_QUERY_INTERSECTION_TRIANGLE {
        mismatch |= u32(!scene_hit_is_triangle(hit) || raw.t!=hit.t
            || raw.instance_custom_data!=hit.triangle.slot || raw.primitive_index!=hit.triangle.primitive
            || any(raw.barycentrics!=hit.triangle.barycentrics) || raw.front_face!=hit.triangle.front_face);
    } else {mismatch |= u32(!scene_hit_is_miss(hit));}
    var invalid=scene_triangle(3.0,0u,0u,vec2f(0.25),false);
    invalid.kind=array<u32,3>(SCENE_HIT_MISS,SCENE_HIT_COARSE,SCENE_HIT_INVALID)[id.x%3u];
    resolves=0u;
    let rejected=resolve_ray_hit_filtered(invalid,vec3f(0.0),vec2f(0.0));
    var guard_bad=u32(scene_hit_material(invalid).flags!=0u || scene_hit_alpha(invalid,vec2f(0.0))!=0.0
        || any(rejected.world_normal!=vec3f(0.0)) || resolves!=0u);
    fixture_coarse=true;
    let mixed=trace_ray_impl(r.origin_min.xyz,r.direction_max.xyz,r.origin_min.w,r.direction_max.w,r.policy.x,r.policy.y!=0u);
    let expected=r.origin_min.w<=1.25 && r.direction_max.w>1.25 && (scene_hit_is_miss(hit) || hit.t>1.25);
    guard_bad |= u32(scene_hit_is_coarse(mixed)!=expected);
    if expected {
        resolves=0u;
        let surface=resolve_ray_hit_filtered(mixed,r.direction_max.xyz,vec2f(0.0));
        guard_bad |= u32(resolves!=0u || mixed.t!=1.25 || scene_hit_alpha(mixed,vec2f(0.0))!=1.0
            || surface.material.perceptual_roughness!=0.5 || surface.material.roughness!=0.25
            || surface.material.reflectance!=0.5 || any(surface.material.base_color!=vec3f(0.2,0.4,0.1))
            || any(surface.world_position!=(r.origin_min.xyz+r.direction_max.xyz*1.25))
            || any(surface.previous_frame_world_position!=surface.world_position+vec3f(3.0,0.0,0.0)));
    } else {guard_bad |= u32(!same(mixed,hit));}
    var unknown:RayIntersection;
    unknown.kind=2u;
    let foreign=scene_hit_from_triangle(unknown);
    let ray_bad=u32(foreign.kind!=SCENE_HIT_INVALID || scene_hit_is_miss(foreign) || scene_hit_is_triangle(foreign));
    output[id.x]=Output(vec4u(mismatch,hit.kind,hit.triangle.slot,hit.triangle.primitive),
        vec4u(bitcast<u32>(hit.t),u32(hit.triangle.front_face),guard_bad,ray_bad));
}
