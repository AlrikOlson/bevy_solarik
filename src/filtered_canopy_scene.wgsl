#define_import_path bevy_solarik::filtered_canopy_scene
#import bevy_solarik::filtered_canopy_transport::ForestEvent
#import bevy_solarik::filtered_canopy_region::{forest_box, forest_trace_region}
@group(0) @binding(28) var<storage,read> forest_scene:array<vec4u>;
struct ForestSceneEvent { event:ForestEvent, previous_position:vec3f }
// Invocation-local state, initialized by the ray's caller. Never an arena address
// or floating-origin position. Successive queries get independent cell clocks.
var<private> forest_query_seed:vec2u;
var<private> forest_query_index:u32;
fn forest_set_seed(seed:vec2u) {forest_query_seed=seed;forest_query_index=0u;}
fn forest_next_seed()->vec2u {
    let seed=forest_query_seed+vec2u(forest_query_index*0x9e3779b9u,0u);
    forest_query_index+=1u;return seed;
}
fn forest_scene_matrix(address:u32)->mat4x4f {
    return mat4x4f(bitcast<vec4f>(forest_scene[address]),bitcast<vec4f>(forest_scene[address+1u]),
        bitcast<vec4f>(forest_scene[address+2u]),bitcast<vec4f>(forest_scene[address+3u]));
}
fn forest_primary_seed(pixel:vec2u,width:u32)->vec2u {
    return vec2u(pixel.x+pixel.y*width,forest_scene[1].x);
}
// Explicit controls: 1 isolates guides, 2 conditions primary radiance.
fn forest_background_guides()->bool {return forest_scene[1].y!=0u;}
fn forest_conditioning_enabled()->bool {return forest_scene[1].y==2u;}
fn forest_trace_scene(origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,seed:vec2u)->ForestSceneEvent {
    return forest_trace_scene_impl(origin,direction,minimum,maximum,seed,false);
}
fn forest_trace_primary(origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,seed:vec2u)->ForestSceneEvent {
    return forest_trace_scene_impl(origin,direction,minimum,maximum,seed,true);
}
fn forest_scene_event(local:ForestEvent,address:u32,direction:vec3f)->ForestSceneEvent {
    var empty:ForestSceneEvent;
    if local.normal_hit.w==0.0 { return empty; }
    let world=forest_scene_matrix(address);
    let previous=forest_scene_matrix(address+8u);
    var result:ForestSceneEvent;
    var event=local;
    result.previous_position=(previous*vec4f(event.position_distance.xyz,1.0)).xyz;
    event.position_distance=vec4f((world*vec4f(event.position_distance.xyz,1.0)).xyz,event.position_distance.w);
    event.normal_hit=vec4f(normalize((world*vec4f(event.normal_hit.xyz,0.0)).xyz),1.0);
    if dot(event.normal_hit.xyz,direction)>0.0 {event.normal_hit=vec4f(-event.normal_hit.xyz,1.0);}
    result.event=event;
    return result;
}
fn forest_trace_scene_impl(origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,seed:vec2u,primary:bool)->ForestSceneEvent {
    var result:ForestSceneEvent;
    let header=forest_scene[0];var node=0u;var closest=maximum;
    loop {
        if node>=header.x {break;}
        let low=forest_scene[header.y+node*2u];let high=forest_scene[header.y+node*2u+1u];
        if !forest_box(bitcast<vec3f>(low.xyz),bitcast<vec3f>(high.xyz),origin,direction,minimum,closest) {
            node=low.w;continue;
        }
        if high.w==0xffffffffu {node+=1u;continue;}
        let address=header.z+high.w*14u;
        if !primary && forest_scene[address+13u].w==0u {node=low.w;continue;}
        let inverse=forest_scene_matrix(address+4u);
        let region_origin=(inverse*vec4f(origin,1.0)).xyz;
        let region_direction=(inverse*vec4f(direction,0.0)).xyz;
        var event=forest_trace_region(forest_scene[address+12u].w,region_origin,region_direction,minimum,closest,seed);
        if event.normal_hit.w>0.0 && event.position_distance.w<closest {
            result=forest_scene_event(event,address,direction);
            closest=event.position_distance.w;
        }
        node=low.w;
    }
    return result;
}
