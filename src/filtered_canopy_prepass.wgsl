#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View
#import bevy_pbr::prepass_bindings::PreviousViewUniforms
#import bevy_pbr::pbr_deferred_types::pack_24bit_normal_and_flags
#import bevy_pbr::utils::octahedral_encode
#import bevy_solarik::filtered_canopy_scene::{forest_trace_primary,forest_primary_seed,forest_background_guides}
@group(1) @binding(0) var<uniform> view:View;
@group(1) @binding(1) var<uniform> previous_view:PreviousViewUniforms;
@group(1) @binding(2) var opaque_depth:texture_depth_2d;
struct Output {
    @builtin(frag_depth) depth:f32,
    @location(0) gbuffer:vec4u,
    @location(1) motion:vec2f,
    @location(2) lighting:u32,
#ifdef WRITE_NORMAL
    @location(3) normal:vec4f,
#endif
}
fn unproject(ndc:vec2f,depth:f32)->vec3f {
    let p=view.world_from_clip*vec4f(ndc,depth,1.0);return p.xyz/p.w;
}
@fragment
fn fragment(input:FullscreenVertexOutput)->Output {
    if forest_background_guides() {discard;}
    let pixel=vec2u(input.position.xy-view.main_pass_viewport.xy);
    let ndc=((vec2f(pixel)+0.5)/view.main_pass_viewport.zw-0.5)*vec2f(2.0,-2.0);
    let near=unproject(ndc,1.0);
    let direction=normalize(unproject(ndc,0.0000001)-near);
    let depth=textureLoad(opaque_depth,vec2i(input.position.xy),0);
    var maximum=1e30;
    if depth>0.0 {maximum=distance(near,unproject(ndc,depth));}
    let sample=forest_trace_primary(near,direction,0.0,maximum,forest_primary_seed(pixel,u32(view.main_pass_viewport.z)));
    if sample.event.normal_hit.w==0.0 {discard;}
    let event=sample.event;
    var normal=event.normal_hit.xyz;
    if dot(normal,direction)>0.0 {normal=-normal;}
    let clip=view.clip_from_world*vec4f(event.position_distance.xyz,1.0);
    let current=view.unjittered_clip_from_world*vec4f(event.position_distance.xyz,1.0);
    let previous=previous_view.clip_from_world*vec4f(sample.previous_position,1.0);
    var output:Output;
    output.depth=clip.z/clip.w;
    // bit5 is the finite-source surface tag; ordinary Bevy flags retain bits0..2.
    output.gbuffer=vec4u(pack4x8unorm(vec4f(pow(event.color_coverage.xyz,vec3f(1.0/2.2)),pow(event.surface.x,0.25))),
        0u,pack4x8unorm(vec4f(event.surface.z,event.surface.y,0.0,event.surface.w)),
        pack_24bit_normal_and_flags(octahedral_encode(normal),0x24u));
    output.motion=(current.xy/current.w-previous.xy/previous.w)*vec2f(0.5,-0.5);
    output.lighting=1u;
#ifdef WRITE_NORMAL
    output.normal=vec4f(normal*0.5+0.5,1.0);
#endif
    return output;
}
