#import bevy_render::view::View
#import bevy_solarik::ring_transport::ring_radiance
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var depth:texture_depth_2d;
@group(0) @binding(2) var output:texture_storage_2d<rgba16float,read_write>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let size=textureDimensions(output);if any(id.xy>=size) {return;}
 let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(size);
 let ndc=uv*vec2(2.0,-2.0)+vec2(-1.0,1.0);
 let h=view.view_from_clip*vec4(ndc,1.0,1.0);
 let direction=normalize((view.world_from_view*vec4(h.xyz/h.w,0.0)).xyz);
 let ds=textureDimensions(depth);let z=textureLoad(depth,min(vec2<u32>(uv*vec2<f32>(ds)),ds-1u),0);
 var end=1e30;if z>0.0 {let at=view.view_from_clip*vec4(ndc,z,1.0);end=length(at.xyz/at.w);}
 // Four stratified subpixel rays integrate real subpixel ringlets, not
 // arbitrary blurring or exaggerated gap widths.
 var integrated=vec4(0.0);
 for(var i=0u;i<4u;i++) {
  let offset=(vec2(f32(i&1u),f32(i>>1u))-0.5)*0.5;
  let hn=view.view_from_clip*vec4(ndc+offset/vec2<f32>(size)*vec2(2.0,-2.0),1.0,1.0);
  let dn=normalize((view.world_from_view*vec4(hn.xyz/hn.w,0.0)).xyz);
  integrated+=ring_radiance(vec3(0.0),dn,end)*0.25;
 }
 let old=textureLoad(output,id.xy);
 textureStore(output,id.xy,vec4(min(old.rgb*integrated.a+integrated.rgb*view.exposure,vec3(65000.0)),old.a));
}
