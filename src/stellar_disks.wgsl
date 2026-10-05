#import bevy_render::view::View
#import bevy_solarik::atmosphere_model::disk_per_lux
struct Disks { direction:array<vec4<f32>,2>, irradiance:array<vec4<f32>,2> }
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var depth:texture_depth_2d;
@group(0) @binding(2) var output:texture_storage_2d<rgba16float,read_write>;
@group(0) @binding(3) var<uniform> disks:Disks;
fn ray(uv:vec2<f32>)->vec3<f32> {
 let h=view.view_from_clip*vec4(uv*vec2(2.0,-2.0)+vec2(-1.0,1.0),1.0,1.0);
 return normalize((view.world_from_view*vec4(h.xyz/h.w,0.0)).xyz);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let size=textureDimensions(output); if any(id.xy>=size) {return;}
 let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(size);
 let depth_size=textureDimensions(depth);
 if textureLoad(depth,min(vec2<u32>(uv*vec2<f32>(depth_size)),depth_size-1u),0)!=0.0 {return;}
 let direction=ray(uv);
 let pixel=max(length(ray(uv+vec2(1.0/f32(size.x),0.0))-direction),length(ray(uv+vec2(0.0,1.0/f32(size.y)))-direction));
 var rgb=vec3(0.0);
 for(var i=0u;i<2u;i++) {
  let source=disks.direction[i]; if source.w<=0.0 {continue;}
  let angle=atan2(length(cross(direction,source.xyz)),dot(direction,source.xyz));
  rgb+=disks.irradiance[i].rgb*disk_per_lux(angle,source.w,pixel);
 }
 let previous=textureLoad(output,id.xy);
 textureStore(output,id.xy,vec4(min(previous.rgb+rgb*view.exposure,vec3(65000.0)),previous.a));
}
