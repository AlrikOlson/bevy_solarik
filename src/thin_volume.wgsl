#import bevy_render::view::View
#import bevy_solarik::thin_volume_math::{gaussian_segment, volume_pixel_scale}
struct Kernel {centre:vec4<f32>,x:vec4<f32>,y:vec4<f32>,z:vec4<f32>,emission:vec4<f32>}
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var depth:texture_depth_2d;
@group(0) @binding(2) var output:texture_storage_2d<rgba16float,read_write>;
@group(0) @binding(3) var<storage,read> kernels:array<Kernel>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let size=textureDimensions(output); if any(id.xy>=size) {return;}
 let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(size);
 let ndc=uv*vec2(2.0,-2.0)+vec2(-1.0,1.0);
 let h=view.view_from_clip*vec4(ndc,1.0,1.0);
 let direction=normalize((view.world_from_view*vec4(h.xyz/h.w,0.0)).xyz);
 let neighbour=view.view_from_clip*vec4(ndc+vec2(0.0,2.0/f32(size.y)),1.0,1.0);
 let pixel_angle=length(normalize(neighbour.xyz/neighbour.w)-normalize(h.xyz/h.w));
 let ds=textureDimensions(depth);
 let d=textureLoad(depth,min(vec2<u32>(uv*vec2<f32>(ds)),ds-1u),0);
 var end=1e30;
 if d>0.0 {let p=view.view_from_clip*vec4(ndc,d,1.0);end=length(p.xyz/p.w);}
 var rgb=vec3(0.0);
 for(var i=0u;i<arrayLength(&kernels);i++) {
  let k=kernels[i];
  let width=max(0.0,dot(k.centre.xyz,direction))*pixel_angle;
  let reconstruction=vec3(volume_pixel_scale(length(k.x.xyz),width),volume_pixel_scale(length(k.y.xyz),width),volume_pixel_scale(length(k.z.xyz),width));
  let q=vec3(dot(-k.centre.xyz,k.x.xyz),dot(-k.centre.xyz,k.y.xyz),dot(-k.centre.xyz,k.z.xyz))*reconstruction;
  let v=vec3(dot(direction,k.x.xyz),dot(direction,k.y.xyz),dot(direction,k.z.xyz))*reconstruction;
  rgb+=k.emission.rgb*reconstruction.x*reconstruction.y*reconstruction.z*gaussian_segment(q,v,end);
 }
 let previous=textureLoad(output,id.xy);
 textureStore(output,id.xy,vec4(min(previous.rgb+rgb*view.exposure,vec3(65000.0)),previous.a));
}
