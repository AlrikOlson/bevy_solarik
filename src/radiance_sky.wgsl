#import bevy_render::view::View
#import bevy_solarik::radiance_sky_sample::sample_sky
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var depth:texture_depth_2d;
@group(0) @binding(2) var output:texture_storage_2d<rgba16float,read_write>;
@group(0) @binding(3) var field:texture_2d<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let size=textureDimensions(output);
 if any(id.xy>=size) {return;}
 if textureLoad(depth,id.xy,0)!=0.0 {return;}
 let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(size);
 let clip=vec4(uv*vec2(2.0,-2.0)+vec2(-1.0,1.0),1.0,1.0);
 let local=view.view_from_clip*clip;
 let direction=normalize((view.world_from_view*vec4(local.xyz/local.w,0.0)).xyz);
 let previous=textureLoad(output,id.xy);
 textureStore(output,id.xy,vec4(previous.rgb+sample_sky(field,direction)*view.exposure,previous.a));
}
