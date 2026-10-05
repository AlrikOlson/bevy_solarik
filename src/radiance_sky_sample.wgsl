#define_import_path bevy_solarik::radiance_sky_sample
// Manual bilinear interpolation: RGBA32Float need not support hardware filtering.
fn sky_uv(direction:vec3<f32>)->vec2<f32> {
 return vec2(fract(atan2(-direction.z,direction.x)/6.28318530718+1.0),acos(clamp(direction.y,-1.0,1.0))/3.14159265359);
}
fn sky_texel(field:texture_2d<f32>,p:vec2<i32>)->vec3<f32> {
 let n=vec2<i32>(textureDimensions(field));
 return textureLoad(field,vec2((p.x%n.x+n.x)%n.x,clamp(p.y,0,n.y-1)),0).rgb;
}
fn sample_sky(field:texture_2d<f32>,direction:vec3<f32>)->vec3<f32> {
 let p=sky_uv(direction)*vec2<f32>(textureDimensions(field))-0.5;
 let b=vec2<i32>(floor(p));let f=fract(p);
 return mix(mix(sky_texel(field,b),sky_texel(field,b+vec2(1,0)),f.x),mix(sky_texel(field,b+vec2(0,1)),sky_texel(field,b+vec2(1,1)),f.x),f.y);
}
