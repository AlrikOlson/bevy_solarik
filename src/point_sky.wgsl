#import bevy_render::view::View
struct Point {position:vec4<f32>,colour:vec4<f32>}
@group(0) @binding(0) var<uniform> view:View;
@group(0) @binding(1) var depth:texture_depth_2d;
@group(0) @binding(2) var output:texture_storage_2d<rgba16float,read_write>;
@group(0) @binding(3) var<storage,read> points:array<Point>;
@group(0) @binding(4) var<storage,read_write> pixels:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> observer:vec4<f32>;
fn add(index:u32,value:f32) {
 if value<=0.0 {return;}
 var old=atomicLoad(&pixels[index]);
 loop {
  let next=bitcast<u32>(min(bitcast<f32>(old)+value,65000.0));
  let result=atomicCompareExchangeWeak(&pixels[index],old,next);
  if result.exchanged {break;}
  old=result.old_value;
 }
}
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id:vec3<u32>) {
 if id.x>=u32(observer.w) {return;}
 let star=points[id.x];let relative=star.position.xyz-select(observer.xyz,vec3(0.0),star.colour.w>0.5);
 let d2=dot(relative,relative);
 if d2<=1e-20 {return;}
 let direction=(view.view_from_world*vec4(relative*inverseSqrt(d2),0.0)).xyz;
 if direction.z>=0.0 {return;}
 let clip=view.clip_from_view*vec4(direction,0.0);
 let dimensions=vec2<f32>(textureDimensions(output));
 let centre=(clip.xy/clip.w*vec2(0.5,-0.5)+0.5)*dimensions;
 if any(centre<vec2(-5.0)) || any(centre>dimensions+5.0) {return;}
 let base=vec2<i32>(floor(centre));
 let omega=4.0*pow(-direction.z,3.0)/(dimensions.x*dimensions.y*view.clip_from_view[0][0]*view.clip_from_view[1][1]);
 let radiance=star.colour.rgb*(star.position.w/d2)*view.exposure/omega;
 // Normalize discrete reconstruction weights; the disk's size is not exaggerated.
 var norm=vec2(0.0);
 for(var a=-4;a<=4;a++) {
  let delta=vec2<f32>(base)+f32(a)+0.5-centre;
  norm+=exp(-delta*delta/(2.0*0.65*0.65));
 }
 for(var y=-4;y<=4;y++) {for(var x=-4;x<=4;x++) {
  let p=base+vec2(x,y);
  if any(p<vec2(0)) || any(p>=vec2<i32>(dimensions)) {continue;}
  if textureLoad(depth,p,0)!=0.0 {continue;}
  let delta=vec2<f32>(p)+0.5-centre;
  let weight=exp(-dot(delta,delta)/(2.0*0.65*0.65))/(norm.x*norm.y);
  let index=u32(p.y)*u32(dimensions.x)+u32(p.x);
  add(3u*index,radiance.r*weight);add(3u*index+1u,radiance.g*weight);add(3u*index+2u,radiance.b*weight);
 }}
}
@compute @workgroup_size(8,8)
fn composite(@builtin(global_invocation_id) id:vec3<u32>) {
 let size=textureDimensions(output);if any(id.xy>=size) {return;}
 let i=3u*(id.y*size.x+id.x);
 let light=vec3(bitcast<f32>(atomicLoad(&pixels[i])),bitcast<f32>(atomicLoad(&pixels[i+1u])),bitcast<f32>(atomicLoad(&pixels[i+2u])));
 let previous=textureLoad(output,id.xy);
 textureStore(output,id.xy,vec4(min(previous.rgb+light,vec3(65000.0)),previous.a));
}
