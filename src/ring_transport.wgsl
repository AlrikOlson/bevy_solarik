#define_import_path bevy_solarik::ring_transport
#import bevy_solarik::ring_math::{RingSegment,ring_segment,ring_source,ring_phase}
struct RingParameters {
 centre:vec4<f32>, normal:vec4<f32>, axis:vec4<f32>,
 sun:vec4<f32>, irradiance:vec4<f32>, planet:vec4<f32>, profile:vec4<f32>,
}
struct RingData { p:RingParameters, samples:array<vec4<f32>> }
@group(0) @binding(24) var<storage,read> ring_data:RingData;
fn ring_local(v:vec3<f32>)->vec3<f32> {
 let p=ring_data.p;return vec3(dot(v,p.axis.xyz),dot(v,p.normal.xyz),dot(v,cross(p.axis.xyz,p.normal.xyz)));
}
fn ring_sample(radius:f32)->vec4<f32> {
 let p=ring_data.p;let i=(radius-p.planet.w)/max(p.profile.x,1.0);
 if i< -0.5 || i>f32(arrayLength(&ring_data.samples))-0.5 {return vec4(0.0);}
 return ring_data.samples[u32(clamp(round(i),0.0,f32(arrayLength(&ring_data.samples)-1u)))];
}
fn ring_column(local_origin:vec3<f32>,local_direction:vec3<f32>)->f32 {
 let p=ring_data.p;if p.centre.w<=0.0 {return 0.0;}
 let hit=ring_segment(local_origin,local_direction,p.normal.w,p.axis.w,p.centre.w,1e30);
 if hit.bounds.y<=hit.bounds.x {return 0.0;}
 var total=0.0;
 // Explicit inner-hole clipping; a thin ring crossing normally takes one
 // homogeneous sample, grazing radial paths use bounded quadrature.
 for(var part=0u;part<2u;part++) {
  var lo=hit.bounds.x;var hi=hit.bounds.y;
  if hit.bounds.w>hit.bounds.z {if part==0u {hi=min(hi,hit.bounds.z);}else{lo=max(lo,hit.bounds.w);}}
  else if part==1u {continue;}
  if hi<=lo {continue;}
  let count=u32(clamp(ceil((hi-lo)*length(local_direction.xz)/max(p.profile.x,1.0)),1.0,64.0));
  let step=(hi-lo)/f32(count);
  for(var j=0u;j<count;j++) {
   let q=hit.q+local_direction*(lo+(f32(j)+0.5)*step);
   total+=ring_sample(length(q.xz)).x*step/(2.0*p.centre.w);
  }
 }
 return total;
}
fn ring_shadow(origin:vec3<f32>,direction:vec3<f32>)->f32 {
 if ring_data.p.profile.z<0.5 {return 1.0;}
 return exp(-ring_column(ring_local(origin-ring_data.p.centre.xyz),ring_local(direction)));
}
// Fraction of a small uniform stellar disk above an ellipsoid's local limb.
// Straight-limb approximation: stellar angular radius << apparent planet size.
fn ring_planet_visibility(q:vec3<f32>,sun:vec3<f32>)->f32 {
 let p=ring_data.p;if p.irradiance.w<0.5 {return 1.0;}
 let o=q/p.planet.xyz;let d=sun/p.planet.xyz;
 let t=-dot(o,d)/dot(d,d);if t<=0.0 {return 1.0;}
 let distance=length(o+t*d)-1.0;
 let width=max(t*p.sun.w/min(min(p.planet.x,p.planet.y),p.planet.z),1e-9);
 let x=clamp(distance/width,-1.0,1.0);
 return 1.0-(acos(x)-x*sqrt(max(0.0,1.0-x*x)))/3.14159265359;
}
// RGB radiance and background transmittance in physical luminance units.
fn ring_radiance(origin:vec3<f32>,direction:vec3<f32>,end:f32)->vec4<f32> {
 let p=ring_data.p;if p.centre.w<=0.0 {return vec4(0.0,0.0,0.0,1.0);}
 let o=ring_local(origin-p.centre.xyz);let d=ring_local(direction);let sun=ring_local(p.sun.xyz);
 let hit=ring_segment(o,d,p.normal.w,p.axis.w,p.centre.w,end);
 var radiance=vec3(0.0);var transmission=1.0;
 if hit.bounds.y<=hit.bounds.x {return vec4(radiance,transmission);}
 for(var part=0u;part<2u;part++) {
  var lo=hit.bounds.x;var hi=hit.bounds.y;
  if hit.bounds.w>hit.bounds.z {if part==0u {hi=min(hi,hit.bounds.z);}else{lo=max(lo,hit.bounds.w);}}
  else if part==1u {continue;}
  if hi<=lo {continue;}
  let count=u32(clamp(ceil((hi-lo)*length(d.xz)/max(p.profile.x,1.0)),1.0,64.0));
  let step=(hi-lo)/f32(count);
  for(var j=0u;j<count;j++) {
   let a=hit.q+d*(lo+f32(j)*step);let b=hit.q+d*(lo+f32(j+1u)*step);let q=(a+b)*0.5;
   let material=ring_sample(length(q.xz));let opacity=material.x*step/(2.0*p.centre.w);
   if opacity<=0.0 {continue;}
   let incoming_a=ring_column(a,sun);let incoming_b=ring_column(b,sun);
   let source=ring_source(opacity,incoming_a,incoming_b);
   radiance+=transmission*p.irradiance.rgb*material.y*ring_phase(dot(d,sun),material.z)/(4.0*3.14159265359)*source*ring_planet_visibility(q,sun);
   transmission*=exp(-opacity);
   if transmission<1e-7 {break;}
  }
 }
 return vec4(radiance,transmission);
}
