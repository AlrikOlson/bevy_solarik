#define_import_path bevy_solarik::ring_math
struct RingSegment { q:vec3<f32>, reference:f32, bounds:vec4<f32> }
// Distances relative to the ring-plane crossing preserve metre thickness at
// astronomical observer distances. Inner-cylinder interval is excluded later.
fn cylinder(p:vec2<f32>,d:vec2<f32>,radius:f32)->vec2<f32> {
 let a=dot(d,d);
 if a<1e-20 {return select(vec2(1.0,-1.0),vec2(-1e30,1e30),dot(p,p)<radius*radius);}
 let c=-dot(p,d)/a; let closest=p+c*d;
 let h2=(radius*radius-dot(closest,closest))/a;
 if h2<0.0 {return vec2(1.0,-1.0);}
 return vec2(c-sqrt(h2),c+sqrt(h2));
}
fn ring_segment(p:vec3<f32>,d:vec3<f32>,inner:f32,outer:f32,half_height:f32,end:f32)->RingSegment {
 var reference=0.0;var q=p;
 if abs(d.y)>1e-12 {reference=-p.y/d.y;q=p+d*reference;q.y=0.0;}
 var plane=vec2(-1e30,1e30);
 if abs(d.y)>1e-12 {let h=half_height/abs(d.y);plane=vec2(-h,h);}
 else if abs(p.y)>half_height {plane=vec2(1.0,-1.0);}
 let radial=cylinder(q.xz,d.xz,outer);
 let hole=cylinder(q.xz,d.xz,inner);
 let low=max(max(plane.x,radial.x),-reference);
 let high=min(min(plane.y,radial.y),end-reference);
 return RingSegment(q,reference,vec4(low,high,hole));
}
// Exact integral of a homogeneous cell's once-scattered source. Incoming
// optical depth varies linearly from a to b; x is camera-ray optical depth.
fn ring_source(x:f32,a:f32,b:f32)->f32 {
 if x<=0.0 {return 0.0;}
 let rate=1.0+(b-a)/x;
 let u=rate*x;
 if abs(u)<0.2 {return exp(-a)*x*(1.0+u*(-0.5+u*(1.0/6.0+u*(-1.0/24.0+u*(1.0/120.0+u*(-1.0/720.0+u/5040.0))))));}
 return max(0.0,(exp(-a)-exp(-x-b))/rate);
}
fn ring_phase(cos_scatter:f32,g:f32)->f32 {
 return (1.0-g*g)/pow(max(1.0+g*g-2.0*g*cos_scatter,1e-6),1.5);
}
