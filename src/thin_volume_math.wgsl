#define_import_path bevy_solarik::thin_volume_math
// An unnormalized anisotropic Gaussian integrated along a finite ray segment.
// q = inverse_axes * (ray_origin-centre), v = inverse_axes * unit_direction.
fn volume_erf(x:f32)->f32 {
 let a=abs(x); let t=1.0/(1.0+0.3275911*a);
 let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;
 return sign(x)*(1.0-p*exp(-a*a));
}
fn gaussian_segment(q:vec3<f32>,v:vec3<f32>,end:f32)->f32 {
 let a=dot(v,v); if a<=0.0 || end<=0.0 {return 0.0;}
 let t=-dot(q,v)/a;
 let closest=q+v*t;
 let perpendicular=dot(closest,closest);
 // Six sigma perpendicular truncation bounds absolute relative peak error.
 if perpendicular>36.0 {return 0.0;}
 let root=sqrt(0.5*a);
 let lo=clamp(-t*root,-6.0,6.0);
 let hi=clamp((end-t)*root,-6.0,6.0);
 return max(0.0,exp(-0.5*perpendicular)*1.253314137/sqrt(a)*(volume_erf(hi)-volume_erf(lo)));
}
// Flux-preserving Gaussian pixel reconstruction. The half-pixel kernel prevents
// unresolved volume cores from disappearing between pixel centres. This is a
// numerical image filter, not a change to the physical density distribution.
fn volume_pixel_scale(inverse_sigma:f32,pixel_width:f32)->f32 {
 return inverseSqrt(1.0+pow(0.5*pixel_width*inverse_sigma,2.0));
}
