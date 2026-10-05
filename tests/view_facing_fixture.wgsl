// Diffuse-only numerical fixture for shading normals that face away from the
// viewer. Specular stubs isolate the production diffuse lobe.
const PI: f32 = 3.141592653589793;
const MIRROR_ROUGHNESS_THRESHOLD: f32 = 0.001;
const GRAZING: f32 = 0.02;
struct ResolvedMaterial { base_color: vec3<f32>, emissive: vec3<f32>, reflectance: f32, perceptual_roughness: f32, roughness: f32, metallic: f32, diffuse_transmission: f32, gaussian_weight: f32 }
@group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
fn luminance(v: vec3<f32>) -> f32 { return dot(v, vec3(0.2126, 0.7152, 0.0722)); }
fn calculate_F0(c:vec3<f32>, m:f32, r:vec3<f32>)->vec3<f32>{return mix(0.16*r*r,c,m);}
fn calculate_diffuse_color(c:vec3<f32>,m:f32,t:f32,s:f32)->vec3<f32>{return c*(1.0-m);}
fn D_GGX(a:f32,b:f32)->f32{return 0.0;}
fn V_SmithGGXCorrelated(a:f32,b:f32,c:f32)->f32{return 0.0;}
fn specular_multiscatter(a:f32,b:f32,c:vec3<f32>,d:vec3<f32>,e:vec2<f32>,f:f32)->vec3<f32>{return vec3(0.0);}
fn ggx_vndf_pdf(a:vec3<f32>,b:vec3<f32>,c:f32)->f32{return 0.0;}
fn sample_ggx_vndf(a:vec3<f32>,b:f32,r:ptr<function,u32>)->vec3<f32>{return vec3(0.0,0.0,1.0);}
fn ggx_vndf_sample_invalid(a:vec3<f32>)->bool{return false;}
fn orthonormalize(n:vec3<f32>)->mat3x3<f32>{
    let t=normalize(cross(n,select(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),abs(n.x)>0.9)));
    return mat3x3(t,cross(n,t),n);
}
fn rand_f(r:ptr<function,u32>)->f32 {
    *r = *r * 1664525u + 1013904223u;
    return f32(*r >> 8u) / 16777216.0;
}
fn sample_sphere(r:ptr<function,u32>)->vec3<f32> {
    let z=2.0*rand_f(r)-1.0; let phi=2.0*PI*rand_f(r); let s=sqrt(max(0.0,1.0-z*z));
    return vec3(s*cos(phi),s*sin(phi),z);
}
fn sample_cosine_hemisphere(n:vec3<f32>,r:ptr<function,u32>)->vec3<f32>{
    return normalize(n+sample_sphere(r)*0.999);
}
// Per sample: [BRDF with the given normal, its PDF], [BRDF with the normal
// mirrored into the viewer's hemisphere by hand, cosine of the given normal
// to the viewer], [PDF with the mirrored normal, cosine of the mirrored
// normal to the light, sampled PDF, sampled throughput], [green diffuse BRDF
// seen at a cosine of 0.02 and lit along the normal: rough, smooth].
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    var rng = id.x * 747796405u + 2891336453u;
    let wo=vec3(0.0,0.0,1.0);
    let n=sample_sphere(&rng);
    let wi=sample_sphere(&rng);
    let m=ResolvedMaterial(vec3(0.2,0.5,0.8),vec3(0.0),0.5,1.0,1.0,0.0,0.0,0.0);
    let cosine=dot(n,wo);
    let mirrored=select(n,n-2.0*cosine*wo,cosine<0.0);
    let s=evaluate_and_sample_brdf(wo,n,m,&rng);
    let up=vec3(0.0,0.0,1.0);
    let grazing=vec3(sqrt(1.0-GRAZING*GRAZING),0.0,GRAZING);
    let smooth_material=ResolvedMaterial(vec3(0.2,0.5,0.8),vec3(0.0),0.5,0.0,0.0,0.0,0.0,0.0);
    output[id.x*4u+3u]=vec4(evaluate_diffuse_brdf(grazing,up,up,m).g,evaluate_diffuse_brdf(grazing,up,up,smooth_material).g,0.0,0.0);
    output[id.x*4u]=vec4(evaluate_diffuse_brdf(wo,wi,n,m),evaluate_brdf_pdf(wo,wi,n,m));
    output[id.x*4u+1u]=vec4(evaluate_diffuse_brdf(wo,wi,mirrored,m),cosine);
    output[id.x*4u+2u]=vec4(evaluate_brdf_pdf(wo,wi,mirrored,m),dot(mirrored,wi),s.pdf,s.throughput.g);
}
