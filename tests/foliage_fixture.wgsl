// Diffuse-only numerical fixture: normal incidence and F0=0 ensure the
// sampler never chooses GGX. Specular stubs isolate the production diffuse
// lobe; full Bevy shader composition is exercised by scene captures.
const PI: f32 = 3.141592653589793;
const MIRROR_ROUGHNESS_THRESHOLD: f32 = 0.001;
struct ResolvedMaterial { base_color: vec3<f32>, emissive: vec3<f32>, reflectance: f32, perceptual_roughness: f32, roughness: f32, metallic: f32, diffuse_transmission: f32 }
@group(0) @binding(0) var<storage> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var brdf_dfg_lut: texture_2d<f32>;
@group(0) @binding(3) var brdf_dfg_lut_sampler: sampler;
fn luminance(v: vec3<f32>) -> f32 { return dot(v, vec3(0.2126, 0.7152, 0.0722)); }
fn calculate_F0(c:vec3<f32>, m:f32, r:vec3<f32>)->vec3<f32>{return mix(0.16*r*r,c,m);}
fn calculate_diffuse_color(c:vec3<f32>,m:f32,t:f32,s:f32)->vec3<f32>{return c*(1.0-m);}
fn D_GGX(a:f32,b:f32)->f32{return 0.0;}
fn V_SmithGGXCorrelated(a:f32,b:f32,c:f32)->f32{return 0.0;}
fn specular_multiscatter(a:f32,b:f32,c:vec3<f32>,d:vec3<f32>,e:vec2<f32>,f:f32)->vec3<f32>{return vec3(0.0);}
fn ggx_vndf_pdf(a:vec3<f32>,b:vec3<f32>,c:f32)->f32{return 0.0;}
fn sample_ggx_vndf(a:vec3<f32>,b:f32,r:ptr<function,u32>)->vec3<f32>{return vec3(0.0,0.0,1.0);}
fn ggx_vndf_sample_invalid(a:vec3<f32>)->bool{return false;}
fn orthonormalize(n:vec3<f32>)->mat3x3<f32>{return mat3x3(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),n);}
fn rand_f(r:ptr<function,u32>)->f32 {
    *r = *r * 1664525u + 1013904223u;
    return f32(*r >> 8u) / 16777216.0;
}
fn sample_cosine_hemisphere(n:vec3<f32>,r:ptr<function,u32>)->vec3<f32>{
    let u=rand_f(r); let phi=2.0*PI*rand_f(r);
    return vec3(sqrt(u)*cos(phi),sqrt(u)*sin(phi),sqrt(1.0-u))*n.z;
}
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    var rng = id.x * 747796405u + 2891336453u;
    let n=vec3(0.0,0.0,1.0);
    let m=ResolvedMaterial(vec3(0.2,0.5,0.8),vec3(0.0),0.0,1.0,1.0,0.0,inputs[0].x);
    let s=evaluate_and_sample_brdf(n,n,m,&rng);
    output[id.x*3u]=vec4(s.wi,0.0);
    output[id.x*3u+1u]=vec4(s.throughput,s.pdf);
    output[id.x*3u+2u]=vec4(evaluate_diffuse_brdf(n,n,n,m)+evaluate_diffuse_brdf(n,-n,n,m),evaluate_brdf_pdf(n,s.wi,n,m));
}

