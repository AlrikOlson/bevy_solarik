// These synthetic scenes use ordinary, nondirectional emission.
fn emitted_radiance(m: Material, outgoing: vec3f) -> vec3f { return m.emissive; }
struct Material { base_color:vec3<f32>, roughness:f32, metallic:f32, reflectance:f32, emissive:vec3<f32> }
struct ResolvedRayHitFull { world_position:vec3<f32>, world_normal:vec3<f32>, geometric_world_normal:vec3<f32>, material:Material, uv:vec2<f32> }
struct RawMaterial { flags:u32 }
struct Ray { kind:u32, instance_index:u32, t:f32 }
struct Bsdf { wi:vec3<f32>, throughput:vec3<f32>, pdf:f32 }
struct Light { radiance:vec3<f32>, inverse_pdf:f32, wi:vec3<f32>, brdf_rays_can_hit:bool, solid_angle_pdf:f32 }
struct ResolvedLightSample { world_position:vec4<f32> }
struct LightSample { resolved_light_sample:ResolvedLightSample }
struct TransmittedLightContribution { light:Light, continuation_probability:f32 }
const RAY_T_MIN=0.001; const RAY_T_MAX=10000.0;
const RAY_QUERY_INTERSECTION_NONE=0u; const RAY_FLAG_NONE=0u;
const MIRROR_ROUGHNESS_THRESHOLD=0.001;
const MATERIAL_FLAG_ALPHA_BLEND=1u; const MATERIAL_FLAG_DIFFUSE_BLEND=2u;
@group(0) @binding(0) var<storage,read> config:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;
// Binding 0 doubles as a nonempty light list.

var<private> materials:array<RawMaterial,2>;
var<private> material_ids:array<u32,2>;
var<private> hit_z:f32;
fn rand_f(rng:ptr<function,u32>)->f32 {
 *rng = *rng * 1664525u + 1013904223u;
 var h=*rng; h=((h>>16u)^h)*0x45d9f3bu; h=((h>>16u)^h)*0x45d9f3bu; h=(h>>16u)^h;
 return f32(h>>8u)/16777216.0;
}
fn resolve_material_alpha(m:RawMaterial,uv:vec2<f32>)->f32 { return config[0].a; }
fn resolve_ray_hit_full(r:Ray)->ResolvedRayHitFull {
 let pane=r.instance_index==0u;
 let emission=select(select(1.0,0.0,config[1].y==2.0&&hit_z>0.0),0.0,pane);
 let m=Material(select(vec3(0.0),config[0].rgb,pane),1.0,0.0,0.5,vec3(emission));
 return ResolvedRayHitFull(vec3(0.0,0.0,hit_z),vec3(0.0,0.0,1.0),vec3(0.0,0.0,1.0),m,vec2(0.0));
}
fn trace_glass_ray(o:vec3<f32>,d:vec3<f32>,lo:f32,hi:f32)->Ray {
 if d.z<0.0 {
  if config[1].y!=2.0 { return Ray(0u,0u,0.0); }
  hit_z=-1.0; return Ray(1u,1u,1.0);
 }
 hit_z=ceil(o.z+lo);
 let pane=hit_z<=config[1].x;
 return Ray(1u,select(1u,0u,pane),hit_z-o.z);
}
fn trace_ray(o:vec3<f32>,d:vec3<f32>,lo:f32,hi:f32,flag:u32)->Ray {
 hit_z=config[1].x+1.0; return Ray(1u,1u,hit_z-o.z);
}
fn sample_sky(d:vec3<f32>)->vec3<f32> { return vec3(0.0); }
fn analytic_light_radiance(o:vec3<f32>,d:vec3<f32>,limit:f32,owned:bool,scatter:vec3<f32>)->vec3<f32> { return vec3(0.0); }
fn evaluate_brdf(wo:vec3<f32>,wi:vec3<f32>,n:vec3<f32>,m:Material)->vec3<f32> {
 return select(vec3(0.0),vec3(1.0),all(m.base_color==vec3(1.0)));
}
fn evaluate_brdf_pdf(wo:vec3<f32>,wi:vec3<f32>,n:vec3<f32>,m:Material)->f32 { return 1.0; }
fn evaluate_and_sample_brdf(wo:vec3<f32>,n:vec3<f32>,m:Material,rng:ptr<function,u32>)->Bsdf {
 return Bsdf(vec3(0.0,0.0,1.0),evaluate_brdf(wo,n,n,m),1.0);
}
fn generate_random_light_sample(rng:ptr<function,u32>)->LightSample { return LightSample(ResolvedLightSample(vec4(0.0,0.0,config[1].x+1.0,1.0))); }
fn calculate_resolved_light_contribution(s:ResolvedLightSample,p:vec3<f32>,n:vec3<f32>)->Light {
 return Light(vec3(select(1.0,0.0,config[1].y==2.0)),1.0,vec3(0.0,0.0,1.0),true,1.0);
}
fn trace_light_visibility(o:vec3<f32>,p:vec4<f32>)->f32 { return 1.0; }
fn sample_random_light_transmitted(p:vec3<f32>,n:vec3<f32>,g:vec3<f32>,rng:ptr<function,u32>)->TransmittedLightContribution {
 let a=config[0].a;
 var t=vec3(1.0-a)+a*(12.0/13.0)*config[0].rgb;
 var q=1.0-a/13.0;
 if config[1].y==1.0 { t=vec3(1.0-a); q=1.0-a; }
 t=pow(t,vec3(config[1].x)); q=pow(q,config[1].x);
 if config[1].x>32.0 || config[1].y==2.0 { t=vec3(0.0); }
 return TransmittedLightContribution(Light(t,1.0,vec3(0.0,0.0,1.0),true,1.0),q);
}
fn random_emissive_light_solid_angle_pdf(h:ResolvedRayHitFull,p:vec3<f32>)->f32 { return 1.0; }
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
 materials[0].flags=select(MATERIAL_FLAG_ALPHA_BLEND,MATERIAL_FLAG_DIFFUSE_BLEND,config[1].y==1.0);
 material_ids[1]=1u;
 var rng=id.x+1u;
 let m=Material(vec3(1.0),1.0,0.0,0.5,vec3(0.0));
 let initial=ResolvedRayHitFull(vec3(0.0),vec3(0.0,0.0,1.0),vec3(0.0,0.0,1.0),m,vec2(0.0));
 output[id.x]=vec4(shade_surface_path(initial,vec3(0.0,0.0,1.0),&rng),0.0);
}
