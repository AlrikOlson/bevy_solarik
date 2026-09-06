@group(0) @binding(0) var<storage,read> config: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output: array<vec4<f32>>;
const RAY_T_MIN = 0.001;
const INITIAL_SAMPLES = 8u;
const WORLD_CACHE_DIRECT_LIGHT_SAMPLE_COUNT = 8u;
const NULL_LIGHT_ID = 0xffffffffu;
struct LightSample { light_id:u32, seed:u32 }
struct ResolvedLightSample { radiance:vec3<f32>, world_position:vec4<f32> }
struct LightContribution { radiance:vec3<f32>, inverse_pdf:f32, wi:vec3<f32> }
struct Reservoir { sample:LightSample, confidence_weight:f32, unbiased_contribution_weight:f32 }
struct ReservoirContribution { radiance:vec3<f32>, target_function:f32, wi:vec3<f32> }
struct ReservoirMergeResult { merged_reservoir:Reservoir, selected_sample_radiance:vec3<f32>, wi:vec3<f32>, selected_light_world_position:vec4<f32> }
struct View { exposure:f32 }
var<private> view = View(1.0);
var<private> light_sources:array<u32,1>;
var<private> light_tile_samples:array<LightSample,1>;
var<private> light_tile_resolved_samples:array<u32,1>;
fn rand_range_u(n:u32,rng:ptr<function,u32>)->u32 { return 0u; }
fn rand_f(rng:ptr<function,u32>)->f32 { *rng += 1u; return 0.25; }
fn luminance(v:vec3<f32>)->f32 { return dot(v,vec3(0.2126,0.7152,0.0722)); }
fn resolve_light_sample(s:LightSample,l:u32)->ResolvedLightSample { return ResolvedLightSample(vec3(1.0),vec4(0.0,0.0,2.0,1.0)); }
fn unpack_resolved_light_sample(s:u32,e:f32)->ResolvedLightSample { return resolve_light_sample(LightSample(0u,0u),0u); }
fn calculate_resolved_light_contribution(s:ResolvedLightSample,p:vec3<f32>,n:vec3<f32>)->LightContribution { return LightContribution(s.radiance,2.0,vec3(0.0,0.0,1.0)); }
fn trace_light_visibility(p:vec3<f32>,l:vec4<f32>)->f32 { return select(0.0,1.0,any(config[0].rgb>vec3(0.0))); }
fn trace_light_transmission(p:vec3<f32>,l:vec4<f32>)->vec4<f32> { return vec4(config[0].rgb,1.0); }
@compute @workgroup_size(1)
fn probe() {
    var rng=0u;
    let p=vec3(0.0); let n=vec3(0.0,0.0,1.0);
    var r=generate_initial_reservoir(p,n,vec3(1.0),vec2(0u),&rng);
    var merged=merge_reservoirs(r,p,n,vec3(1.0),r,p,n,vec3(1.0),&rng);
    for(var i=0u;i<u32(config[0].a);i++) {
        output[0]=vec4(shade_di_reservoir(merged,p,n),0.0);
        r=merged.merged_reservoir;
        r.confidence_weight=min(r.confidence_weight,20.0);
        merged=merge_reservoirs(r,p,n,vec3(1.0),r,p,n,vec3(1.0),&rng);
    }
    output[0]=vec4(shade_di_reservoir(merged,p,n),0.0);
    output[1]=vec4(sample_random_light_ris(p,n,vec2(0u),&rng),0.0);
    output[2]=vec4(merged.merged_reservoir.unbiased_contribution_weight,0.0,0.0,0.0);
}
