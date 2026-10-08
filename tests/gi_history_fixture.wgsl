@group(0) @binding(0) var<storage,read> config:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;
const RAY_T_MIN=0.001;
struct Reservoir {
 sample_point_world_position:vec3<f32>, sample_age:f32,
 radiance:vec3<f32>, confidence_weight:f32,
 sample_point_world_normal:vec3<f32>, unbiased_contribution_weight:f32,
#ifdef REGIONAL_HISTORY
 support_radius:f32,
#endif
}
struct ReservoirMergeResult { merged_reservoir:Reservoir, selected_sample_radiance:vec3<f32>, wi:vec3<f32> }
fn rand_f(rng:ptr<function,u32>)->f32 { *rng+=1u; return 0.25; }
fn luminance(v:vec3<f32>)->f32 { return dot(v,vec3(0.2126,0.7152,0.0722)); }
fn trace_point_visibility(p:vec3<f32>,l:vec3<f32>)->f32 { return 1.0; }
fn trace_shadow_transmission_impl(p:vec3<f32>,d:vec3<f32>,t:f32,c:bool)->vec4<f32> { return vec4(config[0].rgb,1.0); }
@compute @workgroup_size(1)
fn probe() {
 var rng=0u;
 let p=vec3(0.0); let n=vec3(0.0,0.0,1.0);
 var r=empty_reservoir();
 r.sample_point_world_position=vec3(0.0,0.0,2.0); r.confidence_weight=1.0;
 r.sample_point_world_normal=-n; r.unbiased_contribution_weight=2.0; r.radiance=vec3(1.0);
 for(var i=0u;i<u32(config[0].a);i++) {
  output[0]=vec4(shade_gi_connection(p,n,r.sample_point_world_position,r.radiance,r.unbiased_contribution_weight),0.0);
  r.confidence_weight=min(r.confidence_weight,8.0);
  // The selected endpoint has already survived several frames. Spatial
  // reuse must preserve its age rather than making it fresh again.
  r.sample_age = 7.0;
  r=merge_reservoirs(r,p,n,vec3(1.0),r,p,n,vec3(1.0),&rng).merged_reservoir;
 }
 output[0]=vec4(shade_gi_connection(p,n,r.sample_point_world_position,r.radiance,r.unbiased_contribution_weight),0.0);
 output[1]=vec4(r.radiance,0.0);
 output[2]=vec4(r.unbiased_contribution_weight,r.sample_age,0.0,0.0);
 // A bright endpoint must die within the production frame bound, even if
 // it wins spatial reuse repeatedly between every temporal step.
 var history=r; history.sample_age=0.0;
 for(var frame=0u;frame<u32(MAX_GI_SAMPLE_AGE);frame++) {
  history=age_temporal_reservoir(history);
  for(var neighbor=0u;neighbor<4u;neighbor++) {
   history=merge_reservoirs(history,p,n,vec3(1.0),history,p,n,vec3(1.0),&rng).merged_reservoir;
  }
 }
 output[2].z=history.confidence_weight;
 output[2].w=length(history.radiance);
#ifdef REGIONAL_HISTORY
 r.support_radius=2.0;
 var other=r; other.sample_point_world_position.x+=0.1; other.support_radius=4.0;
 let merged=merge_reservoirs(r,p,n,vec3(1.0),other,p,n,vec3(1.0),&rng).merged_reservoir;
 output[3]=vec4(merged.support_radius,
   distance(merged.sample_point_world_position,r.sample_point_world_position)+r.support_radius,
   distance(merged.sample_point_world_position,other.sample_point_world_position)+other.support_radius,
   empty_reservoir().support_radius);
#endif
}

