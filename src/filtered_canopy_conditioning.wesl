import bevy_solarik::filtered_canopy_transport::forest_optical_projected;
import bevy_solarik::filtered_canopy_transport::forest_trace_projected_optical;
import bevy_solarik::filtered_canopy_region::ForestSupport;
import bevy_solarik::filtered_canopy_region::forest_regions;
import bevy_solarik::filtered_canopy_region::forest_load_support;
import bevy_solarik::filtered_canopy_region::forest_sample;
import bevy_solarik::filtered_canopy_region::forest_box;
import bevy_solarik::filtered_canopy_scene::ForestSceneEvent;
import bevy_solarik::filtered_canopy_scene::forest_scene;
import bevy_solarik::filtered_canopy_scene::forest_scene_matrix;
import bevy_solarik::filtered_canopy_scene::forest_scene_event;
import bevy_solarik::filtered_canopy_scene::forest_trace_primary;

// Coverage-conditioned first-event transport; see the independent analytic
// controls in space-sim. The proposal preserves joint source marks and normals.

struct ForestConditioned {
    event:ForestSceneEvent, weights:vec2f, coverage:f32, proposal_ratio:f32, fallback:u32,
}
struct ForestIntegral {
    optical:f32, coverage_sum:f32, selected:vec4u, selected_optical:f32, selected_key:f32,
    positive_supports:u32,
}
// Stable expm1/log1p series near zero; the omitted terms at 0.01 are
// smaller than f32 rounding. See PBRT4 A.4.2/A.4.5 and WGSL 15.7.4.
const FOREST_THIN_OPTICAL:f32=0.01;
fn forest_coverage(optical:f32)->f32 {
    if optical<FOREST_THIN_OPTICAL {
        return optical*(1.0+optical*(-0.5+optical*(1.0/6.0+optical*(-1.0/24.0+optical/120.0))));
    }
    return 1.0-exp(-optical);
}
fn forest_conditional_depth(optical:f32,random:f32)->f32 {
    if optical<=0.0 { return 0.0; }
    let probability=min(random,0.99999994)*forest_coverage(optical);
    var optical_target=-log(1.0-probability);
    if probability<FOREST_THIN_OPTICAL {
        optical_target=probability*(1.0+probability*(0.5+probability*(1.0/3.0+probability*(0.25+probability/5.0))));
    }
    // A conditioned sample belongs to the half-open optical interval even
    // when the inverse CDF rounds to its upper endpoint.
    return min(optical_target,bitcast<f32>(bitcast<u32>(optical)-1u));
}
fn forest_support_optical(support:ForestSupport,origin:vec3f,direction:vec3f,minimum:f32,maximum:f32)->f32 {
    return forest_optical_projected(support.cell,support.ground,support.plane,support.supported,
        origin,direction,minimum,maximum);
}
fn forest_integrate_leaf(prior:ForestIntegral,selection:vec4u,origin:vec3f,direction:vec3f,
                          minimum:f32,maximum:f32,seed:vec2u,choose:bool)->ForestIntegral {
    var result=prior;
    let count=max(1u,forest_regions[selection.y+7u].w);
    for(var support_index=0u;support_index<count;support_index+=1u) {
        let support=forest_load_support(selection.x,selection.y,support_index);
        let optical=forest_support_optical(support,origin,direction,minimum,maximum);
        result.optical+=optical;
        if !choose || optical<=0.0 { continue; }
        result.positive_supports+=1u;
        let coverage=forest_coverage(optical);
        result.coverage_sum+=coverage;
        // Independent exponential clocks select with probability q_i/sum(q).
        // Unlike an ordered reservoir, logical clocks survive BVH repacking.
        let draw=forest_sample(seed^vec2u(0x163f72a9u,0x9e34b157u),support.identity).x;
        let key=-log(draw)/coverage;
        if key<result.selected_key {
            result.selected=vec4u(selection.xy,support_index,selection.w);
            result.selected_optical=optical;
            result.selected_key=key;
        }
    }
    return result;
}
fn forest_integrate_region(prior:ForestIntegral,base:u32,instance:u32,origin:vec3f,direction:vec3f,
                            minimum:f32,maximum:f32,seed:vec2u,choose:bool)->ForestIntegral {
    var result=prior;
    let header=forest_regions[base];
    var node=0u;
    loop {
        if node>=header.x { break; }
        let low=forest_regions[base+header.y+node*2u];
        let high=forest_regions[base+header.y+node*2u+1u];
        if !forest_box(bitcast<vec3f>(low.xyz),bitcast<vec3f>(high.xyz),origin,direction,minimum,maximum) {
            node=low.w;continue;
        }
        if high.w==0xffffffffu { node+=1u;continue; }
        let address=base+header.z+high.w*8u;
        result=forest_integrate_leaf(result,vec4u(base,address,0u,instance),origin,direction,
            minimum,maximum,seed,choose);
        node=low.w;
    }
    return result;
}
fn forest_integrate_scene(origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,
                           seed:vec2u,choose:bool)->ForestIntegral {
    var result:ForestIntegral;
    result.selected_key=3.402823e38;
    let header=forest_scene[0];
    var node=0u;
    loop {
        if node>=header.x { break; }
        let low=forest_scene[header.y+node*2u];
        let high=forest_scene[header.y+node*2u+1u];
        if !forest_box(bitcast<vec3f>(low.xyz),bitcast<vec3f>(high.xyz),origin,direction,minimum,maximum) {
            node=low.w;continue;
        }
        if high.w==0xffffffffu { node+=1u;continue; }
        let address=header.z+high.w*14u;
        let inverse=forest_scene_matrix(address+4u);
        let local_origin=(inverse*vec4f(origin,1.0)).xyz;
        let local_direction=(inverse*vec4f(direction,0.0)).xyz;
        result=forest_integrate_region(result,forest_scene[address+12u].w,address,
            local_origin,local_direction,minimum,maximum,seed,choose);
        node=low.w;
    }
    return result;
}
fn forest_conditioned_primary(origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,seed:vec2u)->ForestConditioned {
    let integral=forest_integrate_scene(origin,direction,minimum,maximum,seed,true);
    var result:ForestConditioned;
    result.weights=vec2f(0.0,exp(-integral.optical));
    result.coverage=forest_coverage(integral.optical);
    if integral.coverage_sum<=0.0 { return result; }
    result.proposal_ratio=integral.coverage_sum/result.coverage;
    // Selecting many separately opaque cells has high importance-weight
    // variance. This input-only decision keeps the ordinary estimator unbiased.
    if result.proposal_ratio>2.0 {
        result.event=forest_trace_primary(origin,direction,minimum,maximum,seed);
        let hit=result.event.event.normal_hit.w>0.0;
        result.weights=select(vec2f(0.0,1.0),vec2f(1.0,0.0),hit);
        result.fallback=1u;
        return result;
    }
    let selected=integral.selected;
    let inverse=forest_scene_matrix(selected.w+4u);
    let local_origin=(inverse*vec4f(origin,1.0)).xyz;
    let local_direction=(inverse*vec4f(direction,0.0)).xyz;
    let support=forest_load_support(selected.x,selected.y,selected.z);
    let xi=forest_sample(seed^vec2u(0xb724ac31u,0x51d739e5u),support.identity);
    let optical_target=forest_conditional_depth(integral.selected_optical,xi.x);
    let event=forest_trace_projected_optical(support.cell,support.ground,support.plane,support.supported,
        local_origin,local_direction,minimum,maximum,optical_target,xi.y);
    result.event=forest_scene_event(event,selected.w,direction);
    // With one positive support, every other prefix contribution is zero.
    // Counting supports avoids inferring this from a rounded proposal ratio.
    if integral.positive_supports==1u {
        result.weights.x=integral.coverage_sum;
        return result;
    }
    let prefix=forest_integrate_scene(origin,direction,minimum,event.position_distance.w,seed,false);
    let own_prefix=forest_support_optical(support,local_origin,local_direction,minimum,event.position_distance.w);
    result.weights.x=integral.coverage_sum*exp(-max(0.0,prefix.optical-own_prefix));
    return result;
}
