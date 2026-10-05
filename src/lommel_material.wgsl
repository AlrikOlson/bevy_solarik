#import bevy_solarik::lommel_math::lommel_gbuffer
#import bevy_pbr::{pbr_fragment::pbr_input_from_standard_material, pbr_functions::alpha_discard}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{prepass_io::{VertexOutput, FragmentOutput}, pbr_deferred_functions::deferred_output}
#else
#import bevy_pbr::{forward_io::{VertexOutput, FragmentOutput}, pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing}}
#endif
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> enabled:vec4<f32>;
@fragment
fn fragment(in:VertexOutput,@builtin(front_facing) front:bool)->FragmentOutput {
    var pbr=pbr_input_from_standard_material(in,front);
    pbr.material.base_color=alpha_discard(pbr.material,pbr.material.base_color);
#ifdef PREPASS_PIPELINE
    var out=deferred_output(in,pbr);
    out.deferred=lommel_gbuffer(out.deferred,enabled.x);
    return out;
#else
    var out:FragmentOutput;
    out.color=main_pass_post_lighting_processing(pbr,apply_pbr_lighting(pbr));
    return out;
#endif
}
