import bevy_solarik::lommel_math::lommel_gbuffer;
import bevy_pbr::render::pbr_fragment::pbr_input_from_standard_material;
import bevy_pbr::render::pbr_functions::alpha_discard;
@if((PREPASS_PIPELINE)) import bevy_pbr::prepass::io::VertexOutput;
@if((PREPASS_PIPELINE)) import bevy_pbr::prepass::io::FragmentOutput;
@if((PREPASS_PIPELINE)) import bevy_pbr::deferred::functions::deferred_output;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::forward_io::VertexOutput;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::forward_io::FragmentOutput;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::pbr_functions::apply_pbr_lighting;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::pbr_functions::main_pass_post_lighting_processing;

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> enabled:vec4<f32>;
@fragment
fn fragment(in:VertexOutput,@builtin(front_facing) front:bool)->FragmentOutput {
    var pbr=pbr_input_from_standard_material(in,front);
    pbr.material.base_color=alpha_discard(pbr.material.flags,pbr.material.alpha_cutoff,pbr.material.base_color);
@if(PREPASS_PIPELINE) {
    var out=deferred_output(in,pbr);
    out.deferred=lommel_gbuffer(out.deferred,enabled.x);
    return out;
}
@if(!(PREPASS_PIPELINE)) {
    var out:FragmentOutput;
    out.color=main_pass_post_lighting_processing(pbr,apply_pbr_lighting(pbr));
    return out;
}
}
