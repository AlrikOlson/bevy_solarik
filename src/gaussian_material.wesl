import bevy_solarik::gaussian_math::gaussian_gbuffer;
import bevy_pbr::render::pbr_fragment::pbr_input_from_standard_material;
import bevy_pbr::render::pbr_functions::alpha_discard;
@if((PREPASS_PIPELINE)) import bevy_pbr::prepass::io::VertexOutput;
@if((PREPASS_PIPELINE)) import bevy_pbr::prepass::io::FragmentOutput;
@if((PREPASS_PIPELINE)) import bevy_pbr::deferred::functions::deferred_output;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::forward_io::VertexOutput;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::forward_io::FragmentOutput;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::pbr_functions::apply_pbr_lighting;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::pbr_functions::main_pass_post_lighting_processing;

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> parameters: vec4<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var mask: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var mask_sampler: sampler;
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, front);
    let weight = parameters.z*textureSample(mask, mask_sampler, in.uv).a;
    pbr.material.reflectance = mix(pbr.material.reflectance, vec3(parameters.x), weight);
    pbr.material.perceptual_roughness = mix(pbr.material.perceptual_roughness, parameters.y, weight);
    pbr.material.base_color = alpha_discard(pbr.material.flags, pbr.material.alpha_cutoff, pbr.material.base_color);
@if(PREPASS_PIPELINE) {
    var out = deferred_output(in, pbr);
    out.deferred = gaussian_gbuffer(out.deferred, weight);
    return out;
}
@if(!(PREPASS_PIPELINE)) {
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(pbr, apply_pbr_lighting(pbr));
    return out;
}
}
