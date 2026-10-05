#import bevy_solarik::gaussian_math::gaussian_gbuffer
#import bevy_pbr::{pbr_fragment::pbr_input_from_standard_material, pbr_functions::alpha_discard}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{prepass_io::{VertexOutput, FragmentOutput}, pbr_deferred_functions::deferred_output}
#else
#import bevy_pbr::{forward_io::{VertexOutput, FragmentOutput}, pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing}}
#endif
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> parameters: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var mask: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var mask_sampler: sampler;
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, front);
    let weight = parameters.z*textureSample(mask, mask_sampler, in.uv).a;
    pbr.material.reflectance = mix(pbr.material.reflectance, vec3(parameters.x), weight);
    pbr.material.perceptual_roughness = mix(pbr.material.perceptual_roughness, parameters.y, weight);
    pbr.material.base_color = alpha_discard(pbr.material, pbr.material.base_color);
#ifdef PREPASS_PIPELINE
    var out = deferred_output(in, pbr);
    out.deferred = gaussian_gbuffer(out.deferred, weight);
    return out;
#else
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(pbr, apply_pbr_lighting(pbr));
    return out;
#endif
}
