import bevy_solarik::gaussian_math::gaussian_gbuffer;
import bevy_solarik::lommel_math::lommel_gbuffer;
import bevy_solarik::detail_sampling::DetailCoordinates;
import bevy_solarik::detail_sampling::sample_surface_detail;
import bevy_solarik::detail_sampling::detail_normal;
import bevy_pbr::render::pbr_fragment::pbr_input_from_standard_material;
import bevy_pbr::render::pbr_functions::alpha_discard;
import bevy_pbr::render::mesh_functions::get_local_from_world;
import bevy_pbr::render::mesh_functions::get_world_from_local;
@if((PREPASS_PIPELINE)) import bevy_pbr::prepass::io::VertexOutput;
@if((PREPASS_PIPELINE)) import bevy_pbr::prepass::io::FragmentOutput;
@if((PREPASS_PIPELINE)) import bevy_pbr::deferred::functions::deferred_output;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::forward_io::VertexOutput;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::forward_io::FragmentOutput;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::pbr_functions::apply_pbr_lighting;
@if((!(PREPASS_PIPELINE))) import bevy_pbr::render::pbr_functions::main_pass_post_lighting_processing;


@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> coordinates: DetailCoordinates;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var scan_colour: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var scan_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var scan_detail: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var coverage0: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var cover_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var coverage1: texture_2d<f32>;

@group(constants::MATERIAL_BIND_GROUP) @binding(107) var<uniform> gaussian_parameters: vec4<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var gaussian_mask: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(109) var gaussian_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(110) var<uniform> lommel:vec4<f32>;
// The mesoscale octave: the same layers at tens of metres.
@group(constants::MATERIAL_BIND_GROUP) @binding(111) var meso_colour: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(112) var meso_detail: texture_2d_array<f32>;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, front);
    let weight = gaussian_parameters.z*textureSample(gaussian_mask, gaussian_sampler, in.uv).a;
    pbr.material.reflectance = mix(pbr.material.reflectance, vec3(gaussian_parameters.x), weight);
    pbr.material.perceptual_roughness = mix(pbr.material.perceptual_roughness, gaussian_parameters.y, weight);
    let local_from_world = get_local_from_world(in.instance_index);
    let world_from_local = get_world_from_local(in.instance_index);
    let p = (local_from_world * in.world_position).xyz;
    let n = normalize((local_from_world * vec4f(pbr.N, 0.0)).xyz);
    let footprint = max(length(dpdx(p)), length(dpdy(p)));
    let detail = sample_surface_detail(scan_colour, scan_detail, scan_sampler, meso_colour, meso_detail,
        coordinates, p, n, footprint,
        textureSample(coverage0, cover_sampler, in.uv),
        textureSample(coverage1, cover_sampler, in.uv));
    pbr.material.base_color = alpha_discard(pbr.material.flags, pbr.material.alpha_cutoff, pbr.material.base_color);
    pbr.material.base_color = vec4f(clamp(pbr.material.base_color.rgb * detail.colour, vec3f(0.0), vec3f(1.0)), pbr.material.base_color.a);
    pbr.material.perceptual_roughness = clamp(pbr.material.perceptual_roughness * mix(detail.roughness, 1.0, weight), 0.02, 1.0);
    pbr.N = normalize((world_from_local * vec4f(detail_normal(n, detail.gradient), 0.0)).xyz);
@if(PREPASS_PIPELINE) {
    var out = deferred_output(in, pbr);
    out.deferred = gaussian_gbuffer(out.deferred, weight);
    out.deferred = lommel_gbuffer(out.deferred, lommel.x);
    return out;
}
@if(!(PREPASS_PIPELINE)) {
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(pbr, apply_pbr_lighting(pbr));
    return out;
}
}
