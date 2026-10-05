#import bevy_solarik::detail_sampling::{DetailCoordinates, sample_surface_detail, detail_normal}
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions::{get_local_from_world, get_world_from_local},
}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{prepass_io::{VertexOutput, FragmentOutput}, pbr_deferred_functions::deferred_output}
#else
#import bevy_pbr::{forward_io::{VertexOutput, FragmentOutput}, pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing}}
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> coordinates: DetailCoordinates;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var scan_colour: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var scan_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var scan_detail: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var coverage0: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var cover_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var coverage1: texture_2d<f32>;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, front);
    let local_from_world = get_local_from_world(in.instance_index);
    let world_from_local = get_world_from_local(in.instance_index);
    let p = (local_from_world * in.world_position).xyz;
    let n = normalize((local_from_world * vec4f(pbr.N, 0.0)).xyz);
    let footprint = max(length(dpdx(p)), length(dpdy(p)));
    let detail = sample_surface_detail(scan_colour, scan_detail, scan_sampler, coordinates,
        p, n, footprint,
        textureSample(coverage0, cover_sampler, in.uv),
        textureSample(coverage1, cover_sampler, in.uv));
    pbr.material.base_color = alpha_discard(pbr.material, pbr.material.base_color);
    pbr.material.base_color = vec4f(clamp(pbr.material.base_color.rgb * detail.colour, vec3f(0.0), vec3f(1.0)), pbr.material.base_color.a);
    pbr.material.perceptual_roughness = clamp(pbr.material.perceptual_roughness * detail.roughness, 0.02, 1.0);
    pbr.N = normalize((world_from_local * vec4f(detail_normal(n, detail.gradient), 0.0)).xyz);
#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr);
#else
    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(pbr, apply_pbr_lighting(pbr));
    return out;
#endif
}
