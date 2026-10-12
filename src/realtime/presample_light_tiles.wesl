import bevy_pbr::render::rgb9e5::vec3_to_rgb9e5_;
import bevy_pbr::render::rgb9e5::rgb9e5_to_vec3_;
import bevy_render::utils::octahedral_encode;
import bevy_render::utils::octahedral_decode;
import bevy_render::view::View;
import bevy_solarik::scene::sampling::generate_random_light_sample;
import bevy_solarik::scene::sampling::LightSample;
import bevy_solarik::scene::sampling::ResolvedLightSample;
import bevy_solarik::scene::sampling::LIGHT_SAMPLE_DIRECTIONAL;
import bevy_solarik::scene::sampling::LIGHT_SAMPLE_EMISSIVE_MESH;
import bevy_solarik::scene::sampling::LIGHT_SAMPLE_LOCAL;
import bevy_solarik::realtime::realtime_bindings::light_tile_samples;
import bevy_solarik::realtime::realtime_bindings::light_tile_resolved_samples;
import bevy_solarik::realtime::realtime_bindings::view;
import bevy_solarik::realtime::realtime_bindings::constants;
import bevy_solarik::realtime::realtime_bindings::ResolvedLightSamplePacked;

// https://cwyman.org/papers/hpg21_rearchitectingReSTIR.pdf

enable wgpu_ray_query;



@compute @workgroup_size(1024, 1, 1)
fn presample_light_tiles(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) sample_index: u32) {
    let tile_id = workgroup_id.x;
    var rng = (tile_id * 5782582u) + sample_index + constants.frame_index;

    let sample = generate_random_light_sample(&rng);

    let i = (tile_id * 1024u) + sample_index;
    light_tile_samples[i] = sample.light_sample;
    light_tile_resolved_samples[i] = pack_resolved_light_sample(sample.resolved_light_sample);
}

// The kind rides in two signs: a negative inverse_pdf is a directional light
// (upstream), a positive range is a point/spot light (whose range is always
// positive; emissive meshes store -1).
fn pack_resolved_light_sample(sample: ResolvedLightSample) -> ResolvedLightSamplePacked {
    let local = sample.world_position.w == LIGHT_SAMPLE_LOCAL;
    return ResolvedLightSamplePacked(
        sample.world_position.x,
        sample.world_position.y,
        sample.world_position.z,
        pack2x16unorm(octahedral_encode(sample.world_normal)),
        vec3_to_rgb9e5_(log2(sample.radiance * view.exposure + 1.0)),
        sample.inverse_pdf * select(1.0, -1.0, sample.world_position.w == LIGHT_SAMPLE_DIRECTIONAL),
        pack2x16unorm(octahedral_encode(select(vec3(0.0, 0.0, 1.0), sample.spot_direction, local || sample.range == -2.0))),
        pack2x16float(sample.spot_cos),
        select(select(-1.0, -2.0, sample.range == -2.0), max(sample.range, 1e-4), local),
    );
}

fn unpack_resolved_light_sample(packed: ResolvedLightSamplePacked, exposure: f32) -> ResolvedLightSample {
    let local = packed.range > 0.0;
    let kind = select(select(LIGHT_SAMPLE_EMISSIVE_MESH, LIGHT_SAMPLE_LOCAL, local), LIGHT_SAMPLE_DIRECTIONAL, packed.inverse_pdf < 0.0);
    return ResolvedLightSample(
        vec4(packed.world_position_x, packed.world_position_y, packed.world_position_z, kind),
        octahedral_decode(unpack2x16unorm(packed.world_normal)),
        (exp2(rgb9e5_to_vec3_(packed.radiance)) - 1.0) / exposure,
        abs(packed.inverse_pdf),
        octahedral_decode(unpack2x16unorm(packed.spot_direction)),
        unpack2x16float(packed.spot_cos),
        select(0.0, packed.range, local || packed.range == -2.0),
    );
}
