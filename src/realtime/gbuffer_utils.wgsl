#define_import_path bevy_solarik::gbuffer_utils

#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::rgb9e5::rgb9e5_to_vec3_
#import bevy_pbr::utils::octahedral_decode
#import bevy_render::view::{View, depth_ndc_to_view_z}
#import bevy_solarik::scene_bindings::ResolvedMaterial

struct ResolvedGPixel {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    material: ResolvedMaterial,
}

fn gpixel_resolve(gpixel: vec4<u32>, depth: f32, pixel_id: vec2<u32>, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> ResolvedGPixel {
    let world_position = reconstruct_world_position(pixel_id, depth, view_size, world_from_clip);
    let world_normal = octahedral_decode(unpack_24bit_normal(gpixel.a));

    let base_rough = unpack4x8unorm(gpixel.r);
    let base_color = pow(base_rough.rgb, vec3(2.2));
    let perceptual_roughness = base_rough.a;
    let roughness = perceptual_roughness * perceptual_roughness;
    let props = unpack4x8unorm(gpixel.b);
    let reflectance = props.r;
    let metallic = props.g;
    let emissive = rgb9e5_to_vec3_(gpixel.g);
    let gaussian_weight = select(0.0, props.a, (gpixel.a & 0x08000000u) != 0u);
    let transmission=select(0.0,props.a,(gpixel.a&0x20000000u)!=0u);
    let material = ResolvedMaterial(base_color, emissive, reflectance, perceptual_roughness, roughness, metallic, transmission, vec4(0.0), gaussian_weight, (gpixel.a & 0x10000000u) != 0u);

    return ResolvedGPixel(world_position, world_normal, material);
}

fn reconstruct_world_position(pixel_id: vec2<u32>, depth: f32, view_size: vec2<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel_id) + 0.5) / view_size;
    let xy_ndc = (uv - vec2(0.5)) * vec2(2.0, -2.0);
    let world_pos = world_from_clip * vec4(xy_ndc, depth, 1.0);
    return world_pos.xyz / world_pos.w;
}

// Rasterized vertices have quantized framebuffer XY. Interpolating their depth
// and unprojecting the exact pixel centre can therefore fall below the surface.
// Propagate one subpixel step through the homogeneous quotient, onto the normal.
// A full step covers truncation as well as nearest rounding. The caller omits
// this term for analytic primary intersections, whose depth is not interpolated.
fn raster_reconstruction_error(pixel_id: vec2u, depth: f32, normal: vec3f,
    view_size: vec2f, world_from_clip: mat4x4f, subpixel_step: f32) -> f32 {
    let ndc = ((vec2f(pixel_id) + 0.5) / view_size - 0.5) * vec2f(2.0, -2.0);
    let h = world_from_clip * vec4f(ndc, depth, 1.0);
    let position = h.xyz / h.w;
    let dx = (world_from_clip[0].xyz - position * world_from_clip[0].w) / h.w;
    let dy = (world_from_clip[1].xyz - position * world_from_clip[1].w) / h.w;
    return 2.0 * subpixel_step * (abs(dot(normal, dx)) / view_size.x
        + abs(dot(normal, dy)) / view_size.y);
}

// Reject if tangent plane differs more than 0.3% or angle between normals more than 90 degrees
fn pixel_dissimilar(depth: f32, world_position: vec3<f32>, other_world_position: vec3<f32>, normal: vec3<f32>, other_normal: vec3<f32>, view: View) -> bool {
    // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf#page=45
    let tangent_plane_distance = abs(dot(normal, other_world_position - world_position));
    let view_z = -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);

    return tangent_plane_distance / view_z > 0.003 || dot(normal, other_normal) < 0.0;
}

fn permute_pixel(pixel_id: vec2<u32>, frame_index: u32, view_size: vec2<f32>) -> vec2<u32> {
    let r = frame_index;
    let offset = vec2(r & 3u, (r >> 2u) & 3u);
    var shifted_pixel_id = pixel_id + offset;
    shifted_pixel_id ^= vec2(3u);
    shifted_pixel_id -= offset;
    return min(shifted_pixel_id, vec2<u32>(view_size - 1.0));
}
