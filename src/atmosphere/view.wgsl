#import bevy_render::view::View
#import bevy_solarik::atmosphere_model::{AtmosphereParams, ATM_PI, observer_position, atmosphere_boundary, integrate_atmosphere, sample_transmittance, sky_uv, star_radiance}

@group(0) @binding(0) var<uniform> p: AtmosphereParams;
@group(0) @binding(1) var filtering: sampler;
@group(0) @binding(2) var trans: texture_2d<f32>;
@group(0) @binding(3) var multiple: texture_2d<f32>;
@group(0) @binding(4) var<uniform> view: View;

fn view_direction(uv: vec2<f32>) -> vec3<f32> {
    let h = view.view_from_clip * vec4(uv*vec2(2.0, -2.0)+vec2(-1.0, 1.0), 1.0, 1.0);
    return normalize((view.world_from_view * vec4(h.xyz/h.w, 0.0)).xyz);
}

#ifdef AERIAL
@group(0) @binding(5) var aerial_scattering: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var aerial_transmittance: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(aerial_scattering);
    if any(id >= size) { return; }
    let direction = view_direction((vec2<f32>(id.xy)+0.5)/vec2<f32>(size.xy));
    // Quadratic distance slices place the first nonzero slice at ~2 metres.
    // z=0 is exactly vacuum, so there is no near-camera fog discontinuity.
    let z = f32(id.z)/f32(size.z-1u);
    let distance = z*z*p.observer.y*0.001;
    let result = integrate_atmosphere(p, observer_position(p), direction, distance, 8u, trans, multiple, filtering, false);
    // Store pre-exposed scattering to retain night precision in half floats.
    textureStore(aerial_scattering, id, vec4(min(result.radiance*view.exposure, vec3(65000.0)), 1.0));
    textureStore(aerial_transmittance, id, vec4(result.transmittance, 1.0));
}
#endif

#ifdef COMPOSITE
@group(0) @binding(5) var sky: texture_2d<f32>;
@group(0) @binding(6) var aerial_scattering: texture_3d<f32>;
@group(0) @binding(7) var aerial_transmittance: texture_3d<f32>;
@group(0) @binding(8) var depth: texture_depth_2d;
@group(0) @binding(9) var output: texture_storage_2d<rgba16float, read_write>;

fn disk_radiance(direction: vec3<f32>, source: vec4<f32>, radius: f32, pixel_angle: f32) -> vec3<f32> {
    let angle = atan2(length(cross(direction, source.xyz)), dot(direction, source.xyz));
    let w = max(pixel_angle*0.5, 1e-6);
    let coverage = 1.0-smoothstep(radius-w, radius+w, angle);
    if coverage <= 0.0 || source.w <= 0.0 { return vec3(0.0); }
    let attenuation = sample_transmittance(trans, filtering, observer_position(p), direction);
    // Projected solid angle gives E = integral L cos(theta) dOmega exactly.
    return attenuation * coverage * source.w / (ATM_PI*sin(radius)*sin(radius));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) { return; }
    let uv = (vec2<f32>(id.xy)+0.5)/vec2<f32>(size);
    let direction = view_direction(uv);
    let depth_size = textureDimensions(depth);
    let pixel_depth = textureLoad(depth, min(vec2<u32>(uv*vec2<f32>(depth_size)), depth_size-1u), 0);
#ifdef BACKGROUND
    if pixel_depth != 0.0 { return; }
#else
    if pixel_depth == 0.0 { return; }
#endif
    var color: vec3<f32>;
    if pixel_depth == 0.0 {
        color = textureSampleLevel(sky, filtering, sky_uv(p, direction, textureDimensions(sky)), 0.0).rgb;
        let dx = view_direction(uv+vec2(1.0/f32(size.x), 0.0));
        let dy = view_direction(uv+vec2(0.0, 1.0/f32(size.y)));
        let pixel_angle = max(length(dx-direction), length(dy-direction));
        color += star_radiance(direction, pixel_angle) * p.observer.z
            * sample_transmittance(trans, filtering, observer_position(p), direction);
        color += disk_radiance(direction, p.sun, 0.004675, pixel_angle);
        color += disk_radiance(direction, p.moon, 0.00452, pixel_angle);
        color *= view.exposure;
    } else {
        let h = view.view_from_clip * vec4(uv*vec2(2.0,-2.0)+vec2(-1.0,1.0), pixel_depth, 1.0);
        let distance = length(h.xyz/h.w);
        let z = sqrt(clamp(distance/p.observer.y, 0.0, 1.0));
        let aerial_size = vec3<f32>(textureDimensions(aerial_scattering));
        let coordinates = vec3(clamp(uv, 0.5/aerial_size.xy, 1.0-0.5/aerial_size.xy),
            (0.5+z*(aerial_size.z-1.0))/aerial_size.z);
        let scattering = textureSampleLevel(aerial_scattering, filtering, coordinates, 0.0).rgb;
        let attenuation = textureSampleLevel(aerial_transmittance, filtering, coordinates, 0.0).rgb;
        color = textureLoad(output, id.xy).rgb * attenuation + scattering;
    }
    // The view target is half float. This guards its representable limit;
    // the unexposed atmosphere and directional-light radiometry are unclamped.
    textureStore(output, id.xy, vec4(clamp(color, vec3(0.0), vec3(65000.0)), 1.0));
}
#endif
