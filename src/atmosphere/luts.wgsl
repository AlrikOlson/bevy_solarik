#import bevy_solarik::atmosphere_model::{AtmosphereParams, ATM_PI, GROUND_RADIUS, TOP_RADIUS, observer_position, sample_medium, segment_integral, atmosphere_boundary, optical_transmittance, transmittance_position, sample_transmittance, integrate_atmosphere, sky_direction, sky_uv, cube_direction, star_radiance}

@group(0) @binding(0) var<uniform> p: AtmosphereParams;
@group(0) @binding(1) var filtering: sampler;

#ifdef TRANSMITTANCE
@group(0) @binding(2) var output: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) { return; }
    let rm = transmittance_position(vec2<f32>(id.xy)/vec2<f32>(size-1u));
    let origin = vec3(0.0, rm.x, 0.0);
    let dir = vec3(sqrt(max(0.0, 1.0-rm.y*rm.y)), rm.y, 0.0);
    let b = rm.x*rm.y;
    let distance = max(0.0, -b+sqrt(max(0.0, b*b+(TOP_RADIUS-rm.x)*(TOP_RADIUS+rm.x))));
    textureStore(output, id.xy, vec4(optical_transmittance(p, origin, dir, distance, 128u), 1.0));
}
#endif

#ifdef MULTIPLE
@group(0) @binding(2) var trans: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) { return; }
    let uv = vec2<f32>(id.xy)/vec2<f32>(size-1u);
    let origin = vec3(0.0, GROUND_RADIUS+clamp(uv.y*100.0, 0.001, 99.999), 0.0);
    let mu = uv.x*2.0-1.0;
    let sun = vec3(sqrt(max(0.0, 1.0-mu*mu)), mu, 0.0);
    var l2 = vec3(0.0);
    var feedback = vec3(0.0);
    // Equal-area Fibonacci directions; stable and deterministic across frames.
    for (var k = 0u; k < 64u; k++) {
        let y = 1.0-2.0*(f32(k)+0.5)/64.0;
        let angle = f32(k)*2.39996323;
        let r = sqrt(max(0.0, 1.0-y*y));
        let dir = vec3(r*cos(angle), y, r*sin(angle));
        let boundary = atmosphere_boundary(origin, dir);
        var throughput = vec3(1.0);
        for (var i = 0u; i < 64u; i++) {
            let a = f32(i)/64.0;
            let b = f32(i+1u)/64.0;
            let t0 = a*a*boundary.x;
            let t1 = b*b*boundary.x;
            let dt = t1-t0;
            let position = origin+dir*((t0+t1)*0.5);
            let m = sample_medium(p, length(position)-GROUND_RADIUS);
            let weight = throughput * m.scattering * segment_integral(m.extinction, dt);
            feedback += weight / 64.0;
            l2 += weight * sample_transmittance(trans, filtering, position, sun) / (64.0*4.0*ATM_PI);
            throughput *= exp(-m.extinction*dt);
        }
        if boundary.y > 0.0 {
            let normal = normalize(origin+dir*boundary.x);
            l2 += throughput * p.ground.rgb * sample_transmittance(trans, filtering, normal*(GROUND_RADIUS+0.001), sun)
                * max(dot(normal, sun), 0.0) / (64.0*ATM_PI);
        }
    }
    // Hillaire Eq. 9/10: second order times an infinite geometric series.
    // The denominator floor is a numerical guard at the validated dense limit.
    let multiple = l2 / max(vec3(1.0)-feedback, vec3(0.001));
    textureStore(output, id.xy, vec4(max(multiple, vec3(0.0)), 1.0));
}
#endif

#ifdef SKY_VIEW
@group(0) @binding(2) var trans: texture_2d<f32>;
@group(0) @binding(3) var multiple: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) { return; }
    let uv = vec2((f32(id.x)+0.5)/f32(size.x), f32(id.y)/f32(size.y-1u));
    let direction = sky_direction(p, uv);
    let origin = observer_position(p);
    let transport = integrate_atmosphere(p, origin, direction, 1e6, u32(p.observer.w), trans, multiple, filtering, true);
    let reaches_space = atmosphere_boundary(origin, direction).y == 0.0;
    // Natural background is always present: daylight hides it through radiance,
    // not a time-of-day fade or exposure-dependent gain.
    let background = vec3(0.00012, 0.00022, 0.0004) * p.observer.z;
    let sky = transport.radiance + select(vec3(0.0), transport.transmittance*background, reaches_space);
    textureStore(output, id.xy, vec4(max(sky, vec3(0.0)), 1.0));
}
#endif

#ifdef CUBE
@group(0) @binding(2) var sky: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d_array<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if any(id.xy >= size) || id.z >= 6u { return; }
    let direction = cube_direction(id.z, (vec2<f32>(id.xy)+0.5)/vec2<f32>(size)*2.0-1.0);
    let uv = (vec2<f32>(id.xy)+0.5)/vec2<f32>(size)*2.0-1.0;
    let footprint = 2.0/f32(size.x)/sqrt(1.0+dot(uv, uv));
    let origin = observer_position(p);
    var stars = star_radiance(direction, footprint)*p.observer.z;
    if any(stars > vec3(0.0)) {
        let boundary = atmosphere_boundary(origin, direction);
        if boundary.y > 0.0 { stars = vec3(0.0); }
        else { stars *= optical_transmittance(p, origin, direction, boundary.x, 128u); }
    }
    let radiance = textureSampleLevel(sky, filtering, sky_uv(p, direction, textureDimensions(sky)), 0.0).rgb + stars;
    textureStore(output, id.xy, i32(id.z), vec4(max(radiance, vec3(0.0)), 1.0));
}
#endif
