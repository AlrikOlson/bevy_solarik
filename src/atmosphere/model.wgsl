#define_import_path bevy_solarik::atmosphere_model

// Original implementation of the transport equations in Hillaire 2020,
// https://sebh.github.io/publications/egsr2020.pdf, and Earth density profiles
// in Bruneton 2017, https://ebruneton.github.io/precomputed_atmospheric_scattering/.
// Lengths are km, coefficients km^-1, phase sr^-1, source illuminance lux,
// and integrated radiance cd/m². No exposure or empirical sky gain here.
const ATM_PI: f32 = 3.141592653589793;
const GROUND_RADIUS: f32 = 6360.0;
const TOP_RADIUS: f32 = 6460.0;

struct AtmosphereParams {
    medium: vec4<f32>, // Rayleigh, Mie, ozone multipliers; Mie g
    ground: vec4<f32>, // albedo; multiple-scattering enable
    sun: vec4<f32>, // unit direction, TOA lux
    moon: vec4<f32>, // unit direction, TOA lux
    observer: vec4<f32>, // height metres, aerial distance metres, stars, samples
}

struct MediumSample {
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
}

fn observer_position(p: AtmosphereParams) -> vec3<f32> {
    return vec3(0.0, GROUND_RADIUS + p.observer.x * 0.001, 0.0);
}

fn sample_medium(p: AtmosphereParams, altitude: f32) -> MediumSample {
    let h = max(altitude, 0.0);
    let r = vec3(0.005802, 0.013558, 0.0331) * p.medium.x * exp(-h / 8.0);
    let m = vec3(0.003996 * p.medium.y * exp(-h / 1.2));
    let absorption = vec3(0.00065, 0.001881, 0.000085) * p.medium.z
        * max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    return MediumSample(r, m, r + m, r + m / 0.9 + absorption);
}

// Integrates exp(-extinction * s) over one complete segment, including the
// vacuum limit. The Taylor branch avoids cancellation at small optical depth.
fn segment_integral(extinction: vec3<f32>, distance: f32) -> vec3<f32> {
    let x = max(extinction, vec3(0.0)) * distance;
    let series = distance * (vec3(1.0) - 0.5 * x + x * x / 6.0);
    let quotient = (vec3(1.0) - exp(-x)) / max(extinction, vec3(1e-20));
    return select(quotient, series, x < vec3(0.001));
}

fn ray_hits_ground(origin: vec3<f32>, direction: vec3<f32>) -> bool {
    let r = length(origin);
    let b = dot(origin, direction);
    return b < 0.0 && b*b >= (r-GROUND_RADIUS)*(r+GROUND_RADIUS);
}

// xy = distance to the first boundary, whether it is opaque ground.
fn atmosphere_boundary(origin: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    let r = length(origin);
    let b = dot(origin, direction);
    if ray_hits_ground(origin, direction) {
        let c = max(0.0, (r-GROUND_RADIUS)*(r+GROUND_RADIUS));
        return vec2(c / max(-b + sqrt(max(b*b-c, 0.0)), 1e-8), 1.0);
    }
    return vec2(max(0.0, -b + sqrt(max(b*b+(TOP_RADIUS-r)*(TOP_RADIUS+r), 0.0))), 0.0);
}

fn optical_transmittance(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, distance: f32, samples: u32) -> vec3<f32> {
    let dt = distance / f32(max(samples, 1u));
    var optical = vec3(0.0);
    for (var i = 0u; i < samples; i++) {
        let position = origin + direction * ((f32(i)+0.5)*dt);
        optical += sample_medium(p, length(position)-GROUND_RADIUS).extinction * dt;
    }
    return exp(-optical);
}

fn transmittance_to_space(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, samples: u32) -> vec3<f32> {
    let boundary = atmosphere_boundary(origin, direction);
    if boundary.y > 0.0 { return vec3(0.0); }
    return optical_transmittance(p, origin, direction, boundary.x, samples);
}

fn phase_rayleigh(mu: f32) -> f32 {
    return 3.0 * (1.0 + mu*mu) / (16.0 * ATM_PI);
}

fn phase_mie(mu: f32, g: f32) -> f32 {
    let g2 = g*g;
    return 3.0 * (1.0-g2) * (1.0+mu*mu)
        / (8.0 * ATM_PI * (2.0+g2) * pow(max(1.0+g2-2.0*g*mu, 1e-5), 1.5));
}

// Bruneton's distance/rho mapping: detail at the tangent instead of spending
// half the transmittance texture on rays occluded by the planet.
fn transmittance_uv(radius: f32, mu: f32, dimensions: vec2<u32>) -> vec2<f32> {
    let r = clamp(radius, GROUND_RADIUS, TOP_RADIUS);
    let h = sqrt((TOP_RADIUS-GROUND_RADIUS)*(TOP_RADIUS+GROUND_RADIUS));
    let rho = sqrt(max(0.0, (r-GROUND_RADIUS)*(r+GROUND_RADIUS)));
    let distance = max(0.0, -r*mu + sqrt(max(0.0, r*r*(mu*mu-1.0)+TOP_RADIUS*TOP_RADIUS)));
    let x = clamp(vec2((distance-(TOP_RADIUS-r))/max(rho+h-(TOP_RADIUS-r), 1e-6), rho/h), vec2(0.0), vec2(1.0));
    let size = vec2<f32>(dimensions);
    return (vec2(0.5) + x*(size-1.0))/size;
}

fn transmittance_position(uv: vec2<f32>) -> vec2<f32> {
    let h = sqrt((TOP_RADIUS-GROUND_RADIUS)*(TOP_RADIUS+GROUND_RADIUS));
    let rho = h*uv.y;
    let r = sqrt(rho*rho+GROUND_RADIUS*GROUND_RADIUS);
    let d = mix(TOP_RADIUS-r, rho+h, uv.x);
    let mu = select(clamp(((TOP_RADIUS-r)*(TOP_RADIUS+r)-d*d)/max(2.0*r*d, 1e-6), -1.0, 1.0), 1.0, d <= 1e-6);
    return vec2(r, mu);
}

fn sample_transmittance(tex: texture_2d<f32>, filtering: sampler, position: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    if ray_hits_ground(position, direction) { return vec3(0.0); }
    let r = length(position);
    let uv = transmittance_uv(r, dot(position/r, direction), textureDimensions(tex));
    return clamp(textureSampleLevel(tex, filtering, uv, 0.0).rgb, vec3(0.0), vec3(1.0));
}

fn sample_multiple(tex: texture_2d<f32>, filtering: sampler, position: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let r = length(position);
    let unit = clamp(vec2(dot(position/r, direction)*0.5+0.5, (r-GROUND_RADIUS)/100.0), vec2(0.0), vec2(1.0));
    let size = vec2<f32>(textureDimensions(tex));
    return max(textureSampleLevel(tex, filtering, (0.5+unit*(size-1.0))/size, 0.0).rgb, vec3(0.0));
}

struct AtmosphereTransport {
    radiance: vec3<f32>,
    transmittance: vec3<f32>,
}

fn local_inscattering(p: AtmosphereParams, position: vec3<f32>, direction: vec3<f32>, medium: MediumSample,
    trans: texture_2d<f32>, multi: texture_2d<f32>, filtering: sampler) -> vec3<f32> {
    var result = vec3(0.0);
    let sources = array<vec4<f32>, 2>(p.sun, p.moon);
    for (var i = 0u; i < 2u; i++) {
        let source = sources[i];
        if source.w <= 0.0 { continue; }
        let mu = clamp(dot(direction, source.xyz), -1.0, 1.0);
        let phase = medium.rayleigh * phase_rayleigh(mu) + medium.mie * phase_mie(mu, p.medium.w);
        let direct = sample_transmittance(trans, filtering, position, source.xyz) * phase;
        let ms = sample_multiple(multi, filtering, position, source.xyz) * medium.scattering * p.ground.w;
        result += source.w * (direct+ms);
    }
    return result;
}

fn integrate_atmosphere(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, distance: f32, samples: u32,
    trans: texture_2d<f32>, multi: texture_2d<f32>, filtering: sampler, ground: bool) -> AtmosphereTransport {
    let boundary = atmosphere_boundary(origin, direction);
    let length_max = min(distance, boundary.x);
    var result = AtmosphereTransport(vec3(0.0), vec3(1.0));
    for (var i = 0u; i < samples; i++) {
        // Quadratic intervals improve near-observer integration and still cover
        // the entire segment, including its final endpoint.
        let a = f32(i)/f32(samples);
        let b = f32(i+1u)/f32(samples);
        let t0 = a*a*length_max;
        let t1 = b*b*length_max;
        let dt = t1-t0;
        let position = origin+direction*((t0+t1)*0.5);
        let m = sample_medium(p, length(position)-GROUND_RADIUS);
        let source = local_inscattering(p, position, direction, m, trans, multi, filtering);
        result.radiance += result.transmittance * source * segment_integral(m.extinction, dt);
        result.transmittance *= exp(-m.extinction*dt);
    }
    if ground && boundary.y > 0.0 && distance >= boundary.x {
        let position = origin+direction*boundary.x;
        let normal = normalize(position);
        let sources = array<vec4<f32>, 2>(p.sun, p.moon);
        for (var i = 0u; i < 2u; i++) {
            let source = sources[i];
            let direct = sample_transmittance(trans, filtering, normal*(GROUND_RADIUS+0.001), source.xyz)
                * max(dot(normal, source.xyz), 0.0);
            result.radiance += result.transmittance * p.ground.rgb * source.w * direct / ATM_PI;
        }
    }
    return result;
}

fn horizon_elevation(p: AtmosphereParams) -> f32 {
    let r = GROUND_RADIUS+p.observer.x*0.001;
    return -acos(clamp(GROUND_RADIUS/r, 0.0, 1.0));
}

fn sky_direction(p: AtmosphereParams, uv: vec2<f32>) -> vec3<f32> {
    let horizon = horizon_elevation(p);
    let t = uv.y*2.0-1.0;
    let span = select(ATM_PI*0.5+horizon, ATM_PI*0.5-horizon, t >= 0.0);
    let elevation = horizon+sign(t)*t*t*span;
    let azimuth = (uv.x-0.5)*2.0*ATM_PI;
    return vec3(cos(elevation)*sin(azimuth), sin(elevation), cos(elevation)*cos(azimuth));
}

fn sky_uv(p: AtmosphereParams, direction: vec3<f32>, size: vec2<u32>) -> vec2<f32> {
    let elevation = asin(clamp(direction.y, -1.0, 1.0));
    let horizon = horizon_elevation(p);
    let delta = elevation-horizon;
    let span = select(ATM_PI*0.5+horizon, ATM_PI*0.5-horizon, delta >= 0.0);
    let v = 0.5+0.5*sign(delta)*sqrt(abs(delta)/span);
    return vec2(atan2(direction.x, direction.z)/(2.0*ATM_PI)+0.5, (0.5+v*(f32(size.y)-1.0))/f32(size.y));
}

fn cube_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3(1.0, -uv.y, -uv.x); }
        case 1u: { d = vec3(-1.0, -uv.y, uv.x); }
        case 2u: { d = vec3(uv.x, 1.0, uv.y); }
        case 3u: { d = vec3(uv.x, -1.0, -uv.y); }
        case 4u: { d = vec3(uv.x, -uv.y, 1.0); }
        default: { d = vec3(-uv.x, -uv.y, -1.0); }
    }
    // Bevy and Solarik use left-handed cube textures in a right-handed world.
    return normalize(vec3(d.xy, -d.z));
}

// Stable procedural star catalogue on equal-area spherical cells. Each source
// carries photometric flux (zero-magnitude star ~2.54e-6 lux); a normalized
// Gaussian footprint resolves that flux at the view/cubemap's pixel size.
fn star_hash(cell: vec2<u32>) -> u32 {
    var h = cell.x*1973u + cell.y*9277u + 89173u;
    h = (h ^ (h >> 16u))*2246822519u;
    h = (h ^ (h >> 13u))*3266489917u;
    return h ^ (h >> 16u);
}

fn star_radiance(direction: vec3<f32>, pixel_angle: f32) -> vec3<f32> {
    let uv = vec2(atan2(direction.x, direction.z)/(2.0*ATM_PI)+0.5, direction.y*0.5+0.5);
    let cell = vec2<i32>(floor(uv*vec2(256.0, 128.0)));
    let sigma = max(pixel_angle*0.6, 0.00012);
    var radiance = vec3(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let cy = cell.y+y;
            if cy < 0 || cy >= 128 { continue; }
            let c = vec2<u32>(u32((cell.x+x+256)%256), u32(cy));
            let h = star_hash(c);
            if h % 1000u >= 15u { continue; }
            let a = (f32((h >> 10u)&1023u)+0.5)/1024.0;
            let b = (f32((h >> 20u)&1023u)+0.5)/1024.0;
            let sy = (f32(c.y)+b)/128.0*2.0-1.0;
            let azimuth = ((f32(c.x)+a)/256.0-0.5)*2.0*ATM_PI;
            let r = sqrt(max(0.0, 1.0-sy*sy));
            let source = vec3(r*sin(azimuth), sy, r*cos(azimuth));
            let delta = direction-source;
            let distance2 = dot(delta, delta);
            if distance2 > 25.0*sigma*sigma { continue; }
            let magnitude = 1.0+5.0*f32(h&1023u)/1023.0;
            let flux = 2.54e-6*pow(10.0, -0.4*magnitude);
            let brightness = flux*exp(-distance2/(2.0*sigma*sigma))/(2.0*ATM_PI*sigma*sigma);
            radiance += vec3(brightness);
        }
    }
    return radiance;
}
