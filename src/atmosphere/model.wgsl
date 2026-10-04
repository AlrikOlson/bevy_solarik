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
    planet: vec4<f32>, // observer body-local km; radius km (zero = local-ground mode)
    shell: vec4<f32>, // outer radius km; cloud coverage, extinction/km, seed
    occluder: vec4<f32>, // body-local centre km, radius km
}

fn ground_radius(p: AtmosphereParams) -> f32 { return select(GROUND_RADIUS, p.planet.w, p.planet.w > 0.0); }
fn top_radius(p: AtmosphereParams) -> f32 { return select(TOP_RADIUS, p.shell.x, p.planet.w > 0.0); }

// Negative exit identifies a miss. Stable enough for the validated <=1e6 km observer.
fn sphere_interval(origin: vec3<f32>, direction: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = dot(origin, direction);
    let perpendicular = origin-direction*b;
    let discriminant = radius*radius-dot(perpendicular, perpendicular);
    if discriminant < 0.0 { return vec2(-1.0); }
    let root = sqrt(discriminant);
    return vec2(-b-root, -b+root);
}

fn source_visibility(p: AtmosphereParams, position: vec3<f32>, direction: vec3<f32>) -> f32 {
    if p.occluder.w <= 0.0 { return 1.0; }
    let offset=p.occluder.xyz-position;
    let distance=length(offset);
    if distance<=p.occluder.w {return 0.0;}
    let axis=offset/distance;
    let d=atan2(length(cross(axis,direction)),dot(axis,direction));
    // Matches this atmosphere's fixed nominal solar angular radius.
    let a=0.004675; let b=asin(clamp(p.occluder.w/distance,0.0,1.0));
    if d>=a+b {return 1.0;}
    if d<=abs(a-b) {return max(0.0,1.0-b*b/(a*a));}
    let x=clamp((d*d+a*a-b*b)/(2.0*d*a),-1.0,1.0);
    let y=clamp((d*d+b*b-a*a)/(2.0*d*b),-1.0,1.0);
    let overlap=a*a*acos(x)+b*b*acos(y)-0.5*sqrt(max(0.0,(-d+a+b)*(d+a-b)*(d-a+b)*(d+a+b)));
    return clamp(1.0-overlap/(ATM_PI*a*a),0.0,1.0);
}

// Smooth body-fixed 3D weather: no longitude chart, poles or per-frame random seed.
// The footprint removes detail that the integration segment cannot resolve.
fn weather_hash(cell: vec3<i32>) -> f32 {
    let u = bitcast<vec3<u32>>(cell);
    var h = u.x*1597334677u ^ u.y*3812015801u ^ u.z*2798796415u;
    h = (h^(h>>16u))*2246822519u;
    return f32(h^(h>>13u))/4294967295.0;
}
fn weather_noise(q: vec3<f32>) -> f32 {
    let cell=vec3<i32>(floor(q));
    let t=fract(q); let f=t*t*(3.0-2.0*t);
    let a=mix(weather_hash(cell),weather_hash(cell+vec3(1,0,0)),f.x);
    let b=mix(weather_hash(cell+vec3(0,1,0)),weather_hash(cell+vec3(1,1,0)),f.x);
    let c=mix(weather_hash(cell+vec3(0,0,1)),weather_hash(cell+vec3(1,0,1)),f.x);
    let d=mix(weather_hash(cell+vec3(0,1,1)),weather_hash(cell+vec3(1,1,1)),f.x);
    return mix(mix(a,b,f.y),mix(c,d,f.y),f.z);
}
// Horizontal cloud cover in 0..1 for a body-fixed unit direction.
fn cloud_cover(p: AtmosphereParams, unit: vec3<f32>, footprint: f32) -> f32 {
    if p.shell.y <= 0.0 { return 0.0; }
    let q = unit*12.0 + vec3(p.shell.w*0.013);
    let warp = vec3(weather_noise(q+vec3(7.0,2.0,9.0)),weather_noise(q+vec3(3.0,8.0,1.0)),weather_noise(q+vec3(1.0,4.0,5.0)));
    let w=q+warp*3.0;
    let large=weather_noise(w)*0.65+weather_noise(w*2.13)*0.25;
    let detail=(weather_noise(w*6.7)-0.5)*0.1/(1.0+footprint*0.03);
    return smoothstep(0.85-0.8*p.shell.y, 1.0-0.8*p.shell.y, large+detail);
}

// Vertical density profile of the shell in 0..1 at altitude h km.
fn cloud_envelope(h: f32) -> f32 {
    return smoothstep(CLOUD_BASE,CLOUD_BASE+1.0,h)*(1.0-smoothstep(CLOUD_TOP-2.0,CLOUD_TOP,h));
}

// Integral of smoothstep(0, 1, t) from 0 to x, continued linearly past 1.
fn smoothstep_integral(x: f32) -> f32 {
    let c = clamp(x, 0.0, 1.0);
    return c*c*c-0.5*c*c*c*c+max(x-1.0, 0.0);
}

// Integral of cloud_envelope from altitude h to the shell top, km. The rising
// and falling edges of the profile do not overlap.
fn cloud_column_above(h: f32) -> f32 {
    let shoulder = CLOUD_TOP-2.0;
    let total = smoothstep_integral(shoulder-CLOUD_BASE)+1.0;
    let rise = smoothstep_integral(min(h, shoulder)-CLOUD_BASE);
    let u = clamp((h-shoulder)*0.5, 0.0, 1.0);
    return max(total-rise-2.0*(u-u*u*u+0.5*u*u*u*u), 0.0);
}

// Cloud droplet scattering: Jendersie & d'Eon (2023), "An Approximate Mie
// Scattering Function for Fog and Cloud Rendering", Eq. 4-7 at droplet
// diameter 20 µm. The fit is (1-w_D) HG(g = 0.99461) + w_D Draine(g_D, alpha).
// The HG lobe is a forward peak a few degrees wide. It is treated as
// unscattered light (delta scaling): cloud extinction is multiplied by w_D and
// the Draine lobe alone is the phase function of the scaled medium.
const CLOUD_DRAINE_WEIGHT: f32 = 0.498159;
const CLOUD_DRAINE_G: f32 = 0.593791;
const CLOUD_DRAINE_ALPHA: f32 = 27.11369;
// Mean cosine of that Draine lobe.
const CLOUD_DRAINE_MEAN_COSINE: f32 = 0.763159;

// Draine (2003) phase function, sr^-1, normalized over the sphere.
fn phase_draine(mu: f32) -> f32 {
    let g = CLOUD_DRAINE_G;
    let a = CLOUD_DRAINE_ALPHA;
    return (1.0-g*g)*(1.0+a*mu*mu)
        / (4.0*ATM_PI*(1.0+a*(1.0+2.0*g*g)/3.0)*pow(max(1.0+g*g-2.0*g*mu, 1e-5), 1.5));
}

// Radiance scattered toward the viewer by the multiply scattered (diffuse)
// field, per unit beam-normal irradiance and unit scaled extinction.
//
// Closed-form conservative Eddington solution for a plane-parallel slab over a
// Lambertian boundary (Shettle & Weinman 1970), after a second delta scaling
// f = g1^2 (Joseph, Wiscombe & Weinman 1976). `above` and `total` are vertical
// optical depths of the scaled medium above the point and of the whole column;
// `mu_light` is the cosine between the scattered light's travel direction and
// the downward normal (negative toward space). This is a physically motivated
// approximation: no horizontal transport, and oblique-sun radiance errors of
// 10-20% against Monte Carlo (see docs/atmosphere.md).
fn cloud_diffuse(above: f32, total: f32, mu_sun: f32, mu_light: f32, albedo: f32) -> f32 {
    if mu_sun <= 0.0 || total <= 0.0 { return 0.0; }
    let g1 = CLOUD_DRAINE_MEAN_COSINE;
    let scale = 1.0-g1*g1;
    let g = g1/(1.0+g1);
    let mu0 = max(mu_sun, 0.05);
    let t = above*scale;
    let slab = total*scale;
    let a = 0.75*mu0/ATM_PI;
    let e_slab = exp(-slab/mu0);
    let c1 = (1.0-albedo)*a*(mu0*(1.0-e_slab)+(2.0/3.0)*(1.0+e_slab))
        / ((1.0-albedo)*(1.0-g)*slab+4.0/3.0);
    let c2 = a*mu0+(2.0/3.0)*(a-c1);
    let e = exp(-t/mu0);
    let i1 = c1-a*e;
    let i0 = c2-a*mu0*e-(1.0-g)*c1*t;
    return max(i0+g1*mu_light*i1, 0.0);
}

// Scaled cloud optical depth from `position` toward the source, on five
// geometrically growing segments out to the shell top. Also returns the part
// of that depth exceeding what a column with the local cover would give on the
// same samples: the shadow cast by neighbouring cloud.
fn cloud_sun_depth(p: AtmosphereParams, position: vec3<f32>, direction: vec3<f32>, local_cover: f32) -> vec2<f32> {
    let exit = sphere_interval(position, direction, ground_radius(p)+CLOUD_TOP).y;
    if exit <= 0.0 { return vec2(0.0); }
    let reach = min(exit, 200.0);
    var depth = 0.0;
    var uniform_depth = 0.0;
    var near = 0.0;
    for (var j = 0u; j < 5u; j++) {
        let dt = reach*f32(1u<<j)/31.0;
        let sample_position = position+direction*(near+0.5*dt);
        let envelope = cloud_envelope(length(sample_position)-ground_radius(p));
        if envelope > 0.0 {
            depth += envelope*cloud_cover(p, normalize(sample_position), dt)*dt;
            uniform_depth += envelope*local_cover*dt;
        }
        near += dt;
    }
    let scaled = p.shell.z*CLOUD_DRAINE_WEIGHT;
    return vec2(depth, max(depth-uniform_depth, 0.0))*scaled;
}

struct MediumSample {
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
}

fn observer_position(p: AtmosphereParams) -> vec3<f32> {
    if p.planet.w > 0.0 { return p.planet.xyz; }
    return vec3(0.0, ground_radius(p) + p.observer.x * 0.001, 0.0);
}

fn sample_medium(p: AtmosphereParams, altitude: f32) -> MediumSample {
    if altitude > top_radius(p)-ground_radius(p) { return MediumSample(vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0)); }
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

fn ray_hits_ground(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>) -> bool {
    let r = length(origin);
    let b = dot(origin, direction);
    return b < 0.0 && b*b >= (r-ground_radius(p))*(r+ground_radius(p));
}

// xy = distance to the first boundary, whether it is opaque ground.
fn atmosphere_boundary(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    let r = length(origin);
    let b = dot(origin, direction);
    if ray_hits_ground(p, origin, direction) {
        let c = max(0.0, (r-ground_radius(p))*(r+ground_radius(p)));
        return vec2(c / max(-b + sqrt(max(b*b-c, 0.0)), 1e-8), 1.0);
    }
    return vec2(max(0.0, -b + sqrt(max(b*b+(top_radius(p)-r)*(top_radius(p)+r), 0.0))), 0.0);
}

fn optical_transmittance(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, distance: f32, samples: u32) -> vec3<f32> {
    let dt = distance / f32(max(samples, 1u));
    var optical = vec3(0.0);
    for (var i = 0u; i < samples; i++) {
        let position = origin + direction * ((f32(i)+0.5)*dt);
        optical += sample_medium(p, length(position)-ground_radius(p)).extinction * dt;
    }
    return exp(-optical);
}

fn transmittance_to_space(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, samples: u32) -> vec3<f32> {
    let boundary = atmosphere_boundary(p, origin, direction);
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
fn transmittance_uv(p: AtmosphereParams, radius: f32, mu: f32, dimensions: vec2<u32>) -> vec2<f32> {
    let r = clamp(radius, ground_radius(p), top_radius(p));
    let h = sqrt((top_radius(p)-ground_radius(p))*(top_radius(p)+ground_radius(p)));
    let rho = sqrt(max(0.0, (r-ground_radius(p))*(r+ground_radius(p))));
    let distance = max(0.0, -r*mu + sqrt(max(0.0, r*r*(mu*mu-1.0)+top_radius(p)*top_radius(p))));
    let x = clamp(vec2((distance-(top_radius(p)-r))/max(rho+h-(top_radius(p)-r), 1e-6), rho/h), vec2(0.0), vec2(1.0));
    let size = vec2<f32>(dimensions);
    return (vec2(0.5) + x*(size-1.0))/size;
}

fn transmittance_position(p: AtmosphereParams, uv: vec2<f32>) -> vec2<f32> {
    let h = sqrt((top_radius(p)-ground_radius(p))*(top_radius(p)+ground_radius(p)));
    let rho = h*uv.y;
    let r = sqrt(rho*rho+ground_radius(p)*ground_radius(p));
    let d = mix(top_radius(p)-r, rho+h, uv.x);
    let mu = select(clamp(((top_radius(p)-r)*(top_radius(p)+r)-d*d)/max(2.0*r*d, 1e-6), -1.0, 1.0), 1.0, d <= 1e-6);
    return vec2(r, mu);
}

fn sample_transmittance(p: AtmosphereParams, tex: texture_2d<f32>, filtering: sampler, position: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    if ray_hits_ground(p, position, direction) { return vec3(0.0); }
    let r = length(position);
    let uv = transmittance_uv(p, r, dot(position/r, direction), textureDimensions(tex));
    return clamp(textureSampleLevel(tex, filtering, uv, 0.0).rgb, vec3(0.0), vec3(1.0));
}

fn sample_multiple(p: AtmosphereParams, tex: texture_2d<f32>, filtering: sampler, position: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let r = length(position);
    let unit = clamp(vec2(dot(position/r, direction)*0.5+0.5, (r-ground_radius(p))/(top_radius(p)-ground_radius(p))), vec2(0.0), vec2(1.0));
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
        let direct = sample_transmittance(p, trans, filtering, position, source.xyz) * phase;
        let ms = sample_multiple(p, multi, filtering, position, source.xyz) * medium.scattering * p.ground.w;
        result += source.w * (direct+ms) * source_visibility(p, position, source.xyz);
    }
    return result;
}

// Front-to-back composition of two consecutive ray segments.
fn combine_transport(front: AtmosphereTransport, back: AtmosphereTransport) -> AtmosphereTransport {
    return AtmosphereTransport(front.radiance+front.transmittance*back.radiance,
        front.transmittance*back.transmittance);
}

// Clear air over [t0, t1]. Quadratic intervals concentrate samples near t0,
// which suits an observer inside the medium; both spacings cover the endpoint.
fn integrate_air(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, t0: f32, t1: f32, samples: u32,
    quadratic: bool, trans: texture_2d<f32>, multi: texture_2d<f32>, filtering: sampler) -> AtmosphereTransport {
    var result = AtmosphereTransport(vec3(0.0), vec3(1.0));
    let span = t1-t0;
    if span <= 0.0 { return result; }
    for (var i = 0u; i < samples; i++) {
        let a = f32(i)/f32(samples);
        let b = f32(i+1u)/f32(samples);
        let near = t0+select(a, a*a, quadratic)*span;
        let far = t0+select(b, b*b, quadratic)*span;
        let dt = far-near;
        let position = origin+direction*((near+far)*0.5);
        let m = sample_medium(p, length(position)-ground_radius(p));
        let source = local_inscattering(p, position, direction, m, trans, multi, filtering);
        result.radiance += result.transmittance * source * segment_integral(m.extinction, dt);
        result.transmittance *= exp(-m.extinction*dt);
    }
    return result;
}

// Cloud shell altitudes above the ground radius, km.
const CLOUD_BASE: f32 = 2.0;
const CLOUD_TOP: f32 = 8.0;
// Cloud march step in km at the 128-sample budget; the step scales inversely
// with the budget, which also caps the count on long grazing paths.
const CLOUD_STEP: f32 = 0.25;
const CLOUD_MIN_SAMPLES: u32 = 8u;

// Air and cloud over one crossing of the cloud shell, on uniform midpoints.
// The shell is a few km thick inside a ~100 km atmosphere, so it cannot share
// the clear-air quadrature without aliasing into altitude contours.
fn integrate_cloud(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, t0: f32, t1: f32, budget: u32,
    trans: texture_2d<f32>, multi: texture_2d<f32>, filtering: sampler) -> AtmosphereTransport {
    var result = AtmosphereTransport(vec3(0.0), vec3(1.0));
    let span = t1-t0;
    if span <= 0.0 { return result; }
    let limit = max(budget, CLOUD_MIN_SAMPLES);
    let count = clamp(u32(ceil(span*f32(limit)/(CLOUD_STEP*128.0))), CLOUD_MIN_SAMPLES, limit);
    let dt = span/f32(count);
    for (var i = 0u; i < count; i++) {
        let position = origin+direction*(t0+(f32(i)+0.5)*dt);
        let radius = length(position);
        let normal = position/radius;
        let h = radius-ground_radius(p);
        var m = sample_medium(p, h);
        var source = local_inscattering(p, position, direction, m, trans, multi, filtering);
        let envelope = cloud_envelope(h);
        var cover = 0.0;
        if envelope > 0.0 { cover = cloud_cover(p, normal, dt); }
        // Extinction of the delta-scaled medium, km^-1.
        let column = cover*p.shell.z*CLOUD_DRAINE_WEIGHT;
        let cloud = envelope*column;
        if cloud > 0.0 {
            let irradiance = sample_transmittance(p, trans, filtering, position, p.sun.xyz)
                * source_visibility(p, position, p.sun.xyz)*p.sun.w;
            if max(irradiance.x, max(irradiance.y, irradiance.z)) > 0.0 {
                let sun_depth = cloud_sun_depth(p, position, p.sun.xyz, cover);
                let single = phase_draine(dot(direction, p.sun.xyz))*exp(-sun_depth.x);
                // Neighbouring cloud that shadows the direct beam also starves
                // the diffuse field; applying its excess depth is a heuristic.
                let diffuse = cloud_diffuse(cloud_column_above(h)*column, cloud_column_above(CLOUD_BASE)*column,
                    dot(normal, p.sun.xyz), dot(direction, normal), dot(p.ground.rgb, vec3(1.0/3.0)))
                    * exp(-sun_depth.y);
                source += cloud*irradiance*(single+diffuse);
            }
            m.extinction += vec3(cloud);
        }
        result.radiance += result.transmittance * source * segment_integral(m.extinction, dt);
        result.transmittance *= exp(-m.extinction*dt);
        if max(result.transmittance.x, max(result.transmittance.y, result.transmittance.z)) < 0.0001 { break; }
    }
    return result;
}

// Planetary view transport with clouds: clear air and cloud-shell crossings
// are integrated separately and composited in ray order. A ray crosses the
// shell at most twice, before and after passing below the cloud base.
fn integrate_view(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, distance: f32, samples: u32,
    trans: texture_2d<f32>, multi: texture_2d<f32>, filtering: sampler) -> AtmosphereTransport {
    let outer = sphere_interval(origin, direction, top_radius(p));
    if outer.y <= 0.0 { return AtmosphereTransport(vec3(0.0),vec3(1.0)); }
    let start = max(outer.x, 0.0);
    let end = max(start, min(distance, atmosphere_boundary(p, origin, direction).x));
    if p.shell.y <= 0.0 || p.shell.z <= 0.0 {
        return integrate_air(p, origin, direction, start, end, samples, true, trans, multi, filtering);
    }
    let top = sphere_interval(origin, direction, ground_radius(p)+CLOUD_TOP);
    let base = sphere_interval(origin, direction, ground_radius(p)+CLOUD_BASE);
    var first = top;
    var second = vec2(start);
    if base.y > base.x {
        first = vec2(top.x, base.x);
        second = vec2(base.y, top.y);
    }
    first = clamp(first, vec2(start), vec2(end));
    second = clamp(second, vec2(first.y), vec2(end));
    let has_second = second.y > second.x;
    let middle = select(end, second.x, has_second);
    // Clear-air samples follow path length. An observer inside the medium
    // keeps the near-weighted spacing on the first segment.
    let scale = f32(samples)/max(end-start, 1e-6);
    var result = integrate_air(p, origin, direction, start, first.x,
        clamp(u32(ceil((first.x-start)*scale)), 8u, samples), start <= 0.0, trans, multi, filtering);
    result = combine_transport(result, integrate_cloud(p, origin, direction, first.x, first.y, samples, trans, multi, filtering));
    result = combine_transport(result, integrate_air(p, origin, direction, first.y, middle,
        clamp(u32(ceil((middle-first.y)*scale)), 8u, samples), first.y <= 0.0, trans, multi, filtering));
    if has_second {
        result = combine_transport(result, integrate_cloud(p, origin, direction, second.x, second.y, samples, trans, multi, filtering));
        result = combine_transport(result, integrate_air(p, origin, direction, second.y, end,
            clamp(u32(ceil((end-second.y)*scale)), 8u, samples), false, trans, multi, filtering));
    }
    return result;
}

// Clear-air transport for lookup fields and local-ground views.
fn integrate_atmosphere(p: AtmosphereParams, origin: vec3<f32>, direction: vec3<f32>, distance: f32, samples: u32,
    trans: texture_2d<f32>, multi: texture_2d<f32>, filtering: sampler, ground: bool) -> AtmosphereTransport {
    let outer = sphere_interval(origin, direction, top_radius(p));
    if outer.y <= 0.0 { return AtmosphereTransport(vec3(0.0),vec3(1.0)); }
    let start = max(outer.x, 0.0);
    let boundary = atmosphere_boundary(p, origin, direction);
    let length_max = max(0.0, min(distance, boundary.x)-start);
    var result = integrate_air(p, origin, direction, start, start+length_max, samples, true, trans, multi, filtering);
    if ground && boundary.y > 0.0 && distance >= boundary.x {
        let position = origin+direction*boundary.x;
        let normal = normalize(position);
        let sources = array<vec4<f32>, 2>(p.sun, p.moon);
        for (var i = 0u; i < 2u; i++) {
            let source = sources[i];
            let direct = sample_transmittance(p, trans, filtering, normal*(ground_radius(p)+0.001), source.xyz)
                * max(dot(normal, source.xyz), 0.0);
            result.radiance += result.transmittance * p.ground.rgb * source.w * direct / ATM_PI;
        }
    }
    return result;
}

fn horizon_elevation(p: AtmosphereParams) -> f32 {
    let r = length(observer_position(p));
    return -acos(clamp(ground_radius(p)/r, 0.0, 1.0));
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
