#define_import_path bevy_solarik::light_medium

// A planet's clear air, as the light of a distant source crosses it on the
// way to a surface point. The profiles are those of the atmosphere pass:
// Rayleigh and Mie extinction falling off over 8 km and 1.2 km, and an ozone
// layer centred at 25 km. Distances are metres.
struct LightMedium {
    // Planet centre in world space.
    centre: vec3<f32>,
    // Ground radius; zero means there is no medium.
    radius: f32,
    // Rayleigh extinction at the ground, per metre.
    rayleigh: vec3<f32>,
    // Outer radius of the air.
    top: f32,
    // Ozone absorption at the middle of its layer, per metre.
    ozone: vec3<f32>,
    // Mie extinction at the ground, per metre.
    mie: f32,
    // Cloud layer: radius of its base and of its highest top, peak
    // extinction of the delta-scaled cloud medium per metre, and whether a
    // weather map gives it shape (above zero).
    cloud: vec4<f32>,
    physical: array<vec4<f32>,5>,
}

const MEDIUM_RAYLEIGH_HEIGHT: f32 = 8000.0;
const MEDIUM_MIE_HEIGHT: f32 = 1200.0;
const MEDIUM_OZONE_CENTRE: f32 = 25000.0;
const MEDIUM_OZONE_HALF_WIDTH: f32 = 15000.0;
const MEDIUM_STEPS: u32 = 24u;
// Mean cosine of the cloud droplets' Draine lobe, and the integral of the
// adiabatic cloud profile over a cloud's thickness: the atmosphere pass's
// CLOUD_DRAINE_MEAN_COSINE and cloud_profile_above(0).
const MEDIUM_CLOUD_MEAN_COSINE: f32 = 0.763159;
const MEDIUM_CLOUD_COLUMN: f32 = 0.585;

fn medium_extinction(m: LightMedium, altitude: f32) -> vec3<f32> {
    if m.physical[0].w>0.0 {
        let gas=m.physical[0]; let thermal=m.physical[1]; let aerosol=m.physical[2]; let particle=m.physical[3];
        let h=altitude*0.001-thermal.w;
        var density=exp(-h/gas.w);
        if thermal.y>0.0 {
            let t=max(thermal.x-thermal.y*h,thermal.z);
            let cap=(thermal.x-thermal.z)/thermal.y;
            density=pow(t/thermal.x,thermal.x/(thermal.y*gas.w)-1.0)*exp(-max(h-cap,0.0)/(gas.w*thermal.z/thermal.x));
        }
        var a=exp(-h/aerosol.w);
        if particle.w>0.0 {let z=(h-particle.w)/aerosol.w;a=exp(-0.5*z*z);}
        return (gas.rgb*density+aerosol.rgb*a)*0.001;
    }
    let h = max(altitude, 0.0);
    return m.rayleigh * exp(-h / MEDIUM_RAYLEIGH_HEIGHT)
        + vec3(m.mie * exp(-h / MEDIUM_MIE_HEIGHT))
        + m.ozone * max(0.0, 1.0 - abs(h - MEDIUM_OZONE_CENTRE) / MEDIUM_OZONE_HALF_WIDTH);
}

// Share of the light arriving along `direction` (unit, towards the source)
// that reaches `origin` through the medium. The ground is not tested: shadow
// rays find it.
fn medium_transmittance(m: LightMedium, origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    if m.radius <= 0.0 { return vec3(1.0); }
    let p = origin - m.centre;
    let b = dot(p, direction);
    let discriminant = b * b - dot(p, p) + m.top * m.top;
    if discriminant <= 0.0 { return vec3(1.0); }
    let root = sqrt(discriminant);
    let near = max(-b - root, 0.0);
    let far = -b + root;
    if far <= near { return vec3(1.0); }
    let length_inside = far - near;
    // From inside the air, extinction is densest at the start of the path:
    // samples at t = L u^2 put a third of them in the first ninth of it. From
    // outside, the densest part is in the middle and even steps serve.
    let inside = near == 0.0;
    var depth = vec3(0.0);
    let steps=select(MEDIUM_STEPS,128u,m.physical[0].w>0.0);
    for (var i = 0u; i < steps; i += 1u) {
        let u = (f32(i) + 0.5) / f32(steps);
        let t = near + length_inside * select(u, u * u, inside);
        let weight = length_inside * select(1.0, 2.0 * u, inside) / f32(steps);
        let altitude = length(p + direction * t) - m.radius;
        depth += medium_extinction(m, altitude) * weight;
    }
    return exp(-depth);
}

// Share of a beam's flux that leaves the bottom of a cloud of vertical scaled
// optical depth `total`, direct and diffuse together, for a sun at zenith
// cosine `mu_sun`: the conservative Eddington slab of the atmosphere pass
// (its `cloud_slab_transmission`). The diffuse part is counted as if it came
// from the sun's direction, which is right for level ground.
fn cloud_transmission(mu_sun: f32, total: f32) -> f32 {
    if total <= 0.0 { return 1.0; }
    let g1 = MEDIUM_CLOUD_MEAN_COSINE;
    let g = g1/(1.0+g1);
    let mu0 = max(mu_sun, 0.05);
    let slab = total*(1.0-g1*g1);
    let direct = exp(-slab/mu0);
    let diffuse = 0.75*(1.0-direct)*(mu0+2.0/3.0)
        - 0.75*(1.0-g)*slab*(mu0*(1.0-direct)+(2.0/3.0)*(1.0+direct))/((1.0-g)*slab+4.0/3.0);
    return clamp(direct+diffuse, 0.0, 1.0);
}
