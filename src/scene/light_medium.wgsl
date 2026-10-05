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
}

const MEDIUM_RAYLEIGH_HEIGHT: f32 = 8000.0;
const MEDIUM_MIE_HEIGHT: f32 = 1200.0;
const MEDIUM_OZONE_CENTRE: f32 = 25000.0;
const MEDIUM_OZONE_HALF_WIDTH: f32 = 15000.0;
const MEDIUM_STEPS: u32 = 24u;

fn medium_extinction(m: LightMedium, altitude: f32) -> vec3<f32> {
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
    for (var i = 0u; i < MEDIUM_STEPS; i += 1u) {
        let u = (f32(i) + 0.5) / f32(MEDIUM_STEPS);
        let t = near + length_inside * select(u, u * u, inside);
        let weight = length_inside * select(1.0, 2.0 * u, inside) / f32(MEDIUM_STEPS);
        let altitude = length(p + direction * t) - m.radius;
        depth += medium_extinction(m, altitude) * weight;
    }
    return exp(-depth);
}
