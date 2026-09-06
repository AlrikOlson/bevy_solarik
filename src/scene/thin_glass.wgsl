#define_import_path bevy_solarik::thin_glass

// A smooth, parallel-sided pane collapsed to one interface. Transmission
// exits along the incident direction; no thickness, dispersion or roughness.
struct ThinGlassSample {
    wi: vec3<f32>,
    throughput: vec3<f32>,
    reflected: bool,
}

// RGB transmission energy and scalar reflection energy, before sampling.
fn thin_glass_weights(wo: vec3<f32>, normal: vec3<f32>, base_color: vec3<f32>, alpha: f32, reflectance: f32) -> vec4<f32> {
    let coverage = clamp(alpha, 0.0, 1.0);
    let f0 = clamp(0.16 * reflectance * reflectance, 0.0, 1.0);
    let cosine = clamp(abs(dot(normal, wo)), 0.0, 1.0);
    let f = f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0);
    // Sum internal reflections: F + (1-F)^2 F / (1-F^2) = 2F/(1+F).
    // This form is finite at grazing incidence F=1.
    let pane_reflection = 2.0 * f / (1.0 + f);
    let reflection = coverage * pane_reflection;
    // Uncovered area is an untinted hole. Tint absorbs energy only from
    // covered transmitted paths. Divide by branch probability exactly once;
    // no eta^2 for a pane whose exterior IOR agrees on both sides.
    let transmission = vec3(1.0 - coverage)
        + coverage * (1.0 - pane_reflection) * clamp(base_color, vec3(0.0), vec3(1.0));
    return vec4(transmission, reflection);
}

fn sample_thin_glass(wo: vec3<f32>, normal: vec3<f32>, base_color: vec3<f32>, alpha: f32, reflectance: f32, u: f32) -> ThinGlassSample {
    let weights = thin_glass_weights(wo, normal, base_color, alpha, reflectance);
    if u < weights.a {
        return ThinGlassSample(reflect(-wo, normal), vec3(1.0), true);
    }
    return ThinGlassSample(-wo, weights.rgb / max(1.0 - weights.a, 0.0000001), false);
}

fn offset_thin_glass_ray(position: vec3<f32>, geometric_normal: vec3<f32>, wi: vec3<f32>, epsilon: f32) -> vec3<f32> {
    let side = select(-1.0, 1.0, dot(geometric_normal, wi) >= 0.0);
    return position + geometric_normal * (side * epsilon);
}
