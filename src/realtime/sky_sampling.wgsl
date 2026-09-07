#define_import_path bevy_solarik::sky_sampling

#import bevy_render::maths::PI
#import bevy_pbr::utils::{rand_f, sample_cosine_hemisphere}
#import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance
#import bevy_solarik::scene_bindings::sample_sky

// Fixed sampling grid, independent of the input cubemap's resolution.
// Per-row inclusive CDFs followed by an inclusive marginal CDF over rows.
// Must match prepare.rs's allocation and node.rs's row dispatch count.
struct SkyDistribution {
    columns: array<f32, 98304>,
    rows: array<f32, 768>,
}
@group(1) @binding(23) var<storage, read_write> sky_distribution: SkyDistribution;
var<workgroup> sky_scan: array<f32, 128>;

struct SkyAddress {
    face: u32,
    uv: vec2<f32>,
}

// Cube texture face convention, converted to world space by negating z
// exactly as scene_bindings::sample_sky does in the other direction.
fn sky_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3(1.0, -uv.y, -uv.x); }
        case 1u: { d = vec3(-1.0, -uv.y, uv.x); }
        case 2u: { d = vec3(uv.x, 1.0, uv.y); }
        case 3u: { d = vec3(uv.x, -1.0, -uv.y); }
        case 4u: { d = vec3(uv.x, -uv.y, 1.0); }
        default: { d = vec3(-uv.x, -uv.y, -1.0); }
    }
    return normalize(vec3(d.xy, -d.z));
}

fn sky_address(direction: vec3<f32>) -> SkyAddress {
    let d = vec3(direction.xy, -direction.z);
    let a = abs(d);
    if a.x >= a.y && a.x >= a.z {
        if d.x >= 0.0 { return SkyAddress(0u, vec2(-d.z, -d.y) / a.x); }
        return SkyAddress(1u, vec2(d.z, -d.y) / a.x);
    }
    if a.y >= a.z {
        if d.y >= 0.0 { return SkyAddress(2u, vec2(d.x, d.z) / a.y); }
        return SkyAddress(3u, vec2(d.x, -d.z) / a.y);
    }
    if d.z >= 0.0 { return SkyAddress(4u, vec2(d.x, -d.y) / a.z); }
    return SkyAddress(5u, vec2(-d.x, -d.y) / a.z);
}

fn sky_jacobian(uv: vec2<f32>) -> f32 {
    let r2 = 1.0 + dot(uv, uv);
    return 1.0 / (r2 * sqrt(r2));
}

@compute @workgroup_size(128)
fn build_sky_rows(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) x: u32) {
    let row = group.x;
    let uv = (vec2(f32(x), f32(row % 128u)) + 0.5) * (2.0 / 128.0) - 1.0;
    let radiance = sample_sky(sky_direction(row / 128u, uv));
    // The common cell area cancels in the normalized distribution.
    sky_scan[x] = max(luminance(radiance), 0.0) * sky_jacobian(uv);
    workgroupBarrier();
    for (var stride = 1u; stride < 128u; stride *= 2u) {
        var previous = 0.0;
        if x >= stride { previous = sky_scan[x - stride]; }
        workgroupBarrier();
        sky_scan[x] += previous;
        workgroupBarrier();
    }
    sky_distribution.columns[row * 128u + x] = sky_scan[x];
}

@compute @workgroup_size(1)
fn build_sky_marginal() {
    var total = 0.0;
    for (var row = 0u; row < 768u; row++) {
        total += sky_distribution.columns[row * 128u + 127u];
        sky_distribution.rows[row] = total;
    }
}

fn sky_pdf(direction: vec3<f32>) -> f32 {
    let total = sky_distribution.rows[767u];
    if total <= 0.0 { return 0.0; }
    let address = sky_address(direction);
    let cell = min(vec2<u32>(max((address.uv + 1.0) * 64.0, vec2(0.0))), vec2(127u));
    let row = address.face * 128u + cell.y;
    let index = row * 128u + cell.x;
    var previous = 0.0;
    if cell.x > 0u { previous = sky_distribution.columns[index - 1u]; }
    let mass = max(sky_distribution.columns[index] - previous, 0.0);
    // Uniform jitter in cell UV -> density per steradian via exact Jacobian.
    return (mass / total) * (128.0 * 128.0 / 4.0) / sky_jacobian(address.uv);
}

fn sample_sky_direction(rng: ptr<function, u32>) -> vec3<f32> {
    // Keep the search strictly inside the CDF even after float multiplication.
    let target_row = min(rand_f(rng), 0.99999988) * sky_distribution.rows[767u];
    var lo = 0u;
    var hi = 767u;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if sky_distribution.rows[mid] <= target_row { lo = mid + 1u; }
        else { hi = mid; }
    }
    let row = lo;
    let offset = row * 128u;
    let target_column = min(rand_f(rng), 0.99999988) * sky_distribution.columns[offset + 127u];
    lo = 0u;
    hi = 127u;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if sky_distribution.columns[offset + mid] <= target_column { lo = mid + 1u; }
        else { hi = mid; }
    }
    let uv = (vec2(f32(lo), f32(row % 128u)) + vec2(rand_f(rng), rand_f(rng))) * (2.0 / 128.0) - 1.0;
    return sky_direction(row / 128u, uv);
}

struct SkyMixtureSample {
    direction: vec3<f32>,
    inverse_pdf: f32,
}

fn sample_sky_mixture(normal: vec3<f32>, rng: ptr<function, u32>) -> SkyMixtureSample {
    let sky_probability = select(0.0, 0.5, sky_distribution.rows[767u] > 0.0);
    var direction: vec3<f32>;
    if rand_f(rng) < sky_probability {
        direction = sample_sky_direction(rng);
    } else {
        direction = sample_cosine_hemisphere(normal, rng);
    }
    // One-sample balance MIS. Use this density on both geometry hits and
    // sky misses; choosing a technique does not remove the other technique.
    let pdf = sky_probability * sky_pdf(direction)
        + (1.0 - sky_probability) * max(dot(normal, direction), 0.0) / PI;
    return SkyMixtureSample(direction, 1.0 / max(pdf, 1e-20));
}
