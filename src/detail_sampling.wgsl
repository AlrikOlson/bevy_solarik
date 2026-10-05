#define_import_path bevy_solarik::detail_sampling

#ifdef BINDLESS_SURFACE_DETAIL
@group(0) @binding(21) var detail_textures: binding_array<texture_2d_array<f32>>;
@group(0) @binding(22) var detail_samplers: binding_array<sampler>;
#endif

// A tangent-space normal represents the slope of an unresolved height field.
// Project its three-dimensional gradient onto the actual surface tangent plane.
// Mikkelsen 2020, equation 5. No radial-normal interpolation on vertical faces.
fn detail_normal(normal: vec3f, gradient: vec3f) -> vec3f {
    return normalize(normal - (gradient - normal * dot(normal, gradient)));
}

struct DetailCoordinates {
    phase: array<vec4f, 8>,
    cell: array<vec4i, 8>,
}

struct DetailSample {
    colour: vec3f,
    roughness: f32,
    gradient: vec3f,
}

fn detail_hash(p: vec2i, layer: u32) -> vec2f {
    var h = bitcast<vec2u>(p) * vec2u(1597334677u, 3812015801u);
    h = (h ^ h.yx ^ vec2u(layer * 2246822519u)) * 3266489917u;
    h = h ^ (h >> vec2u(16u));
    return vec2f(h >> vec2u(8u)) * (1.0 / 16777216.0);
}

// Three translated scan patches on a triangular lattice. Translations keep
// the scan's measured length and slope; random placement is a stationary
// microstructure approximation, not a simulation of erosion or deposition.
// Squared barycentric weights make the patch derivative vanish on borders.
fn detail_plane(
#ifdef BINDLESS_SURFACE_DETAIL
    colours: u32, details: u32, scan_sampler: u32,
#else
    colours: texture_2d_array<f32>, details: texture_2d_array<f32>,
    scan_sampler: sampler,
#endif
    p: vec2f, cell: vec2i, layer: u32, lod: f32,
) -> array<vec4f, 2> {
    let base = vec2i(floor(p));
    let f = fract(p);
    var offsets = array<vec2i, 3>(vec2i(0), vec2i(1, 0), vec2i(1, 1));
    var weights = vec3f(1.0 - f.x, f.x - f.y, f.y);
    if f.y > f.x {
        offsets = array<vec2i, 3>(vec2i(0), vec2i(0, 1), vec2i(1, 1));
        weights = vec3f(1.0 - f.y, f.y - f.x, f.x);
    }
    weights *= weights;
    weights /= dot(weights, vec3f(1.0));
    var c = vec4f(0.0);
    var d = vec4f(0.0);
    for (var j = 0u; j < 3u; j++) {
        let uv = f + detail_hash(cell + base + offsets[j], layer);
#ifdef BINDLESS_SURFACE_DETAIL
        c += weights[j] * textureSampleLevel(detail_textures[colours], detail_samplers[scan_sampler], uv, layer, lod);
        d += weights[j] * textureSampleLevel(detail_textures[details], detail_samplers[scan_sampler], uv, layer, lod);
#else
        c += weights[j] * textureSampleLevel(colours, scan_sampler, uv, layer, lod);
        d += weights[j] * textureSampleLevel(details, scan_sampler, uv, layer, lod);
#endif
    }
    return array<vec4f, 2>(c, d);
}

// Inputs have mean 0.5 for colour and roughness; geometric coverage is given
// by the application. Filter footprints are in metres, independent of tile LOD.
fn sample_surface_detail(
#ifdef BINDLESS_SURFACE_DETAIL
    colours: u32, details: u32, scan_sampler: u32,
#else
    colours: texture_2d_array<f32>, details: texture_2d_array<f32>,
    scan_sampler: sampler,
#endif
    coordinates: DetailCoordinates,
    local_position: vec3f, normal: vec3f, footprint: f32,
    coverage0: vec4f, coverage1: vec4f,
) -> DetailSample {
    var plane_weights = pow(abs(normal), vec3f(4.0));
    plane_weights /= dot(plane_weights, vec3f(1.0));
    let cover = array<vec4f, 2>(coverage0, coverage1);
    var result = DetailSample(vec3f(1.0), 1.0, vec3f(0.0));
    for (var k = 0u; k < 8u; k++) {
        let weight = cover[k / 4u][k % 4u];
        let scale = coordinates.phase[k].w;
        let relative_width = footprint * scale;
        let fade = 1.0 - smoothstep(0.1, 0.5, relative_width);
        if weight * fade < 0.0001 { continue; }
#ifdef BINDLESS_SURFACE_DETAIL
        let resolution = f32(textureDimensions(detail_textures[colours]).x);
#else
        let resolution = f32(textureDimensions(colours).x);
#endif
        let lod = max(0.0, log2(max(relative_width * resolution, 1.0)));
        let p = local_position * scale + coordinates.phase[k].xyz;
        let cell = coordinates.cell[k].xyz;
        var c = vec3f(0.0);
        var rough = 0.0;
        var gradient = vec3f(0.0);
        for (var axis = 0u; axis < 3u; axis++) {
            if plane_weights[axis] == 0.0 { continue; }
            var uv = p.yz;
            var ij = cell.yz;
            if axis == 1u { uv = p.zx; ij = cell.zx; }
            if axis == 2u { uv = p.xy; ij = cell.xy; }
            let taps = detail_plane(colours, details, scan_sampler, uv, ij, k, lod);
            c += taps[0].rgb * plane_weights[axis];
            rough += taps[1].z * plane_weights[axis];
            let xy = taps[1].xy * 2.0 - 1.0;
            let slope = -xy / sqrt(max(1.0 - dot(xy, xy), 0.01));
            var g = vec3f(0.0, slope.x, slope.y);
            if axis == 1u { g = vec3f(slope.y, 0.0, slope.x); }
            if axis == 2u { g = vec3f(slope.x, slope.y, 0.0); }
            gradient += plane_weights[axis] * g;
        }
        let w = weight * fade;
        result.colour += w * (c * 2.0 - vec3f(1.0));
        result.roughness += w * (rough * 2.0 - 1.0);
        result.gradient += w * gradient;
    }
    return result;
}
