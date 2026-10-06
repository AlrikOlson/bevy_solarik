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

// Two octaves of scanned detail per layer: the fine scans (metres) and the
// mesoscale scans (tens of metres). Each entry is the fractional phase of
// the material-frame origin in scan cells (xyz) and the inverse scan size
// in 1/m (w); a zero inverse size disables that octave for the layer.
struct DetailCoordinates {
    phase: array<vec4f, 8>,
    cell: array<vec4i, 8>,
    meso_phase: array<vec4f, 8>,
    meso_cell: array<vec4i, 8>,
}

struct DetailSample {
    colour: vec3f,
    roughness: f32,
    gradient: vec3f,
}

// One octave of one layer: colour and roughness as factors about 1, the
// tangent gradient, and the share of the octave kept at this footprint.
struct OctaveSample {
    colour: vec3f,
    roughness: f32,
    gradient: vec3f,
    fade: f32,
}

fn detail_hash(p: vec2i, layer: u32) -> vec2f {
    // PCG3D: Jarzynski & Olano 2020, section 6.1. Sequential mixing
    // matters: symmetric h ^ h.yx collapses both translation lanes.
    var h = vec3u(bitcast<vec2u>(p), layer) * 1664525u + 1013904223u;
    h.x += h.y * h.z; h.y += h.z * h.x; h.z += h.x * h.y;
    h ^= h >> vec3u(16u);
    h.x += h.y * h.z; h.y += h.z * h.x; h.z += h.x * h.y;
    return vec2f(h.xy >> vec2u(8u)) * (1.0 / 16777216.0);
}

fn detail_rotate(p: vec2f, turn: u32) -> vec2f {
    switch turn {
        case 1u: { return vec2f(-p.y, p.x); }
        case 2u: { return -p; }
        case 3u: { return vec2f(p.y, -p.x); }
        default: { return p; }
    }
}

fn detail_patch_turn(cell: vec2i, layer: u32) -> u32 {
    // Grass and forest-floor microstructure has no prescribed orientation.
    // Keep bedded rock and the other directional scan layers unchanged.
    if layer != 5u && layer != 6u { return 0u; }
    return u32(detail_hash(cell, layer + 64u).x * 4.0);
}

// Three translated scan patches on a triangular lattice. Translations keep
// the scan's measured length and slope; random placement is a stationary
// microstructure approximation, not a simulation of erosion or deposition.
// Vegetation patches additionally take quarter turns, with normals transformed
// back by the inverse rotation. Integer turns preserve seamless repeat edges.
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
        let lattice = cell + base + offsets[j];
        let turn = detail_patch_turn(lattice, layer);
        let uv = detail_rotate(f, turn) + detail_hash(lattice, layer);
#ifdef BINDLESS_SURFACE_DETAIL
        c += weights[j] * textureSampleLevel(detail_textures[colours], detail_samplers[scan_sampler], uv, layer, lod);
        var detail = textureSampleLevel(detail_textures[details], detail_samplers[scan_sampler], uv, layer, lod);
#else
        c += weights[j] * textureSampleLevel(colours, scan_sampler, uv, layer, lod);
        var detail = textureSampleLevel(details, scan_sampler, uv, layer, lod);
#endif
        detail = vec4f(detail_rotate(detail.xy * 2.0 - 1.0, (4u-turn)%4u) * 0.5 + 0.5, detail.zw);
        d += weights[j] * detail;
    }
    return array<vec4f, 2>(c, d);
}

// One octave of layer `layer` at `local_position`, projected on the three
// planes with `plane_weights`. Inputs have mean 0.5 for colour and
// roughness, so the factors returned have mean 1. The octave fades to its
// mean between footprints of 0.1 and 0.5 of its own scan size: a filtering
// choice, not a material property.
fn detail_octave(
#ifdef BINDLESS_SURFACE_DETAIL
    colours: u32, details: u32, scan_sampler: u32,
#else
    colours: texture_2d_array<f32>, details: texture_2d_array<f32>,
    scan_sampler: sampler,
#endif
    phase: vec4f, cell: vec3i, plane_weights: vec3f,
    local_position: vec3f, footprint: f32, layer: u32,
) -> OctaveSample {
    var result = OctaveSample(vec3f(1.0), 1.0, vec3f(0.0), 0.0);
    let scale = phase.w;
    if scale <= 0.0 { return result; }
    let relative_width = footprint * scale;
    let fade = 1.0 - smoothstep(0.1, 0.5, relative_width);
    if fade < 0.0001 { return result; }
#ifdef BINDLESS_SURFACE_DETAIL
    let resolution = f32(textureDimensions(detail_textures[colours]).x);
#else
    let resolution = f32(textureDimensions(colours).x);
#endif
    let lod = max(0.0, log2(max(relative_width * resolution, 1.0)));
    let p = local_position * scale + phase.xyz;
    var c = vec3f(0.0);
    var rough = 0.0;
    var gradient = vec3f(0.0);
    for (var axis = 0u; axis < 3u; axis++) {
        if plane_weights[axis] == 0.0 { continue; }
        var uv = p.yz;
        var ij = cell.yz;
        if axis == 1u { uv = p.zx; ij = cell.zx; }
        if axis == 2u { uv = p.xy; ij = cell.xy; }
        let taps = detail_plane(colours, details, scan_sampler, uv, ij, layer, lod);
        c += taps[0].rgb * plane_weights[axis];
        rough += taps[1].z * plane_weights[axis];
        let xy = taps[1].xy * 2.0 - 1.0;
        let slope = -xy / sqrt(max(1.0 - dot(xy, xy), 0.01));
        var g = vec3f(0.0, slope.x, slope.y);
        if axis == 1u { g = vec3f(slope.y, 0.0, slope.x); }
        if axis == 2u { g = vec3f(slope.x, slope.y, 0.0); }
        gradient += plane_weights[axis] * g;
    }
    result.colour = mix(vec3f(1.0), c * 2.0, fade);
    result.roughness = mix(1.0, rough * 2.0, fade);
    result.gradient = fade * gradient;
    result.fade = fade;
    return result;
}

// Inputs have mean 0.5 for colour and roughness; geometric coverage is given
// by the application. Filter footprints are in metres, independent of tile LOD.
// Each layer's fine and mesoscale octaves multiply as factors about their
// means and add their gradients: the coarse octave shapes the fine one, and
// a layer with its mesoscale disabled is the single-octave result.
fn sample_surface_detail(
#ifdef BINDLESS_SURFACE_DETAIL
    colours: u32, details: u32, scan_sampler: u32,
    meso_colours: u32, meso_details: u32,
#else
    colours: texture_2d_array<f32>, details: texture_2d_array<f32>,
    scan_sampler: sampler,
    meso_colours: texture_2d_array<f32>, meso_details: texture_2d_array<f32>,
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
        if weight < 0.0001 { continue; }
        let fine = detail_octave(colours, details, scan_sampler,
            coordinates.phase[k], coordinates.cell[k].xyz, plane_weights,
            local_position, footprint, k);
        let meso = detail_octave(meso_colours, meso_details, scan_sampler,
            coordinates.meso_phase[k], coordinates.meso_cell[k].xyz, plane_weights,
            local_position, footprint, k);
        if fine.fade + meso.fade <= 0.0 { continue; }
        result.colour += weight * (fine.colour * meso.colour - vec3f(1.0));
        result.roughness += weight * (fine.roughness * meso.roughness - 1.0);
        result.gradient += weight * (fine.gradient + meso.gradient);
    }
    return result;
}
