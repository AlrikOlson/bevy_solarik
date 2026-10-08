#define_import_path bevy_solarik::coarse_scene
// Shared by actual raster/ray consumers. This decodes measurements only.
const COARSE_MISSING = 0xffffffffu;
const COARSE_DIRECTIONS = 14u;
struct CoarseGrid {
    origin_resolution: vec4i,
    dimensions: vec4u,
    metadata: vec4u,
}
struct CoarsePacked { words: array<u32,19> }
struct CoarseMeasurements {
    counts: vec4u,
    depth: vec4f,
    diagonal: vec4f,
    cross_moment: vec4f,
    normal: vec4f,
    materials: array<u32,8>,
}
fn coarse_grid_index(grid: CoarseGrid, point: vec3f) -> u32 {
    let address = floor(point*f32(grid.origin_resolution.w))-vec3f(grid.origin_resolution.xyz);
    // Comparisons reject NaN, infinities, negative and upper-face coordinates.
    if !all(address >= vec3f(0.0)) || !all(address < vec3f(grid.dimensions.xyz)) { return COARSE_MISSING; }
    let p = vec3u(address);
    return (p.x*grid.dimensions.y+p.y)*grid.dimensions.z+p.z;
}
fn coarse_unpack(row: CoarsePacked) -> CoarseMeasurements {
    let w = row.words;
    var result: CoarseMeasurements;
    result.counts = vec4u(w[0]&65535u,w[0]>>16u,w[1],0u);
    result.depth = bitcast<vec4f>(vec4u(w[2],w[3],w[4],w[5]));
    result.diagonal = vec4f(bitcast<vec3f>(vec3u(w[6],w[7],w[8])),0.0);
    result.cross_moment = vec4f(bitcast<vec3f>(vec3u(w[9],w[10],w[11])),0.0);
    result.normal = vec4f(bitcast<vec3f>(vec3u(w[12],w[13],w[14])),0.0);
    for (var i = 0u; i < 8u; i += 1u) { result.materials[i] = (w[15u+i/2u] >> ((i%2u)*16u)) & 65535u; }
    return result;
}
