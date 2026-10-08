struct PlacementProbe {
    root_height: vec4<f32>,
    rotation: vec4<f32>,
    prototype: vec4<u32>,
    points: array<vec4<f32>, 8>,
}
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> pages: array<PlacementPage>;
@group(0) @binding(2) var<storage, read> radii: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<PlacementProbe>;

fn write_probe(id: u32, placement: DecodedPlacement) {
    results[id].root_height = vec4(placement.root, placement.height);
    results[id].rotation = placement.rotation;
    results[id].prototype = vec4(placement.prototype, 0u, 0u, 0u);
    let points = array<vec3<f32>, 8>(
        vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
        vec3(0.5, 0.5, 0.5), vec3(-0.5, 0.5, -0.5),
    );
    for (var i = 0u; i < 8u; i += 1u) {
        results[id].points[i] = vec4(placement_transform(placement, points[i] * radii[id]), 1.0);
    }
}
@compute @workgroup_size(64)
fn probe12(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&pages) { return; }
    let offset = id.x * 3u;
    write_probe(id.x, placement_decode12(words[offset], words[offset+1u], words[offset+2u], pages[id.x]));
}
@compute @workgroup_size(64)
fn probe16(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&pages) { return; }
    let offset = id.x * 4u;
    write_probe(id.x, placement_decode16(
        words[offset], words[offset+1u], words[offset+2u], words[offset+3u], pages[id.x],
    ));
}
