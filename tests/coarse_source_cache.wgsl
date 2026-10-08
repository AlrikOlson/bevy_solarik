// Offline coarse source cook. Original source triangles, alpha and sidedness.
// Finite samples do not prove that a candidate cell is empty.
enable wgpu_ray_query;
struct Task { centre_half: vec4f, direction_side: vec4f }
struct Result {
    counts: vec4u,
    depth: vec4f,
    diagonal: vec4f,
    cross_moment: vec4f,
    normal: vec4f,
    materials: array<u32, 8>,
}
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> uvs: array<vec2f>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> parts: array<vec4u>;
@group(0) @binding(5) var<storage, read> tasks: array<Task>;
@group(0) @binding(6) var alpha_image: texture_2d<f32>;
@group(0) @binding(7) var alpha_sampler: sampler;
@group(0) @binding(8) var<storage, read_write> output: array<Result>;

fn box_interval(origin: vec3f, direction: vec3f, half_extent: f32) -> vec2f {
    var low = 0.0;
    var high = 1e10;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if abs(direction[axis]) < 1e-8 {
            if abs(origin[axis]) > half_extent { return vec2f(1.0, 0.0); }
        } else {
            let a = (-half_extent-origin[axis])/direction[axis];
            let b = (half_extent-origin[axis])/direction[axis];
            low = max(low, min(a,b));
            high = min(high, max(a,b));
        }
    }
    return vec2f(low,high);
}

fn sample_cell(task: Task, offset: vec3f, result: ptr<function, Result>) {
    let centre = task.centre_half.xyz;
    let half_extent = task.centre_half.w;
    let direction = task.direction_side.xyz;
    let distance = half_extent*3.0;
    let local_origin = offset-direction*distance;
    let interval = box_interval(local_origin, direction, half_extent);
    if interval.y <= interval.x { return; }
    (*result).counts.x += 1u;
    var query: ray_query;
    // Hardware traversal may exclude hits exactly at t_min/t_max. Widen only
    // the query; the original half-open spatial ownership below is authoritative.
    let margin = max(half_extent*1e-4, 1e-6);
    rayQueryInitialize(&query, scene, RayDesc(0u, 255u, max(0.0, interval.x-margin),
        interval.y+margin, centre+local_origin, direction));
    var candidates = 0u;
    var nearest = interval.y;
    var farthest = interval.x;
    var first_part = 0xffffffffu;
    var first_triangle = 0xffffffffu;
    var first_normal = vec3f(0.0);
    while rayQueryProceed(&query) {
        candidates += 1u;
        if candidates > 4096u {
            (*result).counts.w = 1u;
            rayQueryTerminate(&query);
            break;
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE { continue; }
        let part_id = hit.instance_custom_data;
        if part_id >= 8u { (*result).counts.w = 2u; continue; }
        let part = parts[part_id];
        if part.w == 0u && !hit.front_face { continue; }
        let offset_index = part.x+hit.primitive_index*3u;
        let a = indices[offset_index]; let b = indices[offset_index+1u]; let c = indices[offset_index+2u];
        let weights = vec3f(1.0-hit.barycentrics.x-hit.barycentrics.y, hit.barycentrics);
        let point = weights.x*positions[a].xyz+weights.y*positions[b].xyz+weights.z*positions[c].xyz;
        // Half-open ownership avoids duplicating a surface on two cell faces.
        if any(point < centre-vec3f(half_extent)) || any(point >= centre+vec3f(half_extent)) { continue; }
        let cutoff = bitcast<f32>(part.z);
        if cutoff >= 0.0 {
            let uv = weights.x*uvs[a]+weights.y*uvs[b]+weights.z*uvs[c];
            if textureSampleLevel(alpha_image, alpha_sampler, uv, 0.0).a < cutoff { continue; }
        }
        (*result).counts.z += 1u;
        farthest = max(farthest,hit.t);
        if first_part == 0xffffffffu || hit.t < nearest
            || (hit.t == nearest && (part_id < first_part
                || (part_id == first_part && hit.primitive_index < first_triangle))) {
            nearest = hit.t;
            first_part = part_id;
            first_triangle = hit.primitive_index;
            first_normal = normalize(cross(positions[b].xyz-positions[a].xyz, positions[c].xyz-positions[a].xyz));
        }
    }
    if first_part == 0xffffffffu { return; }
    (*result).counts.y += 1u;
    (*result).materials[first_part] += 1u;
    let projection_radius = half_extent*dot(abs(direction),vec3f(1.0));
    let first = (nearest-distance)/(2.0*projection_radius)+0.5;
    let last = (farthest-distance)/(2.0*projection_radius)+0.5;
    (*result).depth.x = min((*result).depth.x,first);
    (*result).depth.y = max((*result).depth.y,last);
    (*result).depth.z += first;
    (*result).depth.w += last;
    let n = first_normal;
    (*result).diagonal += vec4f(n*n,0.0);
    (*result).cross_moment += vec4f(n.x*n.y,n.x*n.z,n.y*n.z,0.0);
    (*result).normal += vec4f(n,0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&output) { return; }
    let task = tasks[id.x];
    let d = task.direction_side.xyz;
    var axis = vec3f(1.0,0.0,0.0);
    if abs(d.y) < abs(d.x) { axis = vec3f(0.0,1.0,0.0); }
    if abs(d.z) < min(abs(d.x),abs(d.y)) { axis = vec3f(0.0,0.0,1.0); }
    let right = normalize(cross(d,axis));
    let up = cross(right,d);
    let extent = task.centre_half.w*vec2f(dot(abs(right),vec3f(1.0)),dot(abs(up),vec3f(1.0)));
    let side = u32(task.direction_side.w);
    var result: Result;
    result.depth.x = 1.0;
    for (var y = 0u; y < side; y += 1u) {
        for (var x = 0u; x < side; x += 1u) {
            let uv = (vec2f(f32(x),f32(y))+0.5)/f32(side)*2.0-1.0;
            sample_cell(task,right*uv.x*extent.x+up*uv.y*extent.y,&result);
        }
    }
    if result.counts.y == 0u { result.depth = vec4f(0.0); }
    output[id.x] = result;
}
