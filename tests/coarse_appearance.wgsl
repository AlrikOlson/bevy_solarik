// Offline coarse source cook. Original source triangles, alpha and sidedness.
// Finite samples do not prove that a candidate cell is empty.

struct Task { centre_half: vec4f, direction_side: vec4f }
struct Result { moments: array<CoarseAppearanceMoment,8> }
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> uvs: array<vec2f>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> parts: array<vec4u>;
@group(0) @binding(5) var<storage, read> tasks: array<Task>;
@group(0) @binding(6) var alpha_image: texture_2d<f32>;
@group(0) @binding(7) var alpha_sampler: sampler;
@group(0) @binding(8) var<storage, read_write> output: array<Result>;
struct AppearanceVertex { normal: vec4f, tangent: vec4f }
@group(1) @binding(0) var<storage,read> appearance_vertices: array<AppearanceVertex>;
@group(1) @binding(1) var<storage,read> materials: array<CoarseMaterial>;
@group(1) @binding(2) var base_images: texture_2d_array<f32>;
@group(1) @binding(3) var normal_images: texture_2d_array<f32>;
@group(1) @binding(4) var mr_images: texture_2d_array<f32>;
fn appearance_sample(part: u32, uv: vec2f, n: vec3f, t: vec4f) -> CoarseSurface {
    return coarse_surface(materials[part],
        textureSampleLevel(base_images,alpha_sampler,uv,i32(part),0.0),
        textureSampleLevel(normal_images,alpha_sampler,uv,i32(part),0.0),
        textureSampleLevel(mr_images,alpha_sampler,uv,i32(part),0.0),n,t);
}

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

    var query: ray_query;
    // Hardware traversal may exclude hits exactly at t_min/t_max. Widen only
    // the query; the original half-open spatial ownership below is authoritative.
    let margin = max(half_extent*1e-4, 1e-6);
    rayQueryInitialize(&query, scene, RayDesc(0u, 255u, max(0.0, interval.x-margin),
        interval.y+margin, centre+local_origin, direction));
    var candidates = 0u;
    var nearest = interval.y;

    var first_part = 0xffffffffu;
    var first_triangle = 0xffffffffu;
    var first_indices=vec3u(0u);
    var first_weights=vec3f(0.0);
    while rayQueryProceed(&query) {
        candidates += 1u;
        if candidates > 4096u {
            (*result).moments[0].counts.y = 1u;
            rayQueryTerminate(&query);
            break;
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE { continue; }
        let part_id = hit.instance_custom_data;
        if part_id >= arrayLength(&materials) { (*result).moments[0].counts.y = 2u; continue; }
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

        if first_part == 0xffffffffu || hit.t < nearest
            || (hit.t == nearest && (part_id < first_part
                || (part_id == first_part && hit.primitive_index < first_triangle))) {
            nearest = hit.t;
            first_part = part_id;
            first_triangle = hit.primitive_index;
            first_indices=vec3u(a,b,c);
            first_weights=weights;
        }
    }
    if first_part == 0xffffffffu { return; }
    let a=first_indices.x; let b=first_indices.y; let c=first_indices.z;
    let weights=first_weights;
    let uv=weights.x*uvs[a]+weights.y*uvs[b]+weights.z*uvs[c];
    let n=weights.x*appearance_vertices[a].normal.xyz+weights.y*appearance_vertices[b].normal.xyz
        +weights.z*appearance_vertices[c].normal.xyz;
    let t=weights.x*appearance_vertices[a].tangent.xyz+weights.y*appearance_vertices[b].tangent.xyz
        +weights.z*appearance_vertices[c].tangent.xyz;
    if dot(n,n)<1e-20 || dot(t,t)<1e-20 {
        (*result).moments[first_part].counts.y=4u; return;
    }
    let sample=appearance_sample(first_part,uv,n,vec4f(t,appearance_vertices[a].tangent.w));
    var moment=(*result).moments[first_part];
    coarse_accumulate(&moment,sample);
    (*result).moments[first_part]=moment;
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

    for (var y = 0u; y < side; y += 1u) {
        for (var x = 0u; x < side; x += 1u) {
            let uv = (vec2f(f32(x),f32(y))+0.5)/f32(side)*2.0-1.0;
            sample_cell(task,right*uv.x*extent.x+up*uv.y*extent.y,&result);
        }
    }

    output[id.x] = result;
}
