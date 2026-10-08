// Offline reference only. Original source triangles and explicit level-zero masks.
// Vulkan ray traversal: no confirmations prune the candidate interval; the
// NO_DUPLICATE_ANY_HIT_INVOCATION geometry flag prevents duplicated events.
enable wgpu_ray_query;
struct Ray { origin: vec4f, direction: vec4f }
struct Result {
    depth: vec4f,
    counts: vec4u,
    first: vec4u,
    normal: vec4f,
}
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> uvs: array<vec2f>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> parts: array<vec4u>;
@group(0) @binding(5) var<storage, read> rays: array<Ray>;
@group(0) @binding(6) var alpha_image: texture_2d<f32>;
@group(0) @binding(7) var alpha_sampler: sampler;
@group(0) @binding(8) var<storage, read_write> output: array<Result>;

fn trace(ray: Ray) -> Result {
    var result: Result;
    result.depth.x = ray.direction.w;
    var query: ray_query;
    rayQueryInitialize(&query, scene, RayDesc(0u, 255u, ray.origin.w,
        ray.direction.w, ray.origin.xyz, ray.direction.xyz));
    while rayQueryProceed(&query) {
        result.counts.x += 1u;
        if result.counts.x > 4096u {
            result.counts.w = 1u;
            rayQueryTerminate(&query);
            break;
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE { continue; }
        let part = parts[hit.instance_custom_data];
        if part.w == 0u && !hit.front_face { continue; }
        let offset = part.x + hit.primitive_index * 3u;
        let a = indices[offset]; let b = indices[offset+1u]; let c = indices[offset+2u];
        let cutoff = bitcast<f32>(part.z);
        if cutoff >= 0.0 {
            let weights = vec3f(1.0-hit.barycentrics.x-hit.barycentrics.y, hit.barycentrics);
            let uv = weights.x*uvs[a]+weights.y*uvs[b]+weights.z*uvs[c];
            if textureSampleLevel(alpha_image, alpha_sampler, uv, 0.0).a < cutoff { continue; }
            result.counts.z += 1u;
        }
        result.counts.y += 1u;
        let n = normalize(cross(positions[b].xyz-positions[a].xyz, positions[c].xyz-positions[a].xyz));
        result.depth.z += abs(dot(n, ray.direction.xyz));
        result.depth.y = max(result.depth.y, hit.t);
        if hit.t < result.depth.x {
            result.depth.x = hit.t;
            result.first = vec4u(hit.instance_custom_data+1u, hit.primitive_index, u32(hit.front_face), 0u);
            result.normal = vec4f(n, 0.0);
        }
    }
    if result.counts.y == 0u { result.depth = vec4f(0.0); }
    return result;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x < arrayLength(&rays) { output[id.x] = trace(rays[id.x]); }
}
