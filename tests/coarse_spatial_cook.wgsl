struct Task { centre_half: vec4f, direction_side: vec4f }
struct Result { samples: array<u32,64> }
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage,read> positions: array<vec4f>;
@group(0) @binding(2) var<storage,read> uvs: array<vec2f>;
@group(0) @binding(3) var<storage,read> indices: array<u32>;
@group(0) @binding(4) var<storage,read> parts: array<vec4u>;
@group(0) @binding(5) var<storage,read> tasks: array<Task>;
@group(0) @binding(6) var alpha_image: texture_2d<f32>;
@group(0) @binding(7) var alpha_sampler: sampler;
@group(0) @binding(8) var<storage,read_write> output: array<Result>;
fn sample_spatial(task: Task, texel: u32) -> u32 {
    let centre=task.centre_half.xyz; let half_extent=task.centre_half.w;
    let d=task.direction_side.xyz; let axis=spatial_axis(d);
    var offset=vec3f(0.0);
    offset[(axis+1u)%3u]=((f32(texel%8u)+0.5)/8.0*2.0-1.0)*half_extent;
    offset[(axis+2u)%3u]=((f32(texel/8u)+0.5)/8.0*2.0-1.0)*half_extent;
    let origin=centre+offset-d*half_extent*3.0;
    let margin=max(half_extent*1e-4,1e-6);
    var query: ray_query;
    rayQueryInitialize(&query,scene,RayDesc(0u,255u,max(0.0,half_extent*2.0-margin),
        half_extent*4.0+margin,origin,d));
    var candidates=0u; var nearest=half_extent*4.0;
    var first_part=0xffffffffu; var first_triangle=0xffffffffu;
    var first_word=0u;
    while rayQueryProceed(&query) {
        candidates+=1u;
        if candidates>4096u { rayQueryTerminate(&query); return 0x1000u; }
        let hit=rayQueryGetCandidateIntersection(&query);
        if hit.kind!=RAY_QUERY_INTERSECTION_TRIANGLE { continue; }
        let part_id=hit.instance_custom_data;
        if part_id>=4u { return 0x1000u; }
        let part=parts[part_id];
        if part.w==0u && !hit.front_face { continue; }
        let start=part.x+hit.primitive_index*3u;
        let a=indices[start]; let b=indices[start+1u]; let c=indices[start+2u];
        let weights=vec3f(1.0-hit.barycentrics.x-hit.barycentrics.y,hit.barycentrics);
        let point=weights.x*positions[a].xyz+weights.y*positions[b].xyz+weights.z*positions[c].xyz;
        if any(point<centre-vec3f(half_extent)) || any(point>=centre+vec3f(half_extent)) { continue; }
        let cutoff=bitcast<f32>(part.z);
        if cutoff>=0.0 {
            let uv=weights.x*uvs[a]+weights.y*uvs[b]+weights.z*uvs[c];
            if textureSampleLevel(alpha_image,alpha_sampler,uv,0.0).a<cutoff { continue; }
        }
        let depth=(point[axis]-centre[axis])/(half_extent*2.0)+0.5;
        let normal=normalize(cross(positions[b].xyz-positions[a].xyz,positions[c].xyz-positions[a].xyz));
        let word=spatial_pack(depth,normal,part_id);
        if spatial_axis(spatial_normal(word))!=axis { continue; }
        if first_part==0xffffffffu || hit.t<nearest || (hit.t==nearest &&
            (part_id<first_part || (part_id==first_part && hit.primitive_index<first_triangle))) {
            nearest=hit.t; first_part=part_id; first_triangle=hit.primitive_index;
            first_word=word;
        }
    }
    if first_part==0xffffffffu { return 0u; }
    return first_word;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let task=id.x/64u; let texel=id.x%64u;
    if task<arrayLength(&tasks) { output[task].samples[texel]=sample_spatial(tasks[task],texel); }
}
