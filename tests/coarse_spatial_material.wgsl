// Exact original ray selection must reproduce every sealed SFSP0002 hit word.
struct Task { centre_half:vec4f, direction_side:vec4f }
struct Sample { surface:CoarseSurface, hit_data:vec4u }
@group(0) @binding(0) var scene:acceleration_structure;
@group(0) @binding(1) var<storage,read> positions:array<vec4f>;
@group(0) @binding(2) var<storage,read> uvs:array<vec2f>;
@group(0) @binding(3) var<storage,read> indices:array<u32>;
@group(0) @binding(4) var<storage,read> parts:array<vec4u>;
@group(0) @binding(5) var<storage,read> tasks:array<Task>;
@group(0) @binding(6) var alpha_image:texture_2d<f32>;
@group(0) @binding(7) var alpha_sampler:sampler;
@group(0) @binding(8) var<storage,read_write> output:array<Sample>;
struct AppearanceVertex {normal:vec4f,tangent:vec4f}
@group(1) @binding(0) var<storage,read> appearance_vertices:array<AppearanceVertex>;
@group(1) @binding(1) var<storage,read> materials:array<CoarseMaterial>;
@group(1) @binding(2) var base_images:texture_2d_array<f32>;
@group(1) @binding(3) var normal_images:texture_2d_array<f32>;
@group(1) @binding(4) var mr_images:texture_2d_array<f32>;
fn source_sample(origin:vec3f,d:vec3f,limits:vec2f,centre:vec3f,half_extent:f32,axis:u32,spatial:bool)->Sample {
    var result:Sample;
    var query:ray_query;
    rayQueryInitialize(&query,scene,RayDesc(0u,255u,limits.x,limits.y,origin,d));
    var candidates=0u;var nearest=limits.y;
    var first_part=0xffffffffu;var first_triangle=0xffffffffu;
    var first_indices=vec3u(0u);var first_weights=vec3f(0.0);var first_word=0u;
    while rayQueryProceed(&query) {
        candidates+=1u;
        if candidates>4096u {rayQueryTerminate(&query);result.hit_data.y=1u;return result;}
        let hit=rayQueryGetCandidateIntersection(&query);
        if hit.kind!=RAY_QUERY_INTERSECTION_TRIANGLE {continue;}
        let part_id=hit.instance_custom_data;
        if part_id>=arrayLength(&materials) {result.hit_data.y=2u;return result;}
        let part=parts[part_id];
        if part.w==0u && !hit.front_face {continue;}
        let start=part.x+hit.primitive_index*3u;
        let a=indices[start];let b=indices[start+1u];let c=indices[start+2u];
        let weights=vec3f(1.0-hit.barycentrics.x-hit.barycentrics.y,hit.barycentrics);
        let point=weights.x*positions[a].xyz+weights.y*positions[b].xyz+weights.z*positions[c].xyz;
        if spatial && (any(point<centre-vec3f(half_extent)) || any(point>=centre+vec3f(half_extent))) {continue;}
        let cutoff=bitcast<f32>(part.z);
        let uv=weights.x*uvs[a]+weights.y*uvs[b]+weights.z*uvs[c];
        if cutoff>=0.0 && textureSampleLevel(alpha_image,alpha_sampler,uv,0.0).a<cutoff {continue;}
        var word=1u;
        if spatial {
            let depth=(point[axis]-centre[axis])/(half_extent*2.0)+0.5;
            let normal=normalize(cross(positions[b].xyz-positions[a].xyz,positions[c].xyz-positions[a].xyz));
            word=spatial_pack(depth,normal,part_id);
            if spatial_axis(spatial_normal(word))!=axis {continue;}
        }
        if first_part==0xffffffffu || hit.t<nearest || (hit.t==nearest &&
            (part_id<first_part || (part_id==first_part && hit.primitive_index<first_triangle))) {
            nearest=hit.t;first_part=part_id;first_triangle=hit.primitive_index;
            first_word=word;first_indices=vec3u(a,b,c);first_weights=weights;
        }
    }
    if first_part==0xffffffffu {return result;}
    result.hit_data=vec4u(first_word,0u,first_part+1u,bitcast<u32>(nearest));
    let a=first_indices.x;let b=first_indices.y;let c=first_indices.z;let w=first_weights;
    let uv=w.x*uvs[a]+w.y*uvs[b]+w.z*uvs[c];
    let n=w.x*appearance_vertices[a].normal.xyz+w.y*appearance_vertices[b].normal.xyz+w.z*appearance_vertices[c].normal.xyz;
    let t=w.x*appearance_vertices[a].tangent.xyz+w.y*appearance_vertices[b].tangent.xyz+w.z*appearance_vertices[c].tangent.xyz;
    if dot(n,n)<1e-20 || dot(t,t)<1e-20 {result.hit_data.y=4u;return result;}
    result.surface=coarse_surface(materials[first_part],
        textureSampleLevel(base_images,alpha_sampler,uv,i32(first_part),0.0),
        textureSampleLevel(normal_images,alpha_sampler,uv,i32(first_part),0.0),
        textureSampleLevel(mr_images,alpha_sampler,uv,i32(first_part),0.0),
        n,vec4f(t,appearance_vertices[a].tangent.w));
    return result;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3u) {
    if id.x>=arrayLength(&output) {return;}
    let task=tasks[id.x/64u];let texel=id.x%64u;
    let centre=task.centre_half.xyz;let half_extent=task.centre_half.w;
    let d=task.direction_side.xyz;let axis=spatial_axis(d);
    var offset=vec3f(0.0);
    offset[(axis+1u)%3u]=((f32(texel%8u)+0.5)/8.0*2.0-1.0)*half_extent;
    offset[(axis+2u)%3u]=((f32(texel/8u)+0.5)/8.0*2.0-1.0)*half_extent;
    let origin=centre+offset-d*half_extent*3.0;
    let margin=max(half_extent*1e-4,1e-6);
    output[id.x]=source_sample(origin,d,vec2f(max(0.0,half_extent*2.0-margin),half_extent*4.0+margin),
        centre,half_extent,axis,true);
}
@compute @workgroup_size(64)
fn reference(@builtin(global_invocation_id) id:vec3u) {
    if id.x>=arrayLength(&output) {return;}
    let task=tasks[id.x];
    output[id.x]=source_sample(task.centre_half.xyz,task.direction_side.xyz,
        vec2f(task.centre_half.w,task.direction_side.w),vec3f(0.0),1.0,0u,false);
}
