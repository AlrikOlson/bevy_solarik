struct MaterialRequest { part: vec4u, uv: vec4f }
struct MaterialProbe { base: vec4f, normal: vec4f, mr: vec4f, sample: CoarseSurface }
@group(2) @binding(0) var<storage,read> material_requests: array<MaterialRequest>;
@group(2) @binding(1) var<storage,read_write> material_output: array<MaterialProbe>;
@compute @workgroup_size(64)
fn sample_material(@builtin(global_invocation_id) id: vec3u) {
    if id.x>=arrayLength(&material_requests) { return; }
    let request=material_requests[id.x]; let part=request.part.x;
    let v=appearance_vertices[request.part.y];
    let base=textureSampleLevel(base_images,alpha_sampler,request.uv.xy,i32(part),0.0);
    let normal=textureSampleLevel(normal_images,alpha_sampler,request.uv.xy,i32(part),0.0);
    let mr=textureSampleLevel(mr_images,alpha_sampler,request.uv.xy,i32(part),0.0);
    material_output[id.x]=MaterialProbe(base,normal,mr,
        coarse_surface(materials[part],base,normal,mr,v.normal.xyz,v.tangent));
}
