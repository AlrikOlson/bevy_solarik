@group(0) @binding(7) var<storage,read> spatial_materials:array<CoarseSurface>;
@compute @workgroup_size(64)
fn material_view(@builtin(global_invocation_id) id:vec3u) {
    if id.x>=arrayLength(&rays) {return;}
    let hit=spatial_trace(rays[id.x]);var material:CoarseSurface;
    if hit.depth.x>0.0 {material=spatial_materials[hit.first.w];}
    let hit_data=vec4u(u32(hit.depth.x),hit.counts.w,hit.first.x,bitcast<u32>(hit.depth.y));
    output[id.x]=Result(material.color,bitcast<vec4u>(material.normal),
        bitcast<vec4u>(material.surface),bitcast<vec4f>(hit_data));
}
