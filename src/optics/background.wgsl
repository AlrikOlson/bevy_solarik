@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var reference:texture_2d<f32>;
@group(0) @binding(2) var depth:texture_depth_2d;
@group(0) @binding(3) var destination:texture_storage_2d<rgba16float,write>;
@compute @workgroup_size(8,8)
fn save(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(source); if any(id.xy>=size) {return;}
    let uv=(vec2<f32>(id.xy)+0.5)/vec2<f32>(size);
    let depth_size=vec2<i32>(textureDimensions(depth));
    let position=min(vec2<i32>(uv*vec2<f32>(depth_size)),depth_size-1);
    var d=0.0;
    // Conservative two-pixel reconstruction footprint: fractional coverage
    // around a silhouette must stay with the geometry-guided result, avoiding
    // a hard mask seam between anti-aliased radiance and jittered point depth.
    for(var y=-2;y<=2;y++) {for(var x=-2;x<=2;x++) {
        d=max(d,textureLoad(depth,clamp(position+vec2(x,y),vec2(0),depth_size-1),0));
    }}
    // Reverse-Z zero means no rasterized surface. These pixels have no
    // geometric guide for neural relighting; retain their measured radiance.
    textureStore(destination,id.xy,vec4(textureLoad(source,id.xy,0).rgb,select(0.0,1.0,d==0.0)));
}
@compute @workgroup_size(8,8)
fn restore(@builtin(global_invocation_id) id:vec3<u32>) {
    let size=textureDimensions(source); if any(id.xy>=size) {return;}
    let original=textureLoad(reference,id.xy,0);
    let rendered=textureLoad(source,id.xy,0);
    textureStore(destination,id.xy,vec4(select(rendered.rgb,original.rgb,original.a>0.5),rendered.a));
}
