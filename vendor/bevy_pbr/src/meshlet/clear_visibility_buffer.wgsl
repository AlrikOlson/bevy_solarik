#ifdef MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT
@group(0) @binding(0) var meshlet_visibility_buffer: texture_storage_2d<r64uint, write>;
#else
@group(0) @binding(0) var meshlet_visibility_buffer: texture_storage_2d<r32uint, write>;
#endif
@group(0) @binding(1) var<storage, read_write> meshlet_early_depth: array<atomic<u32>>;
var<immediate> view_size: vec2<u32>;

@compute
@workgroup_size(16, 16, 1)
fn clear_visibility_buffer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= view_size) { return; }
#ifdef MESHLET_EARLY_VISIBILITY_TEST
    let index = global_id.y * view_size.x + global_id.x;
    if index < arrayLength(&meshlet_early_depth) { atomicStore(&meshlet_early_depth[index], 0u); }
#endif

#ifdef MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT
    textureStore(meshlet_visibility_buffer, global_id.xy, vec4(0lu));
#else
    textureStore(meshlet_visibility_buffer, global_id.xy, vec4(0u));
#endif
}
