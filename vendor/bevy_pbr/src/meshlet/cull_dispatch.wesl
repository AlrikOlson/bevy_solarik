
// Cull queues can contain more work than one device dispatch dimension.
// Both dimensions grow monotonically while producers publish in any order.
struct CullDispatchArgs {
    x: atomic<u32>,
    y: atomic<u32>,
    z: u32,
}

@if(CULL_DISPATCH_WIDTH)
const CULL_WIDTH: u32 = constants::CULL_DISPATCH_WIDTH;
@if(!CULL_DISPATCH_WIDTH)
const CULL_WIDTH: u32 = 65535u;


fn cull_dispatch_size(count: u32, items_per_group: u32) -> vec2<u32> {
    let groups = (count + items_per_group - 1u) / items_per_group;
    return vec2<u32>(min(groups, CULL_WIDTH), groups / CULL_WIDTH + u32(groups % CULL_WIDTH != 0u));
}

fn cull_dispatch_index(global_id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return global_id.x + global_id.y * groups.x * 128u;
}