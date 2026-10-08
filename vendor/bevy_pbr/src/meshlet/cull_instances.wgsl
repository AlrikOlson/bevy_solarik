#import bevy_pbr::meshlet_bindings::{
    InstancedOffset, MeshletAabb, constants,
    meshlet_view_instance_visibility, meshlet_instance_aabbs,
    meshlet_instance_bvh_root_nodes, meshlet_bvh_cull_count_write,
    meshlet_bvh_cull_dispatch, meshlet_bvh_cull_queue,
    meshlet_second_pass_instance_count, meshlet_second_pass_instance_dispatch,
    meshlet_second_pass_instance_candidates, meshlet_assembly_groups,
    meshlet_assembly_members,
}
#import bevy_pbr::meshlet_cull_shared::{aabb_in_frustum, should_occlusion_cull_aabb}

const GROUP_BIT = 0x80000000u;

fn should_cull_instance(instance_id: u32) -> bool {
    let packed = meshlet_view_instance_visibility[instance_id >> 5u];
    return bool(extractBits(packed, instance_id & 31u, 1u));
}
#ifdef MESHLET_FIRST_CULLING_PASS
fn defer_candidate(candidate: u32) {
    // Each root contributes either one group or at most its member count.
    // Thus the existing live-part-sized queue is a strict upper bound.
    let id = atomicAdd(&meshlet_second_pass_instance_count, 1u);
    meshlet_second_pass_instance_candidates[id] = candidate;
    if (id & 127u) == 0u { atomicAdd(&meshlet_second_pass_instance_dispatch.x, 1u); }
}
#endif
fn cull_part(instance_id: u32) {
    let aabb = meshlet_instance_aabbs[instance_id];
    if should_cull_instance(instance_id) || !aabb_in_frustum(aabb, instance_id) { return; }
    if should_occlusion_cull_aabb(aabb, instance_id) {
#ifdef MESHLET_FIRST_CULLING_PASS
        defer_candidate(instance_id);
#endif
        return;
    }
    let id = atomicAdd(&meshlet_bvh_cull_count_write, 1u);
    meshlet_bvh_cull_queue[id] = InstancedOffset(instance_id, meshlet_instance_bvh_root_nodes[instance_id]);
    if (id & 15u) == 0u { atomicAdd(&meshlet_bvh_cull_dispatch.x, 1u); }
}
fn cull_group(group_id: u32) {
    let group = meshlet_assembly_groups[group_id];
    let count = group.count & ~GROUP_BIT;
    if count == 0u { return; }
    let anchor = meshlet_assembly_members[group.first];
    if (group.count & GROUP_BIT) == 0u {
        let aabb = MeshletAabb(group.center, group.half_extent);
        if should_cull_instance(anchor) || !aabb_in_frustum(aabb, anchor) { return; }
        if should_occlusion_cull_aabb(aabb, anchor) {
#ifdef MESHLET_FIRST_CULLING_PASS
            defer_candidate(group_id | GROUP_BIT);
#endif
            return;
        }
    }
    for (var member = 0u; member < count; member++) {
        cull_part(meshlet_assembly_members[group.first + member]);
    }
}
@compute
@workgroup_size(128, 1, 1)
fn cull_instances(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x;
#ifdef MESHLET_FIRST_CULLING_PASS
    if id >= constants.scene_instance_count { return; }
    cull_group(id);
#else
    if id >= meshlet_second_pass_instance_count { return; }
    let candidate = meshlet_second_pass_instance_candidates[id];
    if (candidate & GROUP_BIT) != 0u { cull_group(candidate & ~GROUP_BIT); }
    else { cull_part(candidate); }
#endif
}
