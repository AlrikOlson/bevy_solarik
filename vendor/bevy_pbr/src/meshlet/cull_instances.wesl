import bevy_pbr::meshlet::bindings::InstancedOffset;
import bevy_pbr::meshlet::bindings::MeshletAabb;
import bevy_pbr::meshlet::bindings::constants;
import bevy_pbr::meshlet::bindings::meshlet_view_instance_visibility;
import bevy_pbr::meshlet::bindings::meshlet_instance_aabbs;
import bevy_pbr::meshlet::bindings::meshlet_instance_bvh_root_nodes;
import bevy_pbr::meshlet::bindings::meshlet_bvh_cull_count_write;
import bevy_pbr::meshlet::bindings::meshlet_bvh_cull_dispatch;
import bevy_pbr::meshlet::bindings::meshlet_bvh_cull_queue;
import bevy_pbr::meshlet::bindings::meshlet_second_pass_instance_count;
import bevy_pbr::meshlet::bindings::meshlet_second_pass_instance_dispatch;
import bevy_pbr::meshlet::bindings::meshlet_second_pass_instance_candidates;
import bevy_pbr::meshlet::bindings::meshlet_assembly_groups;
import bevy_pbr::meshlet::bindings::meshlet_assembly_members;
import bevy_pbr::meshlet::cull_shared::aabb_in_frustum;
import bevy_pbr::meshlet::cull_shared::should_occlusion_cull_aabb;
import bevy_pbr::meshlet::cull_dispatch::cull_dispatch_size;
import bevy_pbr::meshlet::cull_dispatch::cull_dispatch_index;


const GROUP_BIT = 0x80000000u;

fn should_cull_instance(instance_id: u32) -> bool {
    let packed = meshlet_view_instance_visibility[instance_id >> 5u];
    return bool(extractBits(packed, instance_id & 31u, 1u));
}
@if((MESHLET_FIRST_CULLING_PASS))
fn defer_candidate(candidate: u32) {
    // Each root contributes either one group or at most its member count.
    // Thus the existing live-part-sized queue is a strict upper bound.
    let id = atomicAdd(&meshlet_second_pass_instance_count, 1u);
    meshlet_second_pass_instance_candidates[id] = candidate;
    if (id & 127u) == 0u {
        let size = cull_dispatch_size(id + 1u, 128u);
        atomicMax(&meshlet_second_pass_instance_dispatch.x, size.x);
        atomicMax(&meshlet_second_pass_instance_dispatch.y, size.y);
    }
}
fn cull_part(instance_id: u32) {
    let aabb = meshlet_instance_aabbs[instance_id];
    if should_cull_instance(instance_id) || !aabb_in_frustum(aabb, instance_id) { return; }
    if should_occlusion_cull_aabb(aabb, instance_id) {
@if(MESHLET_FIRST_CULLING_PASS) {
        defer_candidate(instance_id);
}
        return;
    }
    let id = atomicAdd(&meshlet_bvh_cull_count_write, 1u);
    meshlet_bvh_cull_queue[id] = InstancedOffset(instance_id, meshlet_instance_bvh_root_nodes[instance_id]);
    if (id & 15u) == 0u {
        let size = cull_dispatch_size(id + 1u, 16u);
        atomicMax(&meshlet_bvh_cull_dispatch.x, size.x);
        atomicMax(&meshlet_bvh_cull_dispatch.y, size.y);
    }
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
@if(MESHLET_FIRST_CULLING_PASS) {
            defer_candidate(group_id | GROUP_BIT);
}
            return;
        }
    }
    for (var member = 0u; member < count; member++) {
        cull_part(meshlet_assembly_members[group.first + member]);
    }
}
@compute
@workgroup_size(128, 1, 1)
fn cull_instances(@builtin(global_invocation_id) invocation: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let id = cull_dispatch_index(invocation, groups);
@if(MESHLET_FIRST_CULLING_PASS) {
    if id >= constants.scene_instance_count { return; }
    cull_group(id);
}
@if(!(MESHLET_FIRST_CULLING_PASS)) {
    if id >= meshlet_second_pass_instance_count { return; }
    let candidate = meshlet_second_pass_instance_candidates[id];
    if (candidate & GROUP_BIT) != 0u { cull_group(candidate & ~GROUP_BIT); }
    else { cull_part(candidate); }
}
}
