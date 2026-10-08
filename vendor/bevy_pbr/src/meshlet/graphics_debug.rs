//! Read-only production VG resource handles for opt-in native capture.
use super::{
    instance_manager::InstanceManager,
    resource_manager::{MeshletViewResources, ResourceManager},
};
use alloc::vec::Vec;
use bevy_ecs::world::World;
use bevy_render::render_resource::Buffer;

/// Return bounded-readback sources; callers own staging and exact frame correlation.
pub fn buffers(world: &mut World) -> Vec<(&'static str, Buffer)> {
    let mut result = Vec::new();
    if let Some(instances) = world.get_resource::<InstanceManager>() {
        for (role, buffer) in [
            ("vg_instance_uniforms", instances.instance_uniforms.buffer()),
            ("vg_active_indices", instances.active_indices.buffer()),
            ("vg_material_ids", instances.instance_material_ids.buffer()),
            ("vg_cutouts", instances.instance_cutouts.buffer()),
            ("vg_root_nodes", instances.instance_bvh_root_nodes.buffer()),
        ] {
            if let Some(buffer) = buffer {
                result.push((role, buffer.clone()));
            }
        }
    }
    if let Some(resources) = world.get_resource::<ResourceManager>() {
        result.push((
            "vg_raster_previous_counts",
            resources
                .visibility_buffer_raster_cluster_prev_counts
                .clone(),
        ));
    }
    for view in world.query::<&MeshletViewResources>().iter(world) {
        result.extend([
            ("vg_second_instance_count", view.second_pass_count.clone()),
            (
                "vg_front_cluster_count",
                view.front_meshlet_cull_count.clone(),
            ),
            (
                "vg_back_cluster_count",
                view.back_meshlet_cull_count.clone(),
            ),
            (
                "vg_software_dispatch",
                view.visibility_buffer_software_raster_indirect_args.clone(),
            ),
            (
                "vg_hardware_draw",
                view.visibility_buffer_hardware_raster_indirect_args.clone(),
            ),
        ]);
    }
    result
}
