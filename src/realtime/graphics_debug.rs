//! Bounded lighting counter handles for opt-in native capture readbacks.
use alloc::vec::Vec;
use bevy_ecs::world::World;
use bevy_render::render_resource::Buffer;

/// Return production cache counters and indirect dispatches, without changing lighting.
pub fn buffers(world: &mut World) -> Vec<(&'static str, Buffer)> {
    let mut result = Vec::new();
    for resources in world
        .query::<&super::prepare::SolarikLightingResources>()
        .iter(world)
    {
        result.extend([
            (
                "cache_active_count",
                resources.world_cache_active_cells_count.clone(),
            ),
            ("cache_work_statistics", resources.world_cache_b.clone()),
            (
                "cache_dispatch",
                resources.world_cache_active_cells_dispatch.clone(),
            ),
        ]);
    }
    result
}
