enable wgpu_ray_query;

#define_import_path bevy_solarik::world_cache_priority

#import bevy_solarik::world_cache::WORLD_CACHE_CELL_UPDATES_SOFT_CAP
#import bevy_solarik::realtime_bindings::{
    constants, world_cache_a, world_cache_b,
    world_cache_geometry_data, world_cache_radiance,
    world_cache_active_cell_indices, world_cache_active_cells_count,
}

// Owned bounded scheduler, informed by UE 5.8.3 Lumen's new-probe/age priority.
// Compaction has finished: A is now the selected list, B the small histogram.
// Neither aliases the active list read by parallel invocations.
const PRIORITY_BUCKET_COUNT = 32u;
const PRIORITY_QUOTAS = 32u;
const PRIORITY_CURSORS = 64u;
const PRIORITY_SELECTED_COUNT = 96u;
const PRIORITY_UNLIT_COUNT = 97u;
const PRIORITY_SELECTED_UNLIT = 98u;
const PRIORITY_SELECTED_REFRESH = 99u;
const PRIORITY_OLDEST_AGE = 100u;
const PRIORITY_UPDATED_COUNT = 101u;
const PRIORITY_RESET = 102u;
const PRIORITY_OFFSETS = 128u;
const PRIORITY_SCRATCH_WORDS = 160u;
const PRIORITY_REFRESH_RESERVE_DIVISOR = 5u;

@group(2) @binding(0) var<storage, read_write> world_cache_active_cells_dispatch: vec3<u32>;
var<workgroup> priority_counts: array<atomic<u32>, 32u>;
var<workgroup> priority_bases: array<u32, 32u>;
var<workgroup> priority_oldest: atomic<u32>;

fn selected_world_cache_cell_count() -> u32 {
    return atomicLoad(&world_cache_b[PRIORITY_SELECTED_COUNT]);
}

fn world_cache_cell_age(cell: u32) -> u32 {
    return constants.render_frame - world_cache_geometry_data[cell].last_traced_frame;
}

fn world_cache_priority_bucket(cell: u32) -> u32 {
    if world_cache_radiance[cell].a == 0.0 { return 0u; }
    return 1u + min(firstLeadingBit(max(world_cache_cell_age(cell), 1u)), 30u);
}

@compute @workgroup_size(32)
fn clear_world_cache_priority(@builtin(global_invocation_id) id: vec3<u32>) {
    for (var i = id.x; i < PRIORITY_SCRATCH_WORDS; i += PRIORITY_BUCKET_COUNT) {
        atomicStore(&world_cache_b[i], 0u);
    }
    if id.x == 0u { world_cache_active_cells_dispatch = vec3(0u, 1u, 1u); }
}

@compute @workgroup_size(256)
fn histogram_world_cache_priority(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if local < PRIORITY_BUCKET_COUNT { atomicStore(&priority_counts[local], 0u); }
    if local == 0u { atomicStore(&priority_oldest, 0u); }
    workgroupBarrier();
    if id.x < world_cache_active_cells_count {
        let cell = world_cache_active_cell_indices[id.x];
        let bucket = world_cache_priority_bucket(cell);
        atomicAdd(&priority_counts[bucket], 1u);
        if bucket != 0u { atomicMax(&priority_oldest, world_cache_cell_age(cell)); }
    }
    workgroupBarrier();
    if local < PRIORITY_BUCKET_COUNT {
        let count = atomicLoad(&priority_counts[local]);
        if count != 0u { atomicAdd(&world_cache_b[local], count); }
    }
    if local == 0u { atomicMax(&world_cache_b[PRIORITY_OLDEST_AGE], atomicLoad(&priority_oldest)); }
}

@compute @workgroup_size(1)
fn budget_world_cache_priority() {
    let budget = min(WORLD_CACHE_CELL_UPDATES_SOFT_CAP, world_cache_active_cells_count);
    let unlit = atomicLoad(&world_cache_b[0u]);
    let refresh_reserve = min(world_cache_active_cells_count - unlit,
        budget / PRIORITY_REFRESH_RESERVE_DIVISOR);
    let first_light = min(unlit, budget - refresh_reserve);
    atomicStore(&world_cache_b[PRIORITY_QUOTAS], first_light);
    atomicStore(&world_cache_b[PRIORITY_UNLIT_COUNT], unlit);
    atomicStore(&world_cache_b[PRIORITY_SELECTED_UNLIT], first_light);
    var remaining = budget - first_light;
    for (var bucket = PRIORITY_BUCKET_COUNT - 1u; bucket > 0u; bucket -= 1u) {
        let quota = min(atomicLoad(&world_cache_b[bucket]), remaining);
        atomicStore(&world_cache_b[PRIORITY_QUOTAS + bucket], quota);
        remaining -= quota;
    }
    var offset = 0u;
    for (var bucket = 0u; bucket < PRIORITY_BUCKET_COUNT; bucket += 1u) {
        atomicStore(&world_cache_b[PRIORITY_OFFSETS + bucket], offset);
        offset += atomicLoad(&world_cache_b[PRIORITY_QUOTAS + bucket]);
    }
    atomicStore(&world_cache_b[PRIORITY_SELECTED_COUNT], offset);
    atomicStore(&world_cache_b[PRIORITY_SELECTED_REFRESH], offset - first_light);
    atomicStore(&world_cache_b[PRIORITY_RESET], constants.reset);
    world_cache_active_cells_dispatch = vec3((offset + 63u) / 64u, 1u, 1u);
}

@compute @workgroup_size(256)
fn select_world_cache_priority(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if local < PRIORITY_BUCKET_COUNT { atomicStore(&priority_counts[local], 0u); }
    workgroupBarrier();
    var cell = 0u;
    var bucket = 0u;
    var rank = 0u;
    let valid = id.x < world_cache_active_cells_count;
    if valid {
        // Rotate ties without changing age priority or depending on RNG state.
        let rotation = (constants.render_frame * WORLD_CACHE_CELL_UPDATES_SOFT_CAP)
            % max(world_cache_active_cells_count, 1u);
        cell = world_cache_active_cell_indices[(id.x + rotation) % world_cache_active_cells_count];
        bucket = world_cache_priority_bucket(cell);
        if atomicLoad(&world_cache_b[PRIORITY_QUOTAS + bucket]) != 0u {
            rank = atomicAdd(&priority_counts[bucket], 1u);
        }
    }
    workgroupBarrier();
    if local < PRIORITY_BUCKET_COUNT {
        let count = atomicLoad(&priority_counts[local]);
        priority_bases[local] = 0u;
        if count != 0u {
            priority_bases[local] = atomicAdd(&world_cache_b[PRIORITY_CURSORS + local], count);
        }
    }
    workgroupBarrier();
    rank += priority_bases[bucket];
    if valid && rank < atomicLoad(&world_cache_b[PRIORITY_QUOTAS + bucket]) {
        world_cache_a[atomicLoad(&world_cache_b[PRIORITY_OFFSETS + bucket]) + rank] = cell;
    }
}
