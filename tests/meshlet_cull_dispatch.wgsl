struct Parameters { count: u32, items_per_group: u32, padding: vec2<u32> }
@group(0) @binding(0) var<storage, read_write> dispatch: CullDispatchArgs;
@group(0) @binding(1) var<uniform> parameters: Parameters;
@group(0) @binding(2) var<storage, read_write> visits: array<atomic<u32>>;

@compute @workgroup_size(128)
fn produce(@builtin(global_invocation_id) id: vec3<u32>) {
    let step = max(1u, (parameters.count + 511u) / 512u);
    let count = min((id.x + 1u) * step, parameters.count);
    let size = cull_dispatch_size(count, parameters.items_per_group);
    atomicMax(&dispatch.x, size.x);
    atomicMax(&dispatch.y, size.y);
}

@compute @workgroup_size(128)
fn consume(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let invocation = cull_dispatch_index(id, groups);
    let lanes = 128u / parameters.items_per_group;
    let item = invocation / lanes;
    if item >= parameters.count || invocation % lanes != 0u { return; }
    atomicAdd(&visits[item], 1u);
}
