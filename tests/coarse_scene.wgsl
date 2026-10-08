struct Request { point: vec4f, parameters: vec4u }
struct Probe { address: vec4u, measurements: CoarseMeasurements }
@group(0) @binding(0) var<storage,read> grid: CoarseGrid;
@group(0) @binding(1) var<storage,read> lookup: array<u32>;
@group(0) @binding(2) var<storage,read> rows: array<CoarsePacked>;
@group(0) @binding(3) var<storage,read> requests: array<Request>;
@group(0) @binding(4) var<storage,read_write> output: array<Probe>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&requests) { return; }
    let request = requests[id.x];
    var result: Probe;
    result.address.y = COARSE_MISSING;
    let index = coarse_grid_index(grid,request.point.xyz);
    let direction = request.parameters.x;
    if grid.metadata.w == 1u && direction < COARSE_DIRECTIONS && index < arrayLength(&lookup) {
        let cell = lookup[index];
        if cell < grid.metadata.x && cell*COARSE_DIRECTIONS+direction < arrayLength(&rows) {
            result.address = vec4u(1u,cell,direction,index);
            result.measurements = coarse_unpack(rows[cell*COARSE_DIRECTIONS+direction]);
        }
    }
    output[id.x] = result;
}
