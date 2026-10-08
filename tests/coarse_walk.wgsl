struct Request { origin: vec4f, direction: vec4f }
struct IntervalRecord { metadata: vec4u, depths: vec4f }
struct Probe { counts: vec4u, intervals: array<IntervalRecord,512> }
@group(0) @binding(0) var<storage,read> grid: CoarseGrid;
@group(0) @binding(1) var<storage,read> lookup: array<u32>;
@group(0) @binding(2) var<storage,read> rows: array<CoarsePacked>;
@group(0) @binding(3) var<storage,read> requests: array<Request>;
@group(0) @binding(4) var<storage,read_write> output: array<Probe>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x>=arrayLength(&requests) { return; }
    let request=requests[id.x];
    var walk=coarse_walk_begin(grid,request.origin.xyz,request.direction.xyz,vec2f(request.origin.w,request.direction.w));
    var count=0u;
    while walk.running!=0u {
        let interval=coarse_walk_next(grid,&walk);
        if interval.valid==0u || interval.grid_index>=arrayLength(&lookup) { continue; }
        let cell=lookup[interval.grid_index];
        if cell>=grid.metadata.x { continue; }
        let row=coarse_unpack(rows[cell*COARSE_DIRECTIONS]);
        output[id.x].intervals[count].metadata=vec4u(cell,row.counts.y,0u,0u);
        output[id.x].intervals[count].depths=vec4f(interval.low,interval.high);
        count+=1u;
    }
    output[id.x].counts=vec4u(count,walk.visited,walk.overflow,0u);
}
