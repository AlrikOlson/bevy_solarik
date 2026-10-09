struct Ray { origin: vec4f, direction: vec4f }
struct Result { depth: vec4f, counts: vec4u, first: vec4u, normal: vec4f }
@group(0) @binding(0) var<storage,read> grid: CoarseGrid;
@group(0) @binding(1) var<storage,read> lookup: array<u32>;
@group(0) @binding(2) var<storage,read> kernels: array<CoarseKernel>;
@group(0) @binding(3) var<storage,read> rays: array<Ray>;
@group(0) @binding(4) var<storage,read_write> output: array<Result>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if id.x>=arrayLength(&rays) { return; }
    let ray=rays[id.x];
    var walk=coarse_walk_begin(grid,ray.origin.xyz,ray.direction.xyz,vec2f(ray.origin.w,ray.direction.w));
    var result: Result;
    let basis=coarse_transport_direction(ray.direction.xyz);
    var transmission=1.0; var moment=0.0;
    loop {
        let interval=coarse_walk_next(grid,&walk);
        if interval.valid!=0u {
            let cell=lookup[interval.grid_index];
            if cell!=COARSE_MISSING {
                let sampled=coarse_transport_rate(kernels[cell],basis);
                let rate=sampled.value*f32(grid.origin_resolution.w);
                let length=(interval.high.x-interval.low.x)+(interval.high.y-interval.low.y);
                let mass=transmission*coarse_transport_coverage(rate*length);
                let centre=interval.low.x+interval.low.y+coarse_transport_centroid(rate,length);
                moment+=mass*centre;
                transmission*=exp(-rate*length);
                result.counts.x+=1u;
                result.counts.y|=sampled.unresolved;
            }
        }
        if walk.running==0u { break; }
    }
    result.depth.x=1.0-transmission;
    if result.depth.x>0.0 { result.depth.y=moment/result.depth.x; }
    result.counts.z=walk.visited;
    result.counts.w=walk.overflow;
    output[id.x]=result;
}
@compute @workgroup_size(64)
fn analytic(@builtin(global_invocation_id) id: vec3u) {
    if id.x>=arrayLength(&rays) { return; }
    let request=rays[id.x]; var result: Result;
    let flight=coarse_free_flight(request.origin.x,request.origin.yz,request.origin.w);
    result.depth=vec4f(flight.distance,
        coarse_transport_coverage(request.origin.x*(request.origin.z-request.origin.y)),
        coarse_transport_centroid(request.origin.x,request.origin.z-request.origin.y),0.0);
    result.counts=vec4u(flight.event,flight.valid,0u,0u);
    let basis=coarse_transport_direction(request.direction.xyz);
    result.first=vec4u(basis.indices,0u);
    result.normal=vec4f(basis.weights,0.0);
    output[id.x]=result;
}
