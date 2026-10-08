#define_import_path bevy_solarik::coarse_walk
#import bevy_solarik::coarse_scene::{CoarseGrid, COARSE_MISSING}
const COARSE_WALK_LIMIT = 512u;
// Compensated face times preserve crossing order when quotients round to the
// same (or adjacent) f32 values. Keep both parts through interval publication.
fn coarse_time_sum(a: f32, b: f32) -> vec2f {
    // FastTwoSum, with |large| >= |small|. The rounded increment has the
    // sign of small. Expressing that invariant through abs/sign preserves
    // its rounding point on backends that reassociate plain subtractions.
    let large=select(b,a,abs(a)>=abs(b));
    let small=select(a,b,abs(a)>=abs(b));
    // All accepted coordinates are below 1e10; the bound preserves their
    // value while making this rounded sum an explicit operation.
    let sum=clamp(large+small,-1e20,1e20);
    let increment=abs(sum-large)*sign(small);
    return vec2f(sum,small-increment);
}
fn coarse_face_time(face: f32, origin: f32, direction: f32) -> vec2f {
    let delta=coarse_time_sum(face,-origin);
    // Only ranges below 1e20 are accepted. Saturate distant face times before
    // residual arithmetic can produce infinities for near-parallel directions.
    let quotient=clamp(delta.x/direction,-1e30,1e30);
    if abs(quotient)>=1e30 { return vec2f(quotient,0.0); }
    let remainder=fma(-quotient,direction,delta.x)+delta.y;
    return vec2f(quotient,remainder/direction);
}
fn coarse_time_less(a: vec2f, b: vec2f) -> bool { return (a.x-b.x)<(b.y-a.y); }
fn coarse_time_min(a: vec2f, b: vec2f) -> vec2f { if coarse_time_less(a,b) { return a; } return b; }
fn coarse_time_max(a: vec2f, b: vec2f) -> vec2f { if coarse_time_less(a,b) { return b; } return a; }
struct CoarseWalk {
    cell: vec3i, running: u32,
    step: vec3i, visited: u32,
    origin: vec3f, overflow: u32,
    direction: vec3f,
    distance: vec2f, end: vec2f,
    next: array<vec2f,3>,
}
struct CoarseInterval { grid_index: u32, valid: u32, low: vec2f, high: vec2f }
fn coarse_boundary_time(grid: CoarseGrid, walk: CoarseWalk, axis: u32, forward: bool) -> vec2f {
    if walk.step[axis]==0 { return vec2f(1e30,0.0); }
    let offset=select(0,1,(walk.step[axis]>0)==forward);
    let address=walk.cell[axis]+grid.origin_resolution[axis]+offset;
    return coarse_face_time(f32(address)/f32(grid.origin_resolution.w),walk.origin[axis],walk.direction[axis]);
}
fn coarse_walk_begin(grid: CoarseGrid, origin: vec3f, direction: vec3f, range: vec2f) -> CoarseWalk {
    var walk: CoarseWalk;
    let length_squared=dot(direction,direction);
    if !all(abs(origin)<vec3f(1e10)) || !(length_squared>=0.99999 && length_squared<=1.00001)
        || !(range.x>=0.0 && range.y>range.x && range.y<1e20)
        || any(grid.dimensions.xyz==vec3u(0u)) || grid.origin_resolution.w<=0 {
        walk.overflow=2u; return walk;
    }
    let minimum=vec3f(grid.origin_resolution.xyz)/f32(grid.origin_resolution.w);
    let maximum=(vec3f(grid.origin_resolution.xyz)+vec3f(grid.dimensions.xyz))/f32(grid.origin_resolution.w);
    var low=vec2f(range.x,0.0); var high=vec2f(range.y,0.0);
    for (var axis=0u; axis<3u; axis+=1u) {
        if direction[axis]==0.0 {
            if origin[axis]<minimum[axis] || origin[axis]>=maximum[axis] { return walk; }
        } else {
            let a=coarse_face_time(minimum[axis],origin[axis],direction[axis]);
            let b=coarse_face_time(maximum[axis],origin[axis],direction[axis]);
            low=coarse_time_max(low,coarse_time_min(a,b));
            high=coarse_time_min(high,coarse_time_max(a,b));
        }
    }
    if !coarse_time_less(low,high) { return walk; }
    walk.origin=origin; walk.direction=direction; walk.distance=low; walk.end=high;
    let point=fma(direction,vec3f(low.x),origin)+direction*low.y;
    walk.cell=vec3i(floor(point*f32(grid.origin_resolution.w)))-grid.origin_resolution.xyz;
    walk.step=vec3i(sign(direction));
    walk.cell=clamp(walk.cell,vec3i(0),vec3i(grid.dimensions.xyz)-vec3i(1));
    // Correct a rounded initial point using the original ray/face times.
    for (var axis=0u; axis<3u; axis+=1u) {
        if walk.step[axis]==0 { continue; }
        for (var attempt=0u; attempt<2u; attempt+=1u) {
            let next=coarse_boundary_time(grid,walk,axis,true);
            let previous=coarse_boundary_time(grid,walk,axis,false);
            if !coarse_time_less(low,next) { walk.cell[axis]+=walk.step[axis]; }
            else if coarse_time_less(low,previous) { walk.cell[axis]-=walk.step[axis]; }
            else { break; }
        }
        let next=coarse_boundary_time(grid,walk,axis,true);
        let previous=coarse_boundary_time(grid,walk,axis,false);
        if !coarse_time_less(low,next) || coarse_time_less(low,previous) {
            walk.overflow=3u; return walk;
        }
    }
    if any(walk.cell<vec3i(0)) || any(walk.cell>=vec3i(grid.dimensions.xyz)) { walk.overflow=3u; return walk; }
    walk.running=1u;
    for (var axis=0u; axis<3u; axis+=1u) { walk.next[axis]=coarse_boundary_time(grid,walk,axis,true); }
    return walk;
}
fn coarse_walk_next(grid: CoarseGrid, walk: ptr<function,CoarseWalk>) -> CoarseInterval {
    var result: CoarseInterval; result.grid_index=COARSE_MISSING;
    if (*walk).running==0u { return result; }
    if (*walk).visited>=COARSE_WALK_LIMIT {
        (*walk).overflow=1u; (*walk).running=0u; return result;
    }
    (*walk).visited+=1u;
    let crossing=coarse_time_min(coarse_time_min((*walk).next[0],(*walk).next[1]),(*walk).next[2]);
    let high=coarse_time_min(crossing,(*walk).end);
    result.low=(*walk).distance; result.high=coarse_time_max(high,result.low);
    if coarse_time_less(result.low,result.high) {
        let p=vec3u((*walk).cell);
        result.grid_index=(p.x*grid.dimensions.y+p.y)*grid.dimensions.z+p.z;
        result.valid=1u;
    }
    if !coarse_time_less(high,(*walk).end) { (*walk).running=0u; return result; }
    for (var axis=0u; axis<3u; axis+=1u) {
        if !coarse_time_less(crossing,(*walk).next[axis]) { (*walk).cell[axis]+=(*walk).step[axis]; }
    }
    if any((*walk).cell<vec3i(0)) || any((*walk).cell>=vec3i(grid.dimensions.xyz)) {
        (*walk).running=0u; return result;
    }
    (*walk).distance=coarse_time_max((*walk).distance,crossing);
    for (var axis=0u; axis<3u; axis+=1u) { (*walk).next[axis]=coarse_boundary_time(grid,*walk,axis,true); }
    return result;
}
