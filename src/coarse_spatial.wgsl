#define_import_path bevy_solarik::coarse_spatial
#import bevy_solarik::coarse_spatial_pack::{spatial_axis, spatial_depth, spatial_normal, spatial_part}
#import bevy_solarik::coarse_walk::{CoarseInterval, coarse_face_time, coarse_time_less, coarse_time_min, coarse_time_max}
struct SpatialColumns {
    address: vec2i, step: vec2i, base: vec2i, axes: vec2u,
    next: array<vec2f,2>, distance: vec2f, end: vec2f,
    origin: vec3f, direction: vec3f, resolution: f32,
    running: u32, visited: u32, overflow: u32,
}
fn spatial_boundary(columns: SpatialColumns, i: u32, forward: bool) -> vec2f {
    if columns.step[i]==0 { return vec2f(1e30,0.0); }
    let index=columns.base[i]+columns.address[i]+select(0,1,(columns.step[i]>0)==forward);
    let axis=columns.axes[i];
    return coarse_face_time(f32(index)/columns.resolution,columns.origin[axis],columns.direction[axis]);
}
fn spatial_columns(cell: vec3i, resolution: f32, axis: u32, origin: vec3f, direction: vec3f,
    low: vec2f, high: vec2f) -> SpatialColumns {
    var result: SpatialColumns;
    result.axes=vec2u((axis+1u)%3u,(axis+2u)%3u);
    result.base=vec2i(cell[result.axes.x],cell[result.axes.y])*8;
    result.origin=origin; result.direction=direction; result.resolution=resolution*8.0;
    result.distance=low; result.end=high;
    let point=fma(direction,vec3f(low.x),origin)+direction*low.y;
    result.address=clamp(vec2i(floor(vec2f(point[result.axes.x],point[result.axes.y])*result.resolution))-result.base,vec2i(0),vec2i(7));
    result.step=vec2i(sign(vec2f(direction[result.axes.x],direction[result.axes.y])));
    for(var i=0u;i<2u;i+=1u) {
        if result.step[i]==0 { result.next[i]=vec2f(1e30,0.0); continue; }
        for(var attempt=0u;attempt<2u;attempt+=1u) {
            let next=spatial_boundary(result,i,true);
            let previous=spatial_boundary(result,i,false);
            if !coarse_time_less(low,next) { result.address[i]+=result.step[i]; }
            else if coarse_time_less(low,previous) { result.address[i]-=result.step[i]; }
            else { break; }
        }
        result.next[i]=spatial_boundary(result,i,true);
    }
    if any(result.address<vec2i(0)) || any(result.address>=vec2i(8)) { result.overflow=2u; return result; }
    result.running=1u; return result;
}
fn spatial_columns_next(columns: ptr<function,SpatialColumns>) -> CoarseInterval {
    var result: CoarseInterval;
    if (*columns).running==0u { return result; }
    if (*columns).visited>=32u { (*columns).overflow=1u; (*columns).running=0u; return result; }
    (*columns).visited+=1u;
    let crossing=coarse_time_min((*columns).next[0],(*columns).next[1]);
    let high=coarse_time_min(crossing,(*columns).end);
    result.low=(*columns).distance; result.high=high;
    result.grid_index=u32((*columns).address.y*8+(*columns).address.x);
    result.valid=u32(coarse_time_less(result.low,high));
    if !coarse_time_less(high,(*columns).end) { (*columns).running=0u; return result; }
    for(var i=0u;i<2u;i+=1u) {
        if !coarse_time_less(crossing,(*columns).next[i]) { (*columns).address[i]+=(*columns).step[i]; }
    }
    if any((*columns).address<vec2i(0)) || any((*columns).address>=vec2i(8)) { (*columns).running=0u; return result; }
    (*columns).distance=coarse_time_max((*columns).distance,crossing);
    for(var i=0u;i<2u;i+=1u) { (*columns).next[i]=spatial_boundary(*columns,i,true); }
    return result;
}
struct SpatialHit { t: f32, normal: vec3f, part: u32, valid: u32 }
fn spatial_intersect(word: u32, cell: vec3i, resolution: f32, axis: u32, column: u32,
    two_sided: u32, origin: vec3f, direction: vec3f, low: vec2f, high: vec2f) -> SpatialHit {
    var result: SpatialHit;
    if word==0u { return result; }
    let normal=spatial_normal(word); let part=spatial_part(word);
    if spatial_axis(normal)!=axis { return result; }
    let denominator=dot(normal,direction);
    if abs(denominator)<1e-7 || ((two_sided&(1u<<part))==0u && denominator>=0.0) { return result; }
    var offset=vec3f(0.0);
    offset[axis]=spatial_depth(word);
    offset[(axis+1u)%3u]=(f32(column%8u)+0.5)/8.0;
    offset[(axis+2u)%3u]=(f32(column/8u)+0.5)/8.0;
    let point=(vec3f(cell)+offset)/resolution;
    let t=dot(normal,point-origin)/denominator;
    if !coarse_time_less(vec2f(t,0.0),low) && coarse_time_less(vec2f(t,0.0),high) {
        result.t=t; result.normal=normal; result.part=part; result.valid=1u;
    }
    return result;
}
