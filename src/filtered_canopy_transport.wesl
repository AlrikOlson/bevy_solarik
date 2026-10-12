

// Finite-support source first events. Inputs use physical metres in a rigid region frame.
@group(0) @binding(26) var<storage, read> forest_sources: array<vec4u>;

struct ForestCell {
    axis_x: vec4f,
    // xyz: source up in region coordinates, w: minus dot(up, root).
    axis_y: vec4f,
    axis_z: vec4f,
    corners_a: vec4f,
    corners_b: vec4f,
    // Mean height, height-squared mass / physical terrain area.
    population_a: vec4f,
    population_b: vec4f,
}
struct ForestEvent {
    position_distance: vec4f,
    normal_hit: vec4f,
    color_coverage: vec4f,
    surface: vec4f,
    source: vec4u,
}
// Both the proposal integral and the sampled event use this same physical walk.
struct ForestWalk { event:ForestEvent, optical:f32 }
struct ForestMark {
    normal: vec3f,
    color: vec3f,
    surface: vec4f,
    material: u32,
}
fn forest_row(source: u32, view: u32, height: u32) -> vec4u {
    let header = forest_sources[0];
    return forest_sources[header.w + (source * header.y + view) * header.z + height];
}
fn forest_view(direction: vec3f) -> vec3f {
    let coordinate = acos(clamp(direction.y, -1.0, 1.0)) * 32.0 / 3.141592653589793;
    let low = min(u32(floor(coordinate)), 32u);
    return vec3f(f32(low), f32(min(low + 1u, 32u)), coordinate - f32(low));
}
fn forest_mark(source: u32, height: u32, direction: vec3f, random: f32) -> ForestMark {
    let view = forest_view(direction);
    let a = forest_row(source, u32(view.x), height);
    let b = forest_row(source, u32(view.y), height);
    let area_a = bitcast<f32>(a.z) * (1.0 - view.z);
    let area_b = bitcast<f32>(b.z) * view.z;
    let draw = clamp(random, 0.0, 0.99999994) * (area_a + area_b);
    var row = a;
    var sample = draw / max(area_a, 1e-30);
    if draw >= area_a {
        row = b;
        sample = (draw - area_a) / max(area_b, 1e-30);
    }
    var mark: ForestMark;
    if row.y == 0u { return mark; }
    var low = 0u;
    var high = row.y;
    loop {
        if low >= high { break; }
        let middle = low + (high - low) / 2u;
        let cumulative = bitcast<f32>(forest_sources[row.x + middle * 3u].w);
        if sample < cumulative { high = middle; } else { low = middle + 1u; }
    }
    let offset = row.x + min(low, row.y - 1u) * 3u;
    let normal = bitcast<vec4f>(forest_sources[offset]).xyz;
    let phi = atan2(direction.x, direction.z);
    let c = cos(phi);
    let s = sin(phi);
    mark.normal = vec3f(c * normal.x + s * normal.z, normal.y, -s * normal.x + c * normal.z);
    mark.color = bitcast<vec4f>(forest_sources[offset + 1u]).xyz;
    mark.material = forest_sources[offset + 1u].w;
    mark.surface = bitcast<vec4f>(forest_sources[offset + 2u]);
    return mark;
}
fn forest_clip(span: vec2f, distance: f32, rate: f32) -> vec2f {
    var result = span;
    if abs(rate) < 1e-12 {
        if distance < 0.0 { return vec2f(1.0, 0.0); }
        return result;
    }
    let time = -distance / rate;
    if rate > 0.0 { result.x = max(result.x, time); }
    else { result.y = min(result.y, time); }
    return result;
}
fn forest_lateral(cell: ForestCell, origin: vec3f, direction: vec3f, minimum: f32, maximum: f32) -> vec2f {
    let corners = array<vec2f, 4>(cell.corners_a.xy, cell.corners_a.zw, cell.corners_b.xy, cell.corners_b.zw);
    let centre = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25;
    var span = vec2f(minimum, maximum);
    for (var i = 0u; i < 4u; i += 1u) {
        let a = corners[i];
        let edge = corners[(i + 1u) % 4u] - a;
        var inward = vec2f(-edge.y, edge.x);
        if dot(inward, centre - a) < 0.0 { inward = -inward; }
        let distance = dot(inward, origin.xz - a);
        let rate = dot(inward, direction.xz);
        if rate == 0.0 && distance == 0.0 && (inward.x < 0.0 || (inward.x == 0.0 && inward.y < 0.0)) {
            return vec2f(1.0, 0.0);
        }
        span = forest_clip(span, distance, rate);
        if span.x >= span.y { return span; }
    }
    return span;
}
// xi.x selects optical free flight; xi.y selects the joint source mark.
// Caller owns the sample sequence. Neither allocation slots nor rebased world
// coordinate hashes are used as stochastic identity.
// Ordered exact optical integration. PBRT4 section 14.1.3.
struct ForestColumn { low:f32, step:f32, mass:f32, source:u32 }
struct ForestSlab { span:vec2f, rate:f32, height:i32 }
fn forest_column(source:u32, population:vec4f) -> ForestColumn {
    let shape=bitcast<vec4f>(forest_sources[source+1u]);
    return ForestColumn(shape.x*population.x, (shape.y-shape.x)*population.x/8.0,
                        select(0.0,population.y,population.x>0.0 && population.y>0.0),source);
}
fn forest_column_rate(column:ForestColumn, height:u32, view:vec3f) -> f32 {
    let a=bitcast<f32>(forest_row(column.source,u32(view.x),height).z);
    let b=bitcast<f32>(forest_row(column.source,u32(view.y),height).z);
    return mix(a,b,view.z)*column.mass/column.step;
}
fn forest_next_slab(column:ForestColumn, lateral:vec2f, origin_y:f32, direction_y:f32,
                    view:vec3f, first:i32, increment:i32) -> ForestSlab {
    var empty=ForestSlab(vec2f(lateral.y),0.0,first);
    if column.mass<=0.0 || column.step<=0.0 { return empty; }
    for(var height=first;height>=0 && height<8;height+=increment) {
        let rate=forest_column_rate(column,u32(height),view);
        if rate<=0.0 { continue; }
        let low=column.low+f32(height)*column.step;
        let high=column.low+f32(height+1)*column.step;
        var span=forest_clip(lateral,origin_y-low,direction_y);
        span=forest_clip(span,high-origin_y,-direction_y);
        if span.x<span.y { return ForestSlab(span,rate,height); }
    }
    return empty;
}
fn forest_make_event(cell:ForestCell, origin:vec3f,direction:vec3f,view:vec3f,
                     source:u32,height:u32,random:f32,time:f32) -> ForestEvent {
    let mark=forest_mark(source,height,view,random);
    let normal=normalize(cell.axis_x.xyz*mark.normal.x+cell.axis_y.xyz*mark.normal.y+cell.axis_z.xyz*mark.normal.z);
    return ForestEvent(vec4f(origin+direction*time,time),vec4f(normal,1.0),
                       vec4f(mark.color,0.0),mark.surface,
                       vec4u(forest_sources[source+1u].zw,mark.material,height));
}
fn forest_select_column(a:ForestColumn,b:ForestColumn,index:u32) -> ForestColumn {
    if index>=8u { return b; }
    return a;
}
fn forest_horizontal_rate(column:ForestColumn, height:u32, origin_y:f32,view:vec3f) -> f32 {
    if column.mass<=0.0 || column.step<=0.0 { return 0.0; }
    let low=column.low+f32(height)*column.step;
    let high=column.low+f32(height+1u)*column.step;
    if origin_y<low || origin_y>high { return 0.0; }
    return max(0.0,forest_column_rate(column,height,view));
}
fn forest_trace_horizontal(cell:ForestCell,origin:vec3f,direction:vec3f,lateral:vec2f,
                           local_view:vec3f,origin_y:f32,optical_target:f32,random:f32,choose:bool) -> ForestWalk {
    let a=forest_column(0u,cell.population_a);
    let b=forest_column(1u,cell.population_b);
    let view=forest_view(local_view);
    var sigma=0.0;
    var last_positive=0u;
    for(var j=0u;j<16u;j+=1u) {
        let rate=forest_horizontal_rate(forest_select_column(a,b,j),j%8u,origin_y,view);
        sigma+=rate;
        if rate>0.0 { last_positive=j; }
    }
    var result:ForestEvent;
    let optical=sigma*(lateral.y-lateral.x);
    if choose && optical_target<optical {
        let time=lateral.x+optical_target/sigma;
        var draw=clamp(random,0.0,0.99999994)*sigma;
        for(var j=0u;j<16u;j+=1u) {
            let rate=forest_horizontal_rate(forest_select_column(a,b,j),j%8u,origin_y,view);
            if rate<=0.0 { continue; }
            // The final positive category owns a residual rounded to its endpoint.
            if draw>=rate && j!=last_positive { draw-=rate;continue; }
            result=forest_make_event(cell,origin,direction,local_view,j/8u,j%8u,
                clamp(draw/rate,0.0,0.99999994),time);
            break;
        }
    }
    result.color_coverage.w=1.0-exp(-optical);
    return ForestWalk(result,optical);
}
fn forest_slab_end(slab:ForestSlab, start:f32) -> f32 {
    return select(slab.span.x,slab.span.y,slab.span.x<=start);
}
fn forest_slab_rate(slab:ForestSlab, start:f32) -> f32 {
    return select(0.0,slab.rate,slab.span.x<=start && start<slab.span.y);
}
struct ForestTraversal {
    a:ForestColumn, b:ForestColumn, lateral:vec2f, view:vec3f,
    origin_y:f32, direction_y:f32, increment:i32,
}
fn forest_next(traversal:ForestTraversal,source:u32,height:i32) -> ForestSlab {
    let column=forest_select_column(traversal.a,traversal.b,source*8u);
    return forest_next_slab(column,traversal.lateral,traversal.origin_y,traversal.direction_y,
                            traversal.view,height,traversal.increment);
}
fn forest_choose_event(cell:ForestCell,origin:vec3f,direction:vec3f,view:vec3f,
                       a:ForestSlab,b:ForestSlab,rates:vec2f,time:f32,random:f32) -> ForestEvent {
    let draw=clamp(random,0.0,0.99999994)*(rates.x+rates.y);
    if rates.x>0.0 && (draw<rates.x || rates.y<=0.0) {
        return forest_make_event(cell,origin,direction,view,0u,u32(a.height),
            clamp(draw/rates.x,0.0,0.99999994),time);
    }
    // Subtraction may round the residual to rates.y. The last positive
    // category must still produce an event; its remapped variate is half-open.
    if rates.y>0.0 {
        return forest_make_event(cell,origin,direction,view,1u,u32(b.height),
            clamp((draw-rates.x)/rates.y,0.0,0.99999994),time);
    }
    var empty:ForestEvent;
    return empty;
}
fn forest_walk_ordered(cell:ForestCell,origin:vec3f,direction:vec3f,local_view:vec3f,
                       traversal:ForestTraversal,optical_target:f32,random:f32,choose:bool) -> ForestWalk {
    var result:ForestEvent;
    let first=select(7,0,traversal.increment>0);
    var a=forest_next(traversal,0u,first);
    var b=forest_next(traversal,1u,first);
    var start=min(a.span.x,b.span.x);
    var optical=0.0;
    loop {
        if start>=traversal.lateral.y { break; }
        let end=min(forest_slab_end(a,start),forest_slab_end(b,start));
        let rates=vec2f(forest_slab_rate(a,start),forest_slab_rate(b,start));
        let sigma=rates.x+rates.y;
        let depth=sigma*(end-start);
        if choose && result.normal_hit.w==0.0 && optical_target<optical+depth {
            let time=start+(optical_target-optical)/sigma;
            result=forest_choose_event(cell,origin,direction,local_view,a,b,rates,time,random);
        }
        optical+=depth;
        start=end;
        if a.span.y<=start { a=forest_next(traversal,0u,a.height+traversal.increment); }
        if b.span.y<=start { b=forest_next(traversal,1u,b.height+traversal.increment); }
    }
    result.color_coverage.w=1.0-exp(-optical);
    return ForestWalk(result,optical);
}
// Share the physical projection between first-event sampling and exact optical
// integration. Ground changes height coordinates, not source orientation.
struct ForestProjection {
    cell:ForestCell, lateral:vec2f, local_view:vec3f, origin_y:f32, direction_y:f32,
}
fn forest_project(cell:ForestCell,ground:ForestCell,plane:vec4f,supported:bool,
                  origin:vec3f,direction:vec3f,minimum:f32,maximum:f32) -> ForestProjection {
    var result:ForestProjection;
    var lateral=forest_lateral(cell,origin,direction,minimum,maximum);
    result.lateral=lateral;
    if lateral.x>=lateral.y { return result; }
    var physical=cell;
    var origin_y=dot(cell.axis_y.xyz,origin)+cell.axis_y.w;
    var direction_y=dot(cell.axis_y.xyz,direction);
    if supported {
        lateral=forest_lateral(ground,origin,direction,lateral.x,lateral.y);
        result.lateral=lateral;
        if lateral.x>=lateral.y { return result; }
        let projection=dot(plane.xyz,cell.axis_y.xyz);
        let jacobian=plane.y/(cell.axis_y.y*projection);
        physical.population_a.y*=jacobian;
        physical.population_b.y*=jacobian;
        let anchor=vec3f(ground.corners_a.x,plane.w,ground.corners_a.y);
        origin_y=dot(plane.xyz,origin-anchor)/projection;
        direction_y=dot(plane.xyz,direction)/projection;
    }
    let local_view=normalize(vec3f(dot(cell.axis_x.xyz,-direction),dot(cell.axis_y.xyz,-direction),dot(cell.axis_z.xyz,-direction)));
    return ForestProjection(physical,lateral,local_view,origin_y,direction_y);
}
// Two already ordered source columns need only a merge, not a 32-event sort.
fn forest_trace_projected(cell:ForestCell,ground:ForestCell,plane:vec4f,supported:bool,
                          origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,xi:vec2f) -> ForestEvent {
    let optical_target=-log(1.0-clamp(xi.x,0.0,0.99999994));
    return forest_trace_projected_optical(cell,ground,plane,supported,origin,direction,minimum,maximum,optical_target,xi.y);
}
// The conditioned caller supplies optical depth directly, avoiding a lossy
// thin-CDF -> 1-CDF -> log round trip. Ordinary sampling retains its original map.
fn forest_trace_projected_optical(cell:ForestCell,ground:ForestCell,plane:vec4f,supported:bool,
                                  origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,
                                  optical_target:f32,random:f32) -> ForestEvent {
    return forest_walk_projected(cell,ground,plane,supported,origin,direction,minimum,maximum,
        optical_target,random,true).event;
}
// A constant false choice removes mark selection from the integration caller.
// Keeping the same interval partition and sum prevents rounded totals from
// defining a conditional target beyond the sampler's last positive interval.
fn forest_walk_projected(cell:ForestCell,ground:ForestCell,plane:vec4f,supported:bool,
                          origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,
                          optical_target:f32,random:f32,choose:bool) -> ForestWalk {
    let p=forest_project(cell,ground,plane,supported,origin,direction,minimum,maximum);
    var empty:ForestWalk;
    if p.lateral.x>=p.lateral.y { return empty; }
    if abs(p.direction_y)<1e-12 {
        return forest_trace_horizontal(p.cell,origin,direction,p.lateral,p.local_view,p.origin_y,optical_target,random,choose);
    }
    let traversal=ForestTraversal(forest_column(0u,p.cell.population_a),forest_column(1u,p.cell.population_b),
        p.lateral,forest_view(p.local_view),p.origin_y,p.direction_y,select(-1,1,p.direction_y>0.0));
    return forest_walk_ordered(p.cell,origin,direction,p.local_view,traversal,optical_target,random,choose);
}
// Optical depth is additive even for overlapping cells. Do not reconstruct it
// from 1 - coverage: coverage rounds to one for optically dense sources.
fn forest_optical_projected(cell:ForestCell,ground:ForestCell,plane:vec4f,supported:bool,
                            origin:vec3f,direction:vec3f,minimum:f32,maximum:f32) -> f32 {
    return forest_walk_projected(cell,ground,plane,supported,origin,direction,minimum,maximum,
        0.0,0.0,false).optical;
}
fn forest_trace_supported(cell:ForestCell,ground:ForestCell,plane:vec4f,
                          origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,xi:vec2f) -> ForestEvent {
    return forest_trace_projected(cell,ground,plane,true,origin,direction,minimum,maximum,xi);
}
fn forest_trace_cell(cell:ForestCell,origin:vec3f,direction:vec3f,minimum:f32,maximum:f32,xi:vec2f) -> ForestEvent {
    return forest_trace_projected(cell,cell,vec4f(0.0),false,origin,direction,minimum,maximum,xi);
}
