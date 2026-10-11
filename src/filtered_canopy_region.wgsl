#define_import_path bevy_solarik::filtered_canopy_region
#import bevy_solarik::filtered_canopy_transport::{ForestCell, ForestEvent, forest_trace_projected}
@group(0) @binding(27) var<storage, read> forest_regions: array<vec4u>;
fn forest_hash(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}
fn forest_sample(seed: vec2u, identity: vec2u) -> vec2f {
    // Independent cell clocks form a superposition of first-event processes.
    // Logical identity survives repacking and origin changes. Both caller seed
    // components must affect both variates: primary rays pass (pixel, frame).
    // Separate domains avoid frozen optical coverage and frame-wide joint marks.
    let a = forest_hash(seed.x ^ forest_hash(seed.y ^ 0x68bc21ebu)
        ^ forest_hash(identity.x) ^ identity.y);
    let b = forest_hash(seed.y ^ forest_hash(seed.x ^ 0x02e5be93u)
        ^ forest_hash(identity.y ^ 0x9e3779b9u) ^ identity.x);
    return (vec2f(vec2u(a >> 8u, b >> 8u)) + vec2f(0.5)) * (1.0 / 16777216.0);
}
fn forest_box(low: vec3f, high: vec3f, origin: vec3f, direction: vec3f, minimum: f32, maximum: f32) -> bool {
    var begin = minimum;
    var end = maximum;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if direction[axis] == 0.0 {
            if origin[axis] < low[axis] || origin[axis] > high[axis] { return false; }
        } else {
            let a = (low[axis] - origin[axis]) / direction[axis];
            let b = (high[axis] - origin[axis]) / direction[axis];
            begin = max(begin, min(a, b));
            end = min(end, max(a, b));
            if begin > end { return false; }
        }
    }
    return begin <= end;
}
fn forest_load_cell(address: u32) -> ForestCell {
    return ForestCell(bitcast<vec4f>(forest_regions[address]),
        bitcast<vec4f>(forest_regions[address + 1u]), bitcast<vec4f>(forest_regions[address + 2u]),
        bitcast<vec4f>(forest_regions[address + 3u]), bitcast<vec4f>(forest_regions[address + 4u]),
        bitcast<vec4f>(forest_regions[address + 5u]), bitcast<vec4f>(forest_regions[address + 6u]));
}
struct ForestSupport {
    cell:ForestCell, ground:ForestCell, plane:vec4f, identity:vec2u, supported:bool,
}
fn forest_load_support(base:u32,address:u32,support_index:u32) -> ForestSupport {
    let record=forest_regions[address+7u];
    let cell=forest_load_cell(address);
    var result=ForestSupport(cell,cell,vec4f(0.0),record.xy,record.w>0u);
    if record.w>0u {
        let ground_address=base+forest_regions[base+record.z+support_index/4u][support_index%4u];
        result.plane=bitcast<vec4f>(forest_regions[ground_address]);
        let last=forest_regions[ground_address+2u];
        result.ground.corners_a=bitcast<vec4f>(forest_regions[ground_address+1u]);
        result.ground.corners_b=bitcast<vec4f>(last.xyxy);
        result.identity^=vec2u(forest_hash(last.z^0x942bac31u),forest_hash(last.w^0x8612cd5bu));
    }
    return result;
}
fn forest_trace_leaf(base:u32,address:u32,origin:vec3f,direction:vec3f,
                     minimum:f32,maximum:f32,seed:vec2u) -> ForestEvent {
    let record=forest_regions[address+7u];
    var result:ForestEvent;
    var closest=maximum;
    for(var i=0u;i<max(1u,record.w);i+=1u) {
        let support=forest_load_support(base,address,i);
        let event=forest_trace_projected(support.cell,support.ground,support.plane,support.supported,
            origin,direction,minimum,closest,forest_sample(seed,support.identity));
        if event.normal_hit.w>0.0 && event.position_distance.w<closest {
            result=event;closest=event.position_distance.w;
        }
    }
    return result;
}
fn forest_trace_region(base: u32, origin: vec3f, direction: vec3f, minimum: f32, maximum: f32, seed: vec2u) -> ForestEvent {
    var result: ForestEvent;
    let header = forest_regions[base];
    var node = 0u;
    var closest = maximum;
    loop {
        if node >= header.x { break; }
        let low = forest_regions[base + header.y + node * 2u];
        let high = forest_regions[base + header.y + node * 2u + 1u];
        if !forest_box(bitcast<vec3f>(low.xyz), bitcast<vec3f>(high.xyz), origin, direction, minimum, closest) {
            node = low.w;
            continue;
        }
        if high.w == 0xffffffffu {
            node += 1u;
            continue;
        }
        let address = base + header.z + high.w * 8u;
        let event = forest_trace_leaf(base, address, origin, direction, minimum, closest, seed);
        if event.normal_hit.w > 0.0 && event.position_distance.w < closest {
            result = event;
            closest = event.position_distance.w;
        }
        node = low.w;
    }
    return result;
}
