#define_import_path bevy_solarik::primary_light_trace

// Debug direct-light ablation only. Preserve the first 32 invalid invocations
// across the entire run; never replace or clamp production radiance.
struct PrimaryLightFaults {
    count: atomic<u32>,
    reserved: array<u32, 3>,
    records: array<array<vec4u, 20>, 32>,
}
@group(3) @binding(0) var<storage, read_write> faults: PrimaryLightFaults;
var<private> light_trace: array<vec4u, 20>;

fn light_trace_set(row:u32, value:vec4f) { light_trace[row]=bitcast<vec4u>(value); }
fn light_trace_set_u(row:u32, value:vec4u) { light_trace[row]=value; }
fn light_trace_reset() {
    for(var row=0u;row<20u;row+=1u) { light_trace[row]=vec4u(0u); }
}
fn light_trace_commit(pixel:vec2u, frame:u32, seed:u32, result:vec3f) {
    if all((bitcast<vec3u>(result)&vec3u(0x7fffffffu))<vec3u(0x7f800000u)) {return;}
    let slot=atomicAdd(&faults.count,1u);
    if slot>=32u {return;}
    light_trace[0]=vec4u(pixel,frame,seed);
    for(var row=0u;row<20u;row+=1u) {faults.records[slot][row]=light_trace[row];}
}
