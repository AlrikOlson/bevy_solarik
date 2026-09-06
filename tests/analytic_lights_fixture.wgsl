struct LocalLight { position: vec3f, radius: f32, radiance: vec3f, inverse_pdf: f32, direction: vec3f, range: f32, cos_outer: f32, cos_inner: f32, padding: vec2f }
struct DirectionalLight { direction_to_light: vec3f, cos_theta_max: f32, luminance: vec3f, inverse_pdf: f32 }
struct LightSource { kind: u32, id: u32 }
const LIGHT_SOURCE_KIND_DIRECTIONAL: u32 = 0u;
const LIGHT_SAMPLE_LOCAL: f32 = 2.0;
const RAY_T_MIN: f32 = 0.001;
const RAY_T_MAX: f32 = 10000.0;
@group(0) @binding(0) var<storage> config: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4f>;
var<private> local_lights: array<LocalLight, 1>;
var<private> directional_lights: array<DirectionalLight, 1>;
// Runtime arrays are replaced by fixed scene inputs in this fixture.
@compute @workgroup_size(16)
fn probe(@builtin(local_invocation_index) i: u32) {
    let s = config[0].x;
    var light = LocalLight(vec3(0.0,0.0,5.0)*s, 0.5*s, vec3(2.0), 1.0,
        vec3(0.0,0.0,-1.0), 100.0*s, -1.0, -1.0, vec2(0.0));
    if i == 5u || i == 6u {
        light.cos_outer = 0.8;
        light.cos_inner = 0.9;
        if i == 6u { light.direction.z = 1.0; }
    }
    if i == 7u { light.range = 2.0*s; }
    if i >= 8u && i != 12u { light.radiance = vec3(0.0); }
    local_lights[0] = light;
    directional_lights[0] = DirectionalLight(vec3(0.0,0.0,1.0), 0.99,
        select(vec3(0.0), vec3(3.0), i >= 8u), 0.0628);
    var origin = vec3(0.0);
    var wi = vec3(0.0,0.0,1.0);
    var limit = RAY_T_MAX;
    if i == 1u || i == 9u { wi = vec3(1.0,0.0,0.0); }
    if i == 2u || i == 10u { limit = 2.0*s; }
    if i == 3u { wi.z = -1.0; }
    if i == 4u { origin.z = 5.0*s; }
    if i == 13u { directional_lights[0].cos_theta_max = 1.0; }
    if i == 14u { limit = 0.0; }
    output[i] = vec4(analytic_light_radiance(origin, wi, limit, i != 11u && i != 15u), 1.0);
}
