#define_import_path bevy_solarik::collimated

// Use tangent radius, not 1-cos(theta): a few MOA loses precision in f32.
// xyz is the world outgoing axis, w is tan(angular radius); w=0 is ordinary.
fn collimated_weight(cone: vec4<f32>, outgoing: vec3<f32>) -> f32 {
    if cone.w <= 0.0 { return 1.0; }
    let forward = dot(outgoing, cone.xyz);
    if forward <= 0.0 { return 0.0; }
    let radius = length(cross(outgoing, cone.xyz)) / forward;
    return 1.0 - smoothstep(0.9 * cone.w, cone.w, radius);
}
