// Random-number helpers from Bevy 0.19.1, bevy_pbr/src/render/utils.wgsl.
// Copyright Bevy Contributors. Dual-licensed under MIT or Apache-2.0.
// Kept here so the production emitter-sampler probe needs no sibling checkout.
fn rand_u(state: ptr<function, u32>) -> u32 {
    *state = *state * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_range_u(n: u32, state: ptr<function, u32>) -> u32 {
    return rand_u(state) % n;
}
