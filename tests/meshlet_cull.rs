//! Actual Bevy WESL composition for both production instance-cull variants.
#[path = "support/shader_graph.rs"]
mod shader_graph;
use bevy_shader::ShaderDefVal;

#[test]
fn production_meshlet_cull_composes_first_and_second_active_slot_variants() {
    let mut graph = shader_graph::Graph::load();
    for order in ["MESHLET_FIRST_CULLING_PASS", "MESHLET_SECOND_CULLING_PASS"] {
        graph.validate(
            "bevy_pbr::meshlet::cull_instances",
            &[
                "MESHLET_INSTANCE_CULLING_PASS".into(),
                order.into(),
                ShaderDefVal::UInt("CULL_DISPATCH_WIDTH".into(), 65535),
            ],
        );
    }
}
