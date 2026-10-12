//! Production material/scene consumers through the actual Bevy WESL compiler.
#[path = "support/shader_graph.rs"]
mod shader_graph;
use bevy_shader::ShaderDefVal;

#[test]
fn scene_consumers_compose_across_production_feature_variants() {
    let mut graph = shader_graph::Graph::load();
    let consumers = [
        "scene::sampling",
        "realtime::primary_foliage",
        "realtime::primary_glass",
        "realtime::restir_gi",
        "realtime::world_cache_update",
        "realtime::specular_gi",
        "realtime::surface_path",
        "pathtracer::pathtracer",
        "filtered_canopy_prepass",
    ];
    for bits in 0..16 {
        let mut definitions = vec![
            ShaderDefVal::UInt("WORLD_CACHE_SIZE".into(), 1 << 20),
            ShaderDefVal::UInt("RASTER_SUBPIXEL_BITS".into(), 8),
            "BINDLESS_SURFACE_DETAIL".into(),
        ];
        for (index, name) in [
            "RAY_MATERIAL_FOOTPRINTS",
            "REGIONAL_HISTORY",
            "DLSS_RR_GUIDE_BUFFERS",
            "FOLIAGE_TRANSMISSION",
        ]
        .iter()
        .enumerate()
        {
            if bits & (1 << index) != 0 {
                definitions.push((*name).into());
            }
        }
        for consumer in consumers {
            let module = format!("bevy_solarik::{consumer}");
            let mut stage_definitions = definitions.clone();
            if consumer == "filtered_canopy_prepass" && bits & 1 != 0 {
                stage_definitions.push("WRITE_NORMAL".into());
            }
            if consumer == "realtime::world_cache_update" {
                stage_definitions.push("WORLD_CACHE_PRIORITY_ATOMIC_BUFFER".into());
                stage_definitions.push("ADAPTIVE_CACHE_BUDGET".into());
                if bits & 8 != 0 {
                    stage_definitions.push("WORLD_CACHE_QUERY_ATOMIC_MAX_LIFETIME".into());
                }
            }
            if consumer == "realtime::restir_gi" && bits & 8 != 0 {
                stage_definitions.push("WORLD_CACHE_FIRST_BOUNCE_LIGHT_LEAK_PREVENTION".into());
            }
            graph.validate(&module, &stage_definitions);
            if consumer == "realtime::primary_foliage" {
                let mut direct_definitions = stage_definitions.clone();
                direct_definitions.push("PRIMARY_CANOPY_DIRECT_ONLY".into());
                graph.validate(&module, &direct_definitions);
                stage_definitions.push("PRIMARY_CANOPY_QUERY_ONLY".into());
                graph.validate(&module, &stage_definitions);
            }
        }
    }
}
