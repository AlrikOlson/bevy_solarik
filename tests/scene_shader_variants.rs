//! Compile complete production consumers with Bevy's real shader cache, without a scene load.
use bevy_asset::Assets;
use bevy_shader::{Shader, ShaderCache, ShaderCacheSource, ShaderDefVal};
use std::path::{Path, PathBuf};
fn sources(root: &Path, result: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, result);
        } else if path.extension().is_some_and(|v| v == "wgsl") {
            result.push(path);
        }
    }
}
#[test]
fn complete_scene_consumers_compile_with_supported_shader_variants() {
    const {
        assert!(
            cfg!(debug_assertions),
            "Bevy's validating composer is required"
        );
    }
    let render = bevy_platform::future::block_on(
        bevy_render::renderer::initialize_headless_renderer(&bevy_render::settings::WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: bevy_solarik::SolarikPlugins::required_wgpu_features(),
            ..Default::default()
        }),
    );
    let mut cache = ShaderCache::new(
        (),
        render.0.features(),
        render.3.get_downlevel_capabilities().flags,
        |_, source, _| match source {
            ShaderCacheSource::Naga(module) => Ok(module.entry_points.len()),
            _ => panic!("expected the production Naga composer"),
        },
    );
    let mut assets = Assets::<Shader>::default();
    let mut inputs = Vec::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    sources(&root.join("src"), &mut inputs);
    for path in std::env::var("SOLARIK_SHADER_ROOTS")
        .expect("wrapper supplies locked Bevy sources")
        .lines()
    {
        sources(Path::new(path), &mut inputs);
    }
    inputs.sort();
    inputs.dedup();
    let mut targets = Vec::new();
    let consumers = [
        "src/scene/sampling.wgsl",
        "src/realtime/primary_foliage.wgsl",
        "src/realtime/primary_glass.wgsl",
        "src/realtime/restir_gi.wgsl",
        "src/realtime/world_cache_update.wgsl",
        "src/realtime/specular_gi.wgsl",
        "src/realtime/surface_path.wgsl",
        "src/pathtracer/pathtracer.wgsl",
    ];
    for path in inputs {
        let source = std::fs::read_to_string(&path).unwrap();
        let mut shader = Shader::from_wgsl(source, path.to_string_lossy().into_owned());
        // Same imported module settings as MeshRenderPlugin in pinned Bevy.
        if path.ends_with("mesh_view_types.wgsl") {
            shader.shader_defs = vec![
                ShaderDefVal::UInt(
                    "MAX_DIRECTIONAL_LIGHTS".into(),
                    bevy_pbr::MAX_DIRECTIONAL_LIGHTS as u32,
                ),
                ShaderDefVal::UInt(
                    "MAX_CASCADES_PER_LIGHT".into(),
                    bevy_pbr::MAX_CASCADES_PER_LIGHT as u32,
                ),
                ShaderDefVal::UInt("MAX_RECT_LIGHTS".into(), bevy_pbr::MAX_RECT_LIGHTS as u32),
            ];
        }
        let handle = assets.add(shader.clone());
        cache.set_shader(handle.id(), shader);
        if consumers.iter().any(|p| path == root.join(p)) {
            targets.push((path, handle));
        }
    }
    assert_eq!(targets.len(), consumers.len());
    for bits in 0..16 {
        let mut definitions = vec![
            ShaderDefVal::UInt("WORLD_CACHE_SIZE".into(), 1 << 20),
            ShaderDefVal::UInt(
                "AVAILABLE_STORAGE_BUFFER_BINDINGS".into(),
                render.0.limits().max_storage_buffers_per_shader_stage,
            ),
            "BINDLESS_SURFACE_DETAIL".into(),
        ];
        for (index, name) in [
            "RAY_MATERIAL_FOOTPRINTS",
            "REGIONAL_HISTORY",
            "DLSS_RR_GUIDE_BUFFERS",
            "FOLIAGE_TRANSMISSION",
        ]
        .into_iter()
        .enumerate()
        {
            if bits & (1 << index) != 0 {
                definitions.push(name.into());
            }
        }
        for (path, handle) in &targets {
            let mut stage_definitions = definitions.clone();
            // The production pipeline factory specializes the shared world-cache module.
            if path.ends_with("world_cache_update.wgsl") {
                stage_definitions.push("WORLD_CACHE_PRIORITY_ATOMIC_BUFFER".into());
                stage_definitions.push("ADAPTIVE_CACHE_BUDGET".into());
                if bits & 8 != 0 {
                    stage_definitions.push("WORLD_CACHE_QUERY_ATOMIC_MAX_LIFETIME".into());
                }
            }
            if path.ends_with("restir_gi.wgsl") && bits & 8 != 0 {
                stage_definitions.push("WORLD_CACHE_FIRST_BOUNCE_LIGHT_LEAK_PREVENTION".into());
            }
            if let Err(error) = cache.get(bits, handle.id(), &stage_definitions) {
                panic!("{} variant{bits}: {error:?}", path.display());
            }
        }
    }
}
