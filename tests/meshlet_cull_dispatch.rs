//! Exercise the production culling grid on Vulkan across the actual 1D limit.
use wgpu::util::DeviceExt;

#[test]
fn production_cull_and_raster_shaders_compile() {
    use bevy_asset::Assets;
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource, ShaderDefVal};
    use std::path::{Path, PathBuf};
    fn sources(root: &Path, result: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sources(&path, result);
            } else if path.extension().is_some_and(|v| v == "wgsl") {
                result.push(path);
            }
        }
    }
    const {
        assert!(cfg!(debug_assertions), "validating Bevy composer required");
    }
    let render = bevy_platform::future::block_on(
        bevy_render::renderer::initialize_headless_renderer(&bevy_render::settings::WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::TEXTURE_INT64_ATOMIC
                | wgpu::Features::TEXTURE_ATOMIC
                | wgpu::Features::SHADER_INT64
                | wgpu::Features::SUBGROUP
                | wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::IMMEDIATES,
            ..Default::default()
        }),
    );
    let mut cache = ShaderCache::new(
        (),
        render.0.features(),
        render.3.get_downlevel_capabilities().flags,
        |_, source, _| match source {
            ShaderCacheSource::Naga(module) => Ok(module.entry_points.len()),
            _ => panic!("expected production Naga composer"),
        },
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/bevy_pbr/src/meshlet");
    let names = [
        "cull_instances.wgsl",
        "cull_bvh.wgsl",
        "cull_clusters.wgsl",
        "visibility_buffer_software_raster.wgsl",
        "visibility_buffer_hardware_raster.wgsl",
    ];
    let mut files = Vec::new();
    for path in std::env::var("SOLARIK_SHADER_ROOTS").unwrap().lines() {
        sources(Path::new(path), &mut files);
    }
    files.sort();
    files.dedup();
    let mut assets = Assets::<Shader>::default();
    let mut targets = Vec::new();
    for path in files {
        let mut shader = Shader::from_wgsl(
            std::fs::read_to_string(&path).unwrap(),
            path.to_string_lossy().into_owned(),
        );
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
        if let Some(index) = names.iter().position(|name| path == root.join(name)) {
            targets.push((index, handle));
        }
    }
    assert_eq!(targets.len(), names.len());
    for (index, handle) in targets {
        for variant in 0..2 {
            let mut defs = vec![
                ShaderDefVal::UInt(
                    "CULL_DISPATCH_WIDTH".into(),
                    render.0.limits().max_compute_workgroups_per_dimension,
                ),
                ShaderDefVal::UInt(
                    "AVAILABLE_STORAGE_BUFFER_BINDINGS".into(),
                    render.0.limits().max_storage_buffers_per_shader_stage,
                ),
                ShaderDefVal::UInt(
                    "RASTER_SUBPIXEL_BITS".into(),
                    render
                        .3
                        .raster_subpixel_precision_bits()
                        .unwrap_or(4)
                        .min(24),
                ),
            ];
            if index < 3 {
                defs.push(
                    [
                        "MESHLET_INSTANCE_CULLING_PASS",
                        "MESHLET_BVH_CULLING_PASS",
                        "MESHLET_CLUSTER_CULLING_PASS",
                    ][index]
                        .into(),
                );
                defs.push(
                    if variant == 0 {
                        "MESHLET_FIRST_CULLING_PASS"
                    } else {
                        "MESHLET_SECOND_CULLING_PASS"
                    }
                    .into(),
                );
            } else {
                defs.push("MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into());
                defs.push("MESHLET_2D_DISPATCH".into());
                if variant == 0 {
                    defs.push("MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into());
                }
            }
            cache
                .get(index * 2 + variant, handle.id(), &defs)
                .unwrap_or_else(|error| panic!("{} variant {variant}: {error:?}", names[index]));
        }
    }
}

#[test]
#[ignore = "requires Vulkan; serialize with builds and captures"]
fn cull_grid_visits_every_item_once_across_limit_and_reuse() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let width = device.limits().max_compute_workgroups_per_dimension;
        let boundary = width * 128;
        let maximum = boundary + 257_588;
        // Use the exact production functions, with the same pipeline constant.
        let production = include_str!("../vendor/bevy_pbr/src/meshlet/cull_dispatch.wgsl");
        let functions = production.split("fn cull_dispatch_size").nth(1).unwrap();
        let structure = production
            .split("struct CullDispatchArgs")
            .nth(1)
            .unwrap()
            .split("#ifdef")
            .next()
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production cull dispatch boundary"),
            source: wgpu::ShaderSource::Wgsl(format!(
                "struct CullDispatchArgs{structure}\nconst CULL_WIDTH:u32={width}u;\nfn cull_dispatch_size{functions}\n{}",
                include_str!("meshlet_cull_dispatch.wgsl")
            ).into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let producer = pipeline("produce");
        let consumer = pipeline("consume");
        let dispatch = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0u32, 0, 1]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let parameters = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visits = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(maximum + 2) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: visits.size() + 12,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let producer_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &producer.get_bind_group_layout(0),
            entries: &[(&dispatch, 0), (&parameters, 1)].map(|(buffer, binding)| {
                wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }
            }),
        });
        let consumer_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &consumer.get_bind_group_layout(0),
            entries: &[(&parameters, 1), (&visits, 2)].map(|(buffer, binding)| {
                wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }
            }),
        });
        for (count, items) in [
            (0, 128),
            (1, 128),
            (boundary - 1, 128),
            (boundary, 128),
            (boundary + 1, 128),
            (maximum, 128),
            (width * 16 + 1, 16),
            (17, 128),
            (0, 128),
        ] {
            queue.write_buffer(&parameters, 0, bytemuck::cast_slice(&[count, items, 0, 0]));
            let mut encoder = device.create_command_encoder(&Default::default());
            // Match the production ping-pong reset: clear both growing axes.
            encoder.clear_buffer(&dispatch, 0, Some(8));
            encoder.clear_buffer(&visits, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&producer);
                pass.set_bind_group(0, &producer_group, &[]);
                pass.dispatch_workgroups(4, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&consumer);
                pass.set_bind_group(0, &consumer_group, &[]);
                pass.dispatch_workgroups_indirect(&dispatch, 0);
            }
            encoder.copy_buffer_to_buffer(&dispatch, 0, &readback, 0, 12);
            encoder.copy_buffer_to_buffer(&visits, 0, &readback, 12, visits.size());
            queue.submit([encoder.finish()]);
            let (send, receive) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                send.send(r).unwrap();
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            receive.recv().unwrap().unwrap();
            let mapped = readback.slice(..).get_mapped_range();
            let values: &[u32] = bytemuck::cast_slice(&mapped);
            let grid = &values[..3];
            assert!(grid[0] <= width && grid[1] <= width);
            assert_eq!(grid[2], 1);
            assert!(
                values[3..3 + count as usize].iter().all(|&v| v == 1),
                "missing or duplicated item for count={count} grid={grid:?}"
            );
            assert!(
                values[3 + count as usize..].iter().all(|&v| v == 0),
                "tail overwritten for count={count}"
            );
            println!(
                "CULL_DISPATCH count={count} items_per_group={items} device_limit={width} grid={grid:?} exact_visits=true tail_clean=true"
            );
            drop(mapped);
            readback.unmap();
        }
    });
}
