//! Exercise the production culling grid on Vulkan across the actual 1D limit.
#![expect(
    clippy::print_stdout,
    reason = "The GPU diagnostic emits actual dispatch coverage receipts."
)]
#[path = "support/preprocess.rs"]
mod preprocess;
#[path = "support/shader_graph.rs"]
mod shader_graph;
use wgpu::util::DeviceExt;

#[test]
fn production_cull_and_raster_shaders_compile() {
    use bevy_shader::ShaderDefVal;
    let mut graph = shader_graph::Graph::load();
    for (index, name) in [
        "cull_instances",
        "cull_bvh",
        "cull_clusters",
        "visibility_buffer_software_raster",
        "visibility_buffer_hardware_raster",
    ]
    .iter()
    .enumerate()
    {
        for variant in 0..2 {
            let mut defs = vec![
                ShaderDefVal::UInt("CULL_DISPATCH_WIDTH".into(), 65535),
                ShaderDefVal::UInt("RASTER_SUBPIXEL_BITS".into(), 8),
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
            graph.validate(&format!("bevy_pbr::meshlet::{name}"), &defs);
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
        let production = include_str!("../vendor/bevy_pbr/src/meshlet/cull_dispatch.wesl");
        let functions = production.split("fn cull_dispatch_size").nth(1).unwrap();
        let structure = production
            .split("struct CullDispatchArgs")
            .nth(1)
            .unwrap()
            .split("@if(")
            .next()
            .unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production cull dispatch boundary"),
            source: preprocess::wgsl(format!(
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
            let mapped = readback
                .slice(..)
                .get_mapped_range()
                .expect("completed GPU readback range");
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
