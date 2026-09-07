//! Metal traversal regression: CPU translation first; separately invoked tiny GPU probes.
#[test]
#[ignore = "Metal device required; 228 bytes of canonical geometry, no rays"]
fn metal_canonical_geometry_crosses_pages() {
    use bevy_mesh::{Indices, Mesh};
    use bevy_render::render_resource::PrimitiveTopology;
    let positions = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let normals = [[0.0f32, 1.0, 0.0]; 3];
    let uv0 = [[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]];
    let uv1 = [[0.7f32, 0.8], [0.9, 1.0], [1.1, 1.2]];
    let tangent = [[1.0f32, 0.0, 0.0, -1.0]; 3];
    let colours = [[0.2f32, 0.4, 0.6, 1.0]; 3];
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, Default::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangent.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours.to_vec());
    mesh.insert_indices(Indices::U32(vec![2, 0, 1]));
    let mut bytes = mesh.create_packed_vertex_buffer_data();
    let index_word = bytes.len() / 4;
    bytes.extend_from_slice(bytemuck::cast_slice(&[2u32, 0, 1]));
    let offset = |id| {
        mesh.attributes()
            .take_while(|(attribute, _)| attribute.id != id)
            .map(|(attribute, _)| attribute.format.size() / 4)
            .sum::<u64>()
    };
    let production = include_str!("../src/scene/raytracing_scene_bindings.wgsl");
    let structure = |name: &str| {
        let start = format!("struct {name} {{");
        format!(
            "{start}{}}}",
            production
                .split_once(&start)
                .expect("struct")
                .1
                .split_once('}')
                .expect("end")
                .0
        )
    };
    let bindings = production
        .split_once("@group(0) @binding(0)")
        .expect("page 0")
        .1
        .split_once("@group(0) @binding(2)")
        .expect("page end")
        .0;
    // Include every production accessor, including the page reader once present.
    let accessors = production
        .split_once("fn vertex2(")
        .expect("vertex read")
        .1
        .split_once("fn transform_positions(")
        .expect("accessor end")
        .0;
    let page_reader = if production.contains("fn geometry_word(") {
        shader_function(production, "geometry_word") + &shader_function(production, "vertex_word")
    } else {
        String::new()
    };
    let source = format!(
        "{}{}\n@group(0) @binding(0){}\n{}\nfn vertex2({}\n\
        @group(0) @binding(2) var<storage,read_write> output:array<vec4f>;\n\
        @compute @workgroup_size(1) fn probe() {{\n\
        let g=InstanceGeometryIds(0u,{}u,18u,0u,{}u,{}u,{}u,{}u,{}u,1u,0.0,0u);\n\
        let v=load_vertices(g,0u);\n\
        for(var i=0u;i<3u;i++) {{\n\
        output[i*5u]=vec4(v[i].position,1.0);\n\
        output[i*5u+1u]=vec4(v[i].normal,0.0);\n\
        output[i*5u+2u]=vec4(v[i].uv,v[i].uv1);\n\
        output[i*5u+3u]=v[i].tangent;\n\
        output[i*5u+4u]=v[i].colour;\n\
        }} }}",
        structure("InstanceGeometryIds"),
        structure("Vertex"),
        bindings,
        page_reader,
        accessors,
        index_word,
        offset(Mesh::ATTRIBUTE_NORMAL.id),
        offset(Mesh::ATTRIBUTE_UV_0.id),
        offset(Mesh::ATTRIBUTE_UV_1.id),
        offset(Mesh::ATTRIBUTE_TANGENT.id),
        offset(Mesh::ATTRIBUTE_COLOR.id)
    );
    let mut expected = Vec::<[f32; 4]>::new();
    for i in [2usize, 0, 1] {
        expected.extend([
            [positions[i][0], positions[i][1], positions[i][2], 1.0],
            [normals[i][0], normals[i][1], normals[i][2], 0.0],
            [uv0[i][0], uv0[i][1], uv1[i][0], uv1[i][1]],
            tangent[i],
            colours[i],
        ]);
    }
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Metal adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production paged vertex accessors"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 240,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 240,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Split inside a position, normal, optional attribute and index triplet,
        // then exercise a one-page pool with the required dummy second binding.
        for split in [1usize, 7, 29, 55, 57] {
            let part = |data: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: if data.is_empty() { &[0u8; 4] } else { data },
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let pages = [part(&bytes[..split * 4]), part(&bytes[split * 4..])];
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pages[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pages[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 240);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let start = std::time::Instant::now();
            loop {
                device.poll(wgpu::PollType::Poll).expect("nonblocking poll");
                if let Ok(r) = rx.try_recv() {
                    r.expect("mapped");
                    break;
                }
                assert!(
                    start.elapsed() < core::time::Duration::from_secs(5),
                    "geometry probe timeout"
                );
                std::thread::sleep(core::time::Duration::from_millis(1));
            }
            let data = readback.slice(..).get_mapped_range();
            let actual: &[[f32; 4]] = bytemuck::cast_slice(&data);
            assert_eq!(
                actual, expected,
                "canonical attributes and winding across page boundary {split}"
            );
            drop(data);
            readback.unmap();
        }
    });
}

use wgpu::util::DeviceExt;

fn production_shader() -> String {
    let source = include_str!("../src/scene/raytracing_scene_bindings.wgsl");
    let header = source
        .split("fn trace_ray_impl(")
        .nth(1)
        .expect("traversal");
    let (signature, body) = header
        .split_once("#ifdef SOLARIK_METAL_ALPHA")
        .expect("Metal path");
    let body = body.split_once("#else").expect("native path").0;
    let alpha = source
        .split("// Alpha test for a candidate hit")
        .nth(1)
        .expect("alpha");
    let alpha = alpha
        .split("fn sample_texture(")
        .next()
        .expect("material boundary");
    let flags = source
        .split("const MATERIAL_FLAG_OPAQUE")
        .nth(1)
        .expect("flags");
    let flags = flags
        .split("const MIRROR_ROUGHNESS_THRESHOLD")
        .next()
        .expect("flag boundary");
    format!(
        "enable wgpu_ray_query;\nconst MATERIAL_FLAG_OPAQUE{flags}\nfn trace_ray_impl({signature}{body}}}\n// Alpha test for a candidate hit{alpha}\n{FIXTURE}"
    )
}

const FIXTURE: &str = r#"
struct Material { flags:u32, alpha_cutoff:f32, base_color_alpha:f32, base_color_texture_id:u32 }
struct Vertex { uv:vec2f }
@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage> materials: array<Material>;
@group(0) @binding(2) var<storage,read_write> output: vec4f;
@group(0) @binding(3) var<uniform> config: vec4f;
var<private> geometry_ids: array<u32,3>;
const RAY_NO_CULL = 255u;
fn instance_material_id(i:u32)->u32 { return i; }
fn load_vertices(g:u32,t:u32)->array<Vertex,3> { return array(Vertex(vec2(0.0)),Vertex(vec2(0.0)),Vertex(vec2(0.0))); }
fn sample_texture_alpha(id:u32,uv:vec2f)->f32 { return 1.0; }
@compute @workgroup_size(1)
fn probe() {
    let hit=trace_ray_impl(vec3(config.w,0.0,0.0),vec3(0.0,0.0,1.0),config.y,config.z,RAY_FLAG_NONE,config.x>0.0);
    output=vec4(f32(hit.kind),hit.t,f32(hit.instance_index),f32(hit.primitive_index));
}
"#;

fn loop_count(block: &naga::Block) -> usize {
    block
        .iter()
        .map(|s| match s {
            naga::Statement::Loop {
                body, continuing, ..
            } => 1 + loop_count(body) + loop_count(continuing),
            naga::Statement::Block(b) => loop_count(b),
            naga::Statement::If { accept, reject, .. } => loop_count(accept) + loop_count(reject),
            naga::Statement::Switch { cases, .. } => {
                cases.iter().map(|c| loop_count(&c.body)).sum()
            }
            _ => 0,
        })
        .sum()
}

fn translate(source: &str) -> (String, usize) {
    let module = naga::front::wgsl::parse_str(source).expect("production WGSL parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("production WGSL validates");
    let traversal = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("trace_ray_impl"))
        .expect("traversal");
    let options = naga::back::msl::Options {
        lang_version: (2, 4),
        ..Default::default()
    };
    let (msl, _) = naga::back::msl::write_string(&module, &info, &options, &Default::default())
        .expect("MSL translation");
    (msl, loop_count(&traversal.1.body))
}

#[test]
fn metal_translation_has_only_the_bounded_recast_loop() {
    let source = production_shader();
    let (msl, loops) = translate(&source);
    assert_eq!(
        loops, 1,
        "only the bounded coverage loop may surround a Metal intersection"
    );
    assert!(
        msl.contains(".intersect("),
        "translate through the actual Metal intersector"
    );
    assert!(msl.contains("128u"), "bounded coverage stack");
    // Never dispatch this negative case. Naga accepts it, but its Metal ready
    // flag does not clear, so the nested proceed loop cannot terminate.
    let broken = source.replacen("rayQueryProceed(&rq);", "while rayQueryProceed(&rq) {}", 1);
    let (_, bad_loops) = translate(&broken);
    assert_eq!(
        bad_loops, 2,
        "the pre-fix GPU hang is caught without executing it"
    );
}

#[test]
#[ignore = "Metal device required; three triangles and one ray per case, after the CPU translation test"]
#[expect(
    unsafe_code,
    reason = "wgpu requires explicit opt-in to its experimental ray-query API"
)]
fn metal_three_triangle_visibility() {
    use core::time::Duration;
    use std::time::Instant;
    metal_translation_has_only_the_bounded_recast_loop();
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Metal adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: adapter.limits(),
                // SAFETY: required by wgpu for this deliberately minimal ray-query probe.
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await
            .expect("ray-query device");
        let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("single triangle"),
            contents: bytemuck::cast_slice(&[
                [-1.0f32, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: 3,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::empty(),
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("non-opaque probe triangle"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("three depth layers"),
            max_instances: 3,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        for i in 0..3 {
            tlas[i] = Some(wgpu::TlasInstance::new(
                &blas,
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    (i + 1) as f32,
                ],
                i as u32,
                255,
            ));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(
            &[wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: &vertex,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: None,
                        first_index: None,
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }],
            [&tlas],
        );
        queue.submit([encoder.finish()]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production bounded Metal traversal"),
            source: wgpu::ShaderSource::Wgsl(production_shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("one ray"),
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Flags: 1 opaque, 2 mask, 4 glass. Glass-aware rays retain a pane;
        // light rays skip it. Alpha comes from the production coverage function.
        let cases: [(&str, [u32; 3], [f32; 3], [f32; 4], f32); 8] = [
            (
                "nearest wall",
                [1u32, 1, 1],
                [1.0f32; 3],
                [0.0, 0.001, 10.0, 0.0],
                1.0,
            ),
            (
                "alpha hole",
                [2, 1, 1],
                [0.0, 1.0, 1.0],
                [0.0, 0.001, 10.0, 0.0],
                2.0,
            ),
            (
                "glass shadow",
                [4, 2, 1],
                [1.0, 0.0, 1.0],
                [0.0, 0.001, 10.0, 0.0],
                3.0,
            ),
            (
                "camera glass",
                [4, 2, 1],
                [1.0, 0.0, 1.0],
                [1.0, 0.001, 10.0, 0.0],
                1.0,
            ),
            (
                "all holes",
                [2, 2, 2],
                [0.0; 3],
                [0.0, 0.001, 10.0, 0.0],
                0.0,
            ),
            (
                "off triangle",
                [1, 1, 1],
                [1.0; 3],
                [0.0, 0.001, 10.0, 5.0],
                0.0,
            ),
            (
                "end before wall",
                [2, 2, 1],
                [0.0, 0.0, 1.0],
                [0.0, 0.001, 2.5, 0.0],
                0.0,
            ),
            (
                "mask threshold",
                [2, 2, 1],
                [0.49, 0.5, 1.0],
                [0.0, 0.001, 10.0, 0.0],
                2.0,
            ),
        ];
        for (name, flags, alpha, config, expected_t) in cases {
            let material: Vec<_> = flags
                .into_iter()
                .zip(alpha)
                .map(|(f, a)| [f, 0.5f32.to_bits(), a.to_bits(), u32::MAX])
                .collect();
            let material = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(&material),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tlas.as_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: material.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: config.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 16);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let start = Instant::now();
            loop {
                device
                    .poll(wgpu::PollType::Poll)
                    .expect("nonblocking GPU poll");
                if let Ok(result) = rx.try_recv() {
                    result.expect("mapped");
                    break;
                }
                assert!(
                    start.elapsed() < Duration::from_secs(5),
                    "{name}: GPU did not complete one ray"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            let data = readback.slice(..).get_mapped_range();
            let hit: &[f32] = bytemuck::cast_slice(&data);
            if expected_t == 0.0 {
                assert_eq!(hit[0], 0.0, "{name}: must miss");
            } else {
                assert_eq!(hit[0], 1.0, "{name}: triangle hit");
                assert!(
                    (hit[1] - expected_t).abs() < 1e-5,
                    "{name}: wrong distance {hit:?}"
                );
                assert_eq!(
                    hit[2],
                    expected_t - 1.0,
                    "{name}: nearest surviving instance"
                );
            }
            drop(data);
            readback.unmap();
        }
    });
}

fn shader_function(source: &str, name: &str) -> String {
    let signature = format!("fn {name}(");
    let body = source
        .split_once(&signature)
        .expect("production function")
        .1
        .split_once("\n}")
        .expect("function end")
        .0;
    format!("{signature}{body}\n}}\n")
}

#[test]
#[ignore = "Metal device required; production light-sampler integer compute, no rays"]
fn metal_large_emitter_sampling() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Metal adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let utils = include_str!("bevy_pbr_rng.wgsl");
        let sampling = include_str!("../src/scene/sampling.wgsl");
        let source = format!(
            "{}{}{}{}",
            shader_function(utils, "rand_u"),
            shader_function(utils, "rand_range_u"),
            shader_function(sampling, "emissive_primitive_sample"),
            r#"
@group(0) @binding(0) var<storage,read_write> samples:array<vec4u>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id:vec3u) {
    let first = emissive_primitive_sample(id.x,130001u);
    let replay = emissive_primitive_sample(id.x,130001u);
    samples[id.x]=vec4(first,replay);
}"#
        );
        let module = naga::front::wgsl::parse_str(&source).expect("production sampler parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("valid sampler");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("full-width production emitter sampler"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4096 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4096 * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let start = std::time::Instant::now();
        loop {
            device.poll(wgpu::PollType::Poll).expect("nonblocking poll");
            if let Ok(result) = rx.try_recv() {
                result.expect("mapped");
                break;
            }
            assert!(
                start.elapsed() < core::time::Duration::from_secs(5),
                "sampler did not complete"
            );
            std::thread::sleep(core::time::Duration::from_millis(1));
        }
        let data = readback.slice(..).get_mapped_range();
        let samples: &[[u32; 4]] = bytemuck::cast_slice(&data);
        let mut bins = [0usize; 8];
        for s in samples {
            assert_eq!(
                s[..2],
                s[2..],
                "reservoir reuse must replay the identical primitive and point"
            );
            assert!(s[0] < 130001);
            bins[(s[0] as usize * 8) / 130001] += 1;
        }
        for count in bins {
            assert!(
                (400..=624).contains(&count),
                "uniform primitive selection: {bins:?}"
            );
        }
        assert!(
            samples.iter().any(|s| s[0] > 120_000),
            "the old 16-bit truncation loses the second half"
        );
        drop(data);
        readback.unmap();
    });
}
