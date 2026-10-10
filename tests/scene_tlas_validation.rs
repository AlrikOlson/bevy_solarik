//! Safe-core resource deduplication must retain all distinct BLAS validation.
use bevy_render::renderer::initialize_headless_renderer;
use bevy_render::settings::WgpuSettings;
use wgpu::util::DeviceExt;

#[test]
fn repeated_instances_preserve_unique_resource_validation() {
    bevy_platform::future::block_on(async {
        let resources = initialize_headless_renderer(&WgpuSettings {
            backends: Some(wgpu::Backends::VULKAN),
            features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            ..Default::default()
        })
        .await;
        let (device, queue) = (resources.0.wgpu_device(), resources.1);
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[-0.4f32, -0.4, 0., 0.4, -0.4, 0., 0., 0.4, 0.]),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: 3,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let create = || {
            device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size.clone()],
                },
            )
        };
        let built = create();
        let later = create();
        let build = |blas: &wgpu::Blas| {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(
                &[wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size: &size,
                            vertex_buffer: &vertices,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: None,
                            first_index: None,
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                }],
                &[],
            );
            queue.submit([encoder.finish()]);
        };
        build(&built);
        let mut scene = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            max_instances: 1024,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        let transform = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.];
        for index in 0..1024 {
            scene[index] = Some(wgpu::TlasInstance::new(
                &built,
                transform,
                index as u32,
                255,
            ));
        }
        scene[1023] = Some(wgpu::TlasInstance::new(&later, transform, 1023, 255));
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(&[], [&scene]);
        queue.submit([encoder.finish()]);
        let error = errors.pop().await.expect("unbuilt distinct BLAS must fail");
        assert!(error.to_string().contains("before it is built"), "{error}");

        build(&later);
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(&[], [&scene]);
        queue.submit([encoder.finish()]);
        assert!(errors.pop().await.is_none());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r#"
enable wgpu_ray_query;
@group(0) @binding(0) var scene: acceleration_structure;
@compute @workgroup_size(1) fn main() {
    var q: ray_query;
    rayQueryInitialize(&q, scene, RayDesc(0u,255u,0.001,10.,vec3(0.,0.,-2.),vec3(0.,0.,1.)));
    while rayQueryProceed(&q) {}
}
"#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene.as_binding(),
            }],
        });
        build(&later);
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let error = errors
            .pop()
            .await
            .expect("BLAS rebuilt after TLAS must fail");
        assert!(error.to_string().contains("newer than"), "{error}");

        scene[1023] = Some(wgpu::TlasInstance::new(&built, transform, 1 << 24, 255));
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.build_acceleration_structures(&[], [&scene]);
        queue.submit([encoder.finish()]);
        let error = errors
            .pop()
            .await
            .expect("invalid custom index on repeated BLAS must fail");
        assert!(error.to_string().contains("24"), "{error}");
    });
}
