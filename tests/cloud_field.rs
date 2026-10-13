//! Production advection/local-column contract, including both physical units.
#[path = "support/preprocess.rs"]
mod preprocess;
use wgpu::util::DeviceExt;

const FIXTURE: &str = r"
@group(0) @binding(0) var<storage> cases:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(2) var weather:texture_cube<f32>;
@group(0) @binding(3) var filtering:sampler;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    let c=cases[id.x];
    let unit=vec3(cos(c.y),sin(c.y),0.0);
    let motion=array<vec4<f32>,2>(vec4(c.x,0.7,1.3,0.0),vec4(0.5,1.1,0.04,0.0));
    let coordinates=weather_coordinates(unit,motion);
    output[id.x*4u]=weather_sample(weather,filtering,coordinates,0.0);
    let metre=weather_shadow_column(unit*(6371000.0+c.z),unit,6371800.0,6387500.0,motion,weather,filtering);
    let km=weather_shadow_column(unit*(6371.0+c.z*0.001),unit,6371.8,6387.5,motion,weather,filtering);
    var p:AtmosphereParams;
    p.planet.w=6371.0; p.shell=vec4(6471.0,1.0,60.0,137.0);
    p.cloud=vec4(0.8,16.5,1.0,0.0); p.weather_motion=motion;
    let local=cloud_weather(p,unit*(6371.0+c.z*0.001),0.0,weather,filtering);
    output[id.x*4u+1u]=vec4(metre,cloud_extinction(p,local,c.z*0.001));
    output[id.x*4u+2u]=vec4(km.y*1000.0,local.base*1000.0,local.top*1000.0,0.0);
    output[id.x*4u+3u]=vec4(coordinates.departure[0],0.0);
}
";

#[test]
#[ignore = "production Vulkan weather sampling; serialize GPU use"]
fn cloud_field_gpu_advection_columns_and_units() {
    let pi = core::f32::consts::PI;
    let cases = [
        [0.0, 0.0, 800.0, 0.0],
        [pi * 0.5, 0.0, 800.0, 0.0],
        [-0.0001, 0.5, 800.0, 0.0],
        [2.0 * pi - 0.0001, 0.5, 800.0, 0.0],
        [0.0001, 0.5, 800.0, 0.0],
        [2.0 * pi + 0.0001, 0.5, 800.0, 0.0],
        [0.0, 1.5, 800.0, 0.0],
        [0.0, 0.0, 8000.0, 0.0],
        [0.0, 0.0, 12000.0, 0.0],
    ];
    let values = run(&cases);
    assert!((values[0][0] - 64.0 / 255.0).abs() < 1e-6);
    assert!(
        (values[4][0] - 224.0 / 255.0).abs() < 1e-6,
        "quarter turn reaches -Z face"
    );
    for (a, b) in [(2, 3), (4, 5)] {
        for i in 0..4 {
            assert!(
                (values[a * 4][i] - values[b * 4][i]).abs() < 1e-5,
                "phase wrap in belt blend"
            );
        }
    }
    let top = 800.0 + 204.0 / 255.0 * 15700.0;
    let thickness = (top - 800.0) * 64.0 / 255.0;
    let base = top - thickness;
    for (case, result) in cases.iter().zip(values.chunks_exact(4)) {
        assert!(result.iter().flatten().all(|v| v.is_finite()));
        assert!((f64::from(result[2][1]) - base).abs() < 1.0);
        assert!((f64::from(result[2][2]) - top).abs() < 1.0);
        let start = ((f64::from(case[2]) - base) / thickness).clamp(0.0, 1.0);
        let step = (1.0 - start) / 4096.0;
        let integral = (0..4096)
            .map(|j| {
                let f = start + (f64::from(j) + 0.5) * step;
                if f < 0.85 {
                    (f / 0.85).powf(2.0 / 3.0)
                } else {
                    let u = (f - 0.85) / 0.15;
                    1.0 - u * u * (3.0 - 2.0 * u)
                }
            })
            .sum::<f64>()
            * step;
        let expected = 51.0 / 255.0 * thickness * integral;
        assert!(
            (f64::from(result[1][1]) / expected - 1.0).abs() < 0.005,
            "physical column"
        );
        assert!(
            (result[1][1] / result[2][0] - 1.0).abs() < 0.005,
            "metres and km agree"
        );
        if f64::from(case[2]) < base {
            assert_eq!(result[1][3], 0.0, "no extinction below local base");
        } else {
            assert!(result[1][3] > 0.0);
        }
        let angle = f64::from(case[0]);
        let lat = f64::from(case[1]);
        let reference = [lat.cos() * angle.cos(), lat.sin(), -lat.cos() * angle.sin()];
        let metres = reference
            .iter()
            .zip(result[3])
            .take(3)
            .map(|(a, b)| (a - f64::from(b)).powi(2))
            .sum::<f64>()
            .sqrt()
            * 6371000.0;
        assert!(metres < 4.0, "f32 departure error {metres} m");
    }
}

fn run(cases: &[[f32; 4]]) -> Vec<[f32; 4]> {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let source = format!(
            "{}\n{FIXTURE}",
            include_str!("../src/atmosphere/model.wesl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shared production cloud field"),
            source: preprocess::wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let weather = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pixels: Vec<u8> = (0..6).flat_map(|i| [64 + i * 32, 204, 51, 64]).collect();
        queue.write_texture(
            weather.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            weather.size(),
        );
        let view = weather.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&Default::default());
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(cases),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (cases.len() * 64) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &read, 0, size);
        queue.submit([encoder.finish()]);
        let (send, recv) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        recv.recv().unwrap().unwrap();
        bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec()
    })
}
