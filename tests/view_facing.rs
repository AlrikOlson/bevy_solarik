//! Shading normals that face away from the viewer, executed against the
//! production BRDF WGSL.
//!
//! A visible surface point faces the direction it is seen from. Interpolated
//! and mapped shading normals can point past the viewer's horizon at grazing
//! angles; a BRDF that returns zero there outlines the silhouette of every
//! normal-mapped surface in black pixels.

const SAMPLES: usize = 65536;

fn run_probe() -> Vec<[f32; 4]> {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Vulkan adapter");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let production = include_str!("../src/scene/brdf.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.starts_with("enable "))
            .collect::<Vec<_>>()
            .join("\n");
        // A stand-in table: the Fresnel limit for a smooth interface, and a
        // constant directional albedo for a rough one.
        let production = production.replace(
            "return textureSampleLevel(brdf_dfg_lut, brdf_dfg_lut_sampler, vec2<f32>(NdotV, perceptual_roughness), 0.0).rg;",
            "return select(vec2(1.0 - pow(1.0 - NdotV, 5.0), pow(1.0 - NdotV, 5.0)), vec2(0.5, 0.05), perceptual_roughness > 0.5);"
        );
        let source = format!("{production}\n{}", include_str!("view_facing_fixture.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production BRDF, view-facing normals"),
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
            size: (SAMPLES * 4 * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
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
            pass.dispatch_workgroups((SAMPLES / 64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).expect("map result");
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU completed");
        receiver.recv().expect("map callback").expect("map");
        let data = readback.slice(..).get_mapped_range();
        bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
    })
}

#[test]
#[ignore = "requires a Vulkan GPU; run separately from builds and scene captures"]
fn view_facing_gpu() {
    let samples = run_probe();
    let (mut away, mut lit_away, mut sampled_away) = (0usize, 0usize, 0usize);
    // Diffuse seen at a cosine of 0.02 and lit along the normal, for a base
    // colour of 0.5 and F0 = 0.16 * 0.5^2 = 0.04. The specular layer takes
    // its directional albedo on the way in and on the way out.
    let lambert = 0.5 / core::f32::consts::PI;
    // The fixture's rough table is constant: 0.04 * 0.5 + 0.05 either way.
    let rough = lambert * (1.0 - 0.07) * (1.0 - 0.07);
    // A smooth interface reflects its Fresnel reflectance.
    let schlick = |cosine: f32| 0.04 + 0.96 * (1.0 - cosine).powi(5);
    let smooth = lambert * (1.0 - schlick(1.0)) * (1.0 - schlick(0.02));
    assert!(
        (samples[3][0] - rough).abs() < 1e-5,
        "rough diffuse at grazing view follows the table: {} against {rough}",
        samples[3][0]
    );
    assert!(
        (samples[3][1] - smooth).abs() < 1e-5,
        "smooth diffuse at grazing view keeps the Fresnel limit: {} against {smooth}",
        samples[3][1]
    );
    assert!(
        rough > 5.0 * smooth,
        "a rough surface seen edge-on is not dark"
    );
    for sample in samples.chunks_exact(4) {
        assert!(sample.iter().flatten().all(|x| x.is_finite()));
        let (given, mirrored) = (sample[0], sample[1]);
        let cosine = mirrored[3];
        let (mirrored_pdf, light_cosine, sampled_pdf, sampled_throughput) =
            (sample[2][0], sample[2][1], sample[2][2], sample[2][3]);
        // The mirrored normal is the given one when it already faces the
        // viewer, so this also pins the unchanged case.
        for (a, b) in given[..3].iter().zip(&mirrored[..3]) {
            assert!(
                (a - b).abs() < 1e-5,
                "a normal {cosine} to the viewer shades as its mirror image: {a} against {b}"
            );
        }
        assert!((given[3] - mirrored_pdf).abs() < 1e-5, "MIS PDF");
        if cosine > -0.01 {
            continue;
        }
        away += 1;
        if light_cosine > 0.05 {
            lit_away += 1;
            assert!(
                given[1] > 0.0,
                "a lit surface seen past its shading normal's horizon is not black"
            );
        }
        if sampled_pdf > 0.0 && sampled_throughput > 0.0 {
            sampled_away += 1;
        }
    }
    // Half of the uniformly drawn normals face away; nearly half of those
    // see the light, and continuation sampling works for all of them.
    assert!(away > SAMPLES * 2 / 5);
    assert!(lit_away > away * 2 / 5);
    assert!(sampled_away > away * 9 / 10);
}
