//! Light that leaves the bottom of a cloud, executed against the production
//! WGSL: the transmission used for cloud shadows on the ground and on the air
//! below the cloud must be the downward flux of the atmosphere pass's own
//! diffuse field, which is validated against Monte Carlo elsewhere.
use wgpu::util::DeviceExt;

const FIXTURE: &str = "
@group(0) @binding(0) var<storage> cases: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
// A case: cosine of the sun's zenith angle, vertical scaled optical depth.
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = cases[id.x];
    // The diffuse field at the slab's base over a black surface: I0 from light
    // travelling sideways, I1 from the difference with light travelling down.
    let level = cloud_diffuse(c.y, c.y, c.x, 0.0, 0.0);
    let down = cloud_diffuse(c.y, c.y, c.x, 1.0, 0.0);
    let i1 = (down - level) / CLOUD_DRAINE_MEAN_COSINE;
    let scaled = c.y * (1.0 - CLOUD_DRAINE_MEAN_COSINE * CLOUD_DRAINE_MEAN_COSINE);
    let field = exp(-scaled / c.x) + ATM_PI * (level + 2.0 / 3.0 * i1) / c.x;
    output[id.x] = vec4(cloud_slab_transmission(c.x, c.y), cloud_transmission(c.x, c.y), field, 0.0);
}
";

#[test]
#[ignore = "requires a Vulkan GPU; run separately from builds and scene captures"]
fn cloud_shadow_gpu() {
    let cases: Vec<[f32; 4]> = [1.0f32, 0.7, 0.4, 0.2]
        .into_iter()
        .flat_map(|mu| [0.0f32, 0.5, 2.0, 5.0, 15.0, 40.0].map(|depth| [mu, depth, 0.0, 0.0]))
        .collect();
    let results = run(&cases);
    for (case, result) in cases.iter().zip(&results) {
        let [atmosphere, scene, field, _] = *result;
        assert!(
            (atmosphere - scene).abs() < 1e-5,
            "ground and air use one transmission: {atmosphere} against {scene} for {case:?}"
        );
        if case[1] == 0.0 {
            assert_eq!(scene, 1.0, "no cloud, no shadow");
            continue;
        }
        assert!(
            (scene - field).abs() < 2e-3,
            "transmission {scene} against the diffuse field's flux {field} for {case:?}"
        );
        assert!(scene > 0.0 && scene < 1.0, "{case:?}: {scene}");
    }
    // Thicker cloud is darker beneath, at every sun height.
    for sun in results.chunks_exact(6) {
        assert!(
            sun.windows(2).all(|pair| pair[1][1] < pair[0][1]),
            "{sun:?}"
        );
    }
    // A cloud of optical depth 30 (scaled 15) under a high sun passes about a
    // third of the light: an overcast day, not night.
    let overcast = results[4][1];
    assert!((0.25..0.45).contains(&overcast), "{overcast}");
}

fn run(cases: &[[f32; 4]]) -> Vec<[f32; 4]> {
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
        let strip = |source: &str| {
            source
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let source = format!(
            "{}\n{}\n{FIXTURE}",
            strip(include_str!("../src/atmosphere/model.wgsl")),
            strip(include_str!("../src/scene/light_medium.wgsl"))
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production cloud shadow"),
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
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(cases),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (cases.len() * 16) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
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
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
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
