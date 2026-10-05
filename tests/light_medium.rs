//! Transmittance of a distant light through a planet's air, executed against
//! the production WGSL and compared with 64-bit quadrature of the same
//! profiles.
use wgpu::util::DeviceExt;

const RADIUS: f64 = 6_360_000.0;
const TOP: f64 = 6_460_000.0;
const RAYLEIGH: [f64; 3] = [5.802e-6, 13.558e-6, 33.1e-6];
const MIE: f64 = 4.44e-6;
const OZONE: [f64; 3] = [0.65e-6, 1.881e-6, 0.085e-6];
const FIXTURE: &str = "
@group(0) @binding(0) var<storage> cases: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
// A case: altitude of the receiver in metres, cosine of the zenith angle of
// the light, whether the receiver is the origin of the planet's frame.
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = cases[id.x];
    // Near the camera the scene is camera-relative: the planet centre is far
    // and the receiver is near the origin.
    let centre = select(vec3(0.0), vec3(0.0, -(RADIUS + c.x), 0.0), c.z > 0.5);
    var m = LightMedium(centre, RADIUS, vec3(5.802e-6, 13.558e-6, 33.1e-6), TOP,
        vec3(0.65e-6, 1.881e-6, 0.085e-6), 4.44e-6, vec4(0.0), array<vec4<f32>,5>());
    if c.w>0.5 {
        m.physical=array<vec4<f32>,5>(vec4(0.01,0.02,0.04,11.0),vec4(214.0,0.0,214.0,0.0),vec4(0.036363636,0.036363636,0.036363636,11.0),vec4(0.95,0.86,0.76,0.0),vec4(0.68,0.73,0.78,0.0));
    }
    if c.w>1.5 {
        m.physical=array<vec4<f32>,5>(vec4(0.3,0.7,1.7,15.9),vec4(737.0,8.0,235.0,0.0),vec4(1.9713,1.9713,1.9713,7.0),vec4(1.0,1.0,1.0,55.0),vec4(0.75,0.75,0.75,0.0));
    }
    let origin = centre + vec3(0.0, RADIUS + c.x, 0.0);
    let direction = vec3(sqrt(max(0.0, 1.0 - c.y * c.y)), c.y, 0.0);
    let none = LightMedium(centre, 0.0, m.rayleigh, TOP, m.ozone, m.mie, vec4(0.0), array<vec4<f32>,5>());
    output[id.x] = vec4(medium_transmittance(m, origin, direction), medium_transmittance(none, origin, direction).x);
}
";

/// Transmittance from `altitude` towards a source at zenith cosine `mu`.
fn reference(altitude: f64, mu: f64) -> [f64; 3] {
    let r = RADIUS + altitude;
    let b = r * mu;
    let far = -b + (b * b - r * r + TOP * TOP).max(0.0).sqrt();
    let near = (-b - (b * b - r * r + TOP * TOP).max(0.0).sqrt()).max(0.0);
    if far <= near {
        // The ray never enters the air.
        return [1.0; 3];
    }
    let steps = 200_000;
    let dt = (far - near) / steps as f64;
    let mut depth = [0.0; 3];
    for i in 0..steps {
        let t = near + (i as f64 + 0.5) * dt;
        let h = ((r * r + 2.0 * b * t + t * t).sqrt() - RADIUS).max(0.0);
        let ozone = (1.0 - (h - 25_000.0).abs() / 15_000.0).max(0.0);
        for channel in 0..3 {
            depth[channel] += (RAYLEIGH[channel] * (-h / 8000.0).exp()
                + MIE * (-h / 1200.0).exp()
                + OZONE[channel] * ozone)
                * dt;
        }
    }
    depth.map(|d| (-d).exp())
}

#[test]
#[ignore = "requires a Vulkan GPU; run separately from builds and scene captures"]
fn light_medium_gpu() {
    // Altitude, zenith cosine, tolerance on transmittance.
    let cases: Vec<[f32; 4]> = [
        (0.0, 1.0, 0.01),
        (0.0, 0.5, 0.01),
        (0.0, 0.1736, 0.015),
        (0.0, 0.0349, 0.02),
        (0.0, 0.0, 0.02),
        (3000.0, 0.0, 0.02),
        (3000.0, 0.7, 0.01),
        (20_000.0, 0.2, 0.01),
        // Above the air, looking past the planet through it.
        (150_000.0, -0.2, 0.01),
        (150_000.0, -0.2086, 0.03),
        // Above the air, looking away from it.
        (150_000.0, 0.5, 1e-6),
    ]
    .into_iter()
    .flat_map(|(h, mu, tolerance)| {
        [0.0, 1.0].map(|relative| [h as f32, mu as f32, relative, tolerance as f32])
    })
    .collect();
    let results = run(&cases);
    for (case, result) in cases.iter().zip(&results) {
        let expected = reference(f64::from(case[0]), f64::from(case[1]));
        for channel in 0..3 {
            assert!(
                (f64::from(result[channel]) - expected[channel]).abs() <= f64::from(case[3]),
                "altitude {} m, zenith cosine {}, camera-relative {}: {:?} against {expected:?}",
                case[0],
                case[1],
                case[2],
                &result[..3]
            );
        }
        assert_eq!(result[3], 1.0, "no medium transmits everything");
    }
    // The setting sun is red: blue is removed first, and only at low sun.
    let zenith = reference(0.0, 1.0);
    let horizon = reference(0.0, 0.0);
    assert!(zenith[2] > 0.7 && horizon[2] < 0.01 && horizon[0] > 10.0 * horizon[2]);
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
        let production = include_str!("../src/scene/light_medium.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "const RADIUS: f32 = {RADIUS:?};\nconst TOP: f32 = {TOP:?};\n{production}\n{FIXTURE}"
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production light medium"),
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

#[test]
#[ignore = "Vulkan GPU; serialize with builds and captures"]
fn physical_surface_sunlight_matches_independent_columns() {
    let cases: Vec<_> = [1.0_f32, 2.0]
        .into_iter()
        .flat_map(|profile| {
            [0.0_f32, 20000.0, 50000.0, 70000.0, 90000.0]
                .into_iter()
                .flat_map(move |h| [0.2_f32, 0.6, 1.0].map(move |mu| [h, mu, 1.0, profile]))
        })
        .collect();
    let actual = run(&cases);
    for (case, result) in cases.iter().zip(actual) {
        let r = RADIUS + f64::from(case[0]);
        let mu = f64::from(case[1]);
        let far = -r * mu + (r * r * mu * mu + TOP * TOP - r * r).sqrt();
        let dt = far / 20000.0;
        let mut tau = [0.0; 3];
        for i in 0..20000 {
            let t = (f64::from(i) + 0.5) * dt;
            let h = ((r * r + 2.0 * r * mu * t + t * t).sqrt() - RADIUS) * 0.001;
            let (density, a, beta) = if case[3] < 1.5 {
                (
                    (-h / 11.0).exp(),
                    (-h / 11.0).exp() * 0.4 / 11.0,
                    [0.01, 0.02, 0.04],
                )
            } else {
                let temp = (737.0 - 8.0 * h).max(235.0);
                (
                    (temp / 737.0).powf(737.0 / (8.0 * 15.9) - 1.0)
                        * (-(h - 62.75).max(0.0) / (15.9 * 235.0 / 737.0)).exp(),
                    1.9713 * (-0.5 * ((h - 55.0) / 7.0).powi(2)).exp(),
                    [0.3, 0.7, 1.7],
                )
            };
            for c in 0..3 {
                tau[c] += (beta[c] * density + a) * dt * 0.001;
            }
        }
        for c in 0..3 {
            let expected = (-tau[c]).exp();
            assert!(
                (f64::from(result[c]) - expected).abs() < 0.002,
                "case{case:?} channel{c} got{} expected{expected}",
                result[c]
            );
        }
    }
}
