//! Atmosphere contracts and numerical comparisons, independent of presentation exposure.
//! GPU tests are ignored by the ordinary gate and run explicitly without parallel captures.
use bevy_math::{DVec3, Vec3};
use wgpu::util::DeviceExt;

// An independent double-precision quadrature: metres, uniform midpoint steps,
// direct solar integration. Production WGSL works in km and uses LUTs.
fn reference_transmittance(state: &AtmosphereState, dir: Vec3, samples: usize) -> DVec3 {
    let radius = 6_360_000.0_f64;
    let origin = DVec3::Y * (radius + f64::from(state.observer_height));
    let dir = dir.as_dvec3();
    let b = origin.dot(dir);
    let ground_c = origin.length_squared() - radius * radius;
    if b < 0.0 && b * b >= ground_c {
        return DVec3::ZERO;
    }
    let top = 6_460_000.0_f64;
    let distance = -b + (b * b - origin.length_squared() + top * top).sqrt();
    let dt = distance / samples as f64;
    let mut optical = DVec3::ZERO;
    for i in 0..samples {
        let h = (origin + dir * ((i as f64 + 0.5) * dt)).length() - radius;
        let rayleigh = DVec3::new(5.802e-6, 13.558e-6, 33.1e-6)
            * f64::from(state.medium.rayleigh)
            * (-h / 8000.0).exp();
        let mie = DVec3::splat(4.44e-6 * f64::from(state.medium.mie) * (-h / 1200.0).exp());
        let ozone = DVec3::new(0.65e-6, 1.881e-6, 0.085e-6)
            * f64::from(state.medium.ozone)
            * (1.0 - (h - 25_000.0).abs() / 15_000.0).max(0.0);
        optical += (rayleigh + mie + ozone) * dt;
    }
    DVec3::new((-optical.x).exp(), (-optical.y).exp(), (-optical.z).exp())
}

#[test]
fn cpu_sun_attenuation_matches_dense_reference() {
    for height in [2.0, 1000.0, 20_000.0] {
        let state = AtmosphereState {
            observer_height: height,
            ..Default::default()
        };
        for angle in [90.0_f32, 30.0, 5.0, 0.0, -1.0, -10.0] {
            let dir = Vec3::new(angle.to_radians().cos(), angle.to_radians().sin(), 0.0);
            let expected = reference_transmittance(&state, dir, 4096);
            let actual = state.transmittance_to_space(dir).as_dvec3();
            assert!(
                (actual - expected).abs().max_element() <= 0.005,
                "height {height}, elevation {angle}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
#[ignore = "Vulkan GPU: run separately from builds and captures"]
fn atmosphere_production_math_gpu() {
    futures_lite::future::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Vulkan");
        let (device, queue) = adapter
            .request_device(&Default::default())
            .await
            .expect("device");
        let production = include_str!("../src/atmosphere/model.wgsl")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production atmosphere mathematics"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    r#"{production}
@group(0) @binding(0) var<uniform> parameters: AtmosphereParams;
@group(0) @binding(1) var<storage, read> directions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {{
    let d = directions[id.x].xyz;
    let origin = observer_position(parameters);
    result[id.x * 2u] = vec4(transmittance_to_space(parameters, origin, d, 128u), 1.0);
    let m = sample_medium(parameters, 0.0);
    result[id.x * 2u + 1u] = vec4(segment_integral(m.extinction, 0.001), 1.0);
}}"#
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("atmosphere contract probe"),
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        let angles = [90.0_f32, 30.0, 5.0, 0.0, -1.0, -10.0, -90.0];
        let directions: Vec<[f32; 4]> = angles
            .iter()
            .map(|a| [a.to_radians().cos(), a.to_radians().sin(), 0.0, 0.0])
            .collect();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&directions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (angles.len() * 32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for density in [0.0, 1.0, 8.0] {
            for height in [2.0, 1000.0, 20_000.0] {
                let mut state = AtmosphereState {
                    observer_height: height,
                    ..Default::default()
                };
                state.medium.rayleigh = density;
                state.medium.mie = density;
                state.medium.ozone = density;
                let values = [
                    [density, density, density, state.medium.mie_anisotropy],
                    [0.3, 0.3, 0.3, 1.0],
                    [0.0, 1.0, 0.0, state.sun_illuminance],
                    [0.0, 1.0, 0.0, state.moon_illuminance],
                    [height, 2000.0, 1.0, 32.0],
                ];
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&values),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: input.as_entire_binding(),
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
                    pass.dispatch_workgroups(angles.len() as u32, 1, 1);
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
                queue.submit([encoder.finish()]);
                let (send, recv) = std::sync::mpsc::channel();
                readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    send.send(r).expect("callback");
                });
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU done");
                recv.recv().expect("callback received").expect("mapped");
                let bytes = readback.slice(..).get_mapped_range();
                let results: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
                for (i, direction) in directions.iter().enumerate() {
                    let actual = DVec3::new(
                        f64::from(results[i * 2][0]),
                        f64::from(results[i * 2][1]),
                        f64::from(results[i * 2][2]),
                    );
                    let expected = reference_transmittance(
                        &state,
                        Vec3::new(direction[0], direction[1], direction[2]),
                        4096,
                    );
                    assert!(
                        actual.is_finite()
                            && actual.min_element() >= 0.0
                            && actual.max_element() <= 1.0
                    );
                    assert!(
                        (actual - expected).abs().max_element() <= 0.005,
                        "density {density}, height {height}, angle {}: {actual} vs {expected}",
                        angles[i]
                    );
                    let segment = &results[i * 2 + 1][..3];
                    assert!(
                        segment
                            .iter()
                            .all(|x| x.is_finite() && *x >= 0.0 && *x <= 0.001001)
                    );
                    if density == 0.0 {
                        assert!(
                            segment.iter().all(|x| (*x - 0.001).abs() < 1e-8),
                            "vacuum segment: {segment:?}"
                        );
                    }
                }
                drop(bytes);
                readback.unmap();
            }
        }
    });
}
use bevy_solarik::atmosphere::{AtmosphereMedium, AtmosphereState};

#[test]
fn atmosphere_rejects_nonphysical_controls() {
    let mut state = AtmosphereState::default();
    assert!(state.validate().is_ok());
    for value in [f32::NAN, f32::INFINITY, -1.0, 17.0] {
        state.medium.mie = value;
        assert!(
            state.validate().is_err(),
            "invalid aerosol density: {value}"
        );
    }
    state.medium = AtmosphereMedium::default();
    state.sun_direction = Vec3::ZERO;
    assert!(state.validate().is_err());
    state.sun_direction = Vec3::Y;
    state.medium.ground_albedo = Vec3::splat(1.01);
    assert!(state.validate().is_err());
}

#[test]
fn atmosphere_vacuum_preserves_illuminance_and_planet_shadow() {
    let mut state = AtmosphereState::default();
    state.medium.rayleigh = 0.0;
    state.medium.mie = 0.0;
    state.medium.ozone = 0.0;
    assert_eq!(state.transmittance_to_space(Vec3::Y), Vec3::ONE);
    assert_eq!(state.transmittance_to_space(Vec3::NEG_Y), Vec3::ZERO);
    state.sun_direction = Vec3::Y;
    assert_eq!(
        state.sun_illuminance_rgb(),
        Vec3::splat(state.sun_illuminance)
    );
    state.sun_illuminance *= 2.0;
    assert_eq!(state.sun_illuminance_rgb(), Vec3::splat(220_000.0));
}

#[test]
fn atmosphere_direct_light_is_bounded_and_reddens_at_sunset() {
    let state = AtmosphereState::default();
    let zenith = state.transmittance_to_space(Vec3::Y);
    let horizon = state.transmittance_to_space(Vec3::new(1.0, 0.02, 0.0).normalize());
    assert!(zenith.min_element() > 0.0 && zenith.max_element() <= 1.0);
    assert!(horizon.x > horizon.z && horizon.x < zenith.x);
    assert_eq!(state.transmittance_to_space(Vec3::NEG_Y), Vec3::ZERO);
}
