//! Executes the production LUT shaders and checks unexposed radiance in cd/m².
use bevy_math::{DVec3, Vec3};
use bevy_solarik::atmosphere::AtmosphereState;
use wgpu::util::DeviceExt;

fn lut_source(definition: &str) -> String {
    let model = include_str!("../src/atmosphere/model.wgsl")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let mut active = true;
    let body = include_str!("../src/atmosphere/luts.wgsl")
        .lines()
        .filter(|line| {
            if let Some(name) = line.strip_prefix("#ifdef ") {
                active = name == definition;
                return false;
            }
            if line.starts_with("#endif") {
                active = true;
                return false;
            }
            active && !line.starts_with('#')
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{model}\n{body}")
}

fn parameters(state: &AtmosphereState) -> [[f32; 4]; 5] {
    [
        [
            state.medium.rayleigh,
            state.medium.mie,
            state.medium.ozone,
            state.medium.mie_anisotropy,
        ],
        state
            .medium
            .ground_albedo
            .extend(f32::from(state.medium.multiple_scattering))
            .to_array(),
        state.sun_direction.extend(state.sun_illuminance).to_array(),
        state
            .moon_direction
            .extend(state.moon_illuminance)
            .to_array(),
        [
            state.observer_height,
            state.aerial_distance,
            state.stars,
            64.0,
        ],
    ]
}

fn mapped(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<[f32; 4]> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).expect("callback");
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU completed");
    recv.recv().expect("callback received").expect("mapped");
    let data = buffer.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    result
}

struct Luts {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    filtering: wgpu::Sampler,
}

impl Luts {
    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("Vulkan");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::FLOAT32_FILTERABLE,
                ..Default::default()
            })
            .await
            .expect("device");
        let pipelines = ["TRANSMITTANCE", "MULTIPLE", "SKY_VIEW", "CUBE"]
            .map(|name| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(lut_source(name).into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .into();
        let textures: Vec<_> = [(256, 64, 1), (32, 32, 1), (512, 256, 1), (256, 256, 6)]
            .map(|(w, h, d)| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: d,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            })
            .into();
        let views = textures
            .iter()
            .map(|t| t.create_view(&Default::default()))
            .collect();
        let filtering = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            device,
            queue,
            pipelines,
            textures,
            views,
            filtering,
        }
    }

    fn generate(&self, state: &AtmosphereState) {
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&parameters(state)),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let dimensions = [(32, 8, 1), (4, 4, 1), (64, 32, 1), (32, 32, 6)];
        let bindings: [&[usize]; 4] = [&[0], &[0, 1], &[0, 1, 2], &[2, 3]];
        for (index, pipeline) in self.pipelines.iter().enumerate() {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }];
            if index != 0 {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.filtering),
                });
            }
            entries.extend(bindings[index].iter().enumerate().map(|(i, &texture)| {
                wgpu::BindGroupEntry {
                    binding: i as u32 + 2,
                    resource: wgpu::BindingResource::TextureView(&self.views[texture]),
                }
            }));
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            let (x, y, z) = dimensions[index];
            pass.dispatch_workgroups(x, y, z);
        }
        self.queue.submit([encoder.finish()]);
    }

    fn pixels(&self, texture: usize) -> Vec<[f32; 4]> {
        let texture = &self.textures[texture];
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(texture.width() * texture.height() * texture.depth_or_array_layers())
                * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(texture.width() * 16),
                    rows_per_image: Some(texture.height()),
                },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        mapped(&self.device, &buffer)
    }
}

// A separate double-precision, metre-based midpoint integrator. The reference
// integrates each solar ray independently; it never samples production LUTs.
fn extinction(state: &AtmosphereState, point: DVec3) -> (DVec3, DVec3, DVec3) {
    let h = (point.length() - 6_360_000.0).max(0.0);
    let r = DVec3::new(5.802e-6, 13.558e-6, 33.1e-6)
        * f64::from(state.medium.rayleigh)
        * (-h / 8000.0).exp();
    let m = DVec3::splat(3.996e-6 * f64::from(state.medium.mie) * (-h / 1200.0).exp());
    let o = DVec3::new(0.65e-6, 1.881e-6, 0.085e-6)
        * f64::from(state.medium.ozone)
        * (1.0 - (h - 25000.0).abs() / 15000.0).max(0.0);
    (r, m, r + m / 0.9 + o)
}

fn exp3(value: DVec3) -> DVec3 {
    DVec3::new(value.x.exp(), value.y.exp(), value.z.exp())
}

fn solar_transmittance(state: &AtmosphereState, origin: DVec3) -> DVec3 {
    let direction = state.sun_direction.as_dvec3();
    let b = origin.dot(direction);
    let ground = origin.length_squared() - 6_360_000.0_f64.powi(2);
    if b < 0.0 && b * b >= ground {
        return DVec3::ZERO;
    }
    let distance = -b + (b * b - origin.length_squared() + 6_460_000.0_f64.powi(2)).sqrt();
    let dt = distance / 512.0;
    let optical: DVec3 = (0..512)
        .map(|i| extinction(state, origin + direction * ((f64::from(i) + 0.5) * dt)).2 * dt)
        .sum();
    exp3(-optical)
}

fn reference_radiance(state: &AtmosphereState, direction: DVec3) -> DVec3 {
    let origin = DVec3::Y * (6_360_000.0 + f64::from(state.observer_height));
    let b = origin.dot(direction);
    let distance = -b + (b * b - origin.length_squared() + 6_460_000.0_f64.powi(2)).sqrt();
    let dt = distance / 2048.0;
    let mu = direction
        .dot(state.sun_direction.as_dvec3())
        .clamp(-1.0, 1.0);
    let g = f64::from(state.medium.mie_anisotropy);
    let phase_r = 3.0 * (1.0 + mu * mu) / (16.0 * core::f64::consts::PI);
    let phase_m = 3.0 * (1.0 - g * g) * (1.0 + mu * mu)
        / (8.0 * core::f64::consts::PI * (2.0 + g * g) * (1.0 + g * g - 2.0 * g * mu).powf(1.5));
    let mut optical = DVec3::ZERO;
    let mut radiance = DVec3::ZERO;
    for i in 0..2048 {
        let point = origin + direction * ((f64::from(i) + 0.5) * dt);
        let (r, m, ext) = extinction(state, point);
        radiance += exp3(-(optical + ext * dt * 0.5))
            * solar_transmittance(state, point)
            * (r * phase_r + m * phase_m)
            * f64::from(state.sun_illuminance)
            * dt;
        optical += ext * dt;
    }
    radiance
}

fn sky_texel_direction(state: &AtmosphereState, x: usize, y: usize) -> DVec3 {
    let horizon = -(6_360_000.0 / (6_360_000.0 + f64::from(state.observer_height))).acos();
    let t = y as f64 / 255.0 * 2.0 - 1.0;
    let elevation =
        horizon + t.signum() * t * t * (core::f64::consts::FRAC_PI_2 - horizon * t.signum());
    let azimuth = ((x as f64 + 0.5) / 512.0 - 0.5) * core::f64::consts::TAU;
    DVec3::new(
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
        elevation.cos() * azimuth.cos(),
    )
}

#[test]
#[ignore = "Vulkan GPU; run serially without other builds or captures"]
fn production_luts_match_reference_and_physical_limits() {
    futures_lite::future::block_on(async {
        let luts = Luts::new().await;
        let mut worst = [0.0_f64; 2];
        for elevation in [40.0_f32, 3.0, -6.0, -18.0] {
            let mut state = AtmosphereState {
                moon_illuminance: 0.0,
                stars: 0.0,
                ..Default::default()
            };
            state.sun_direction = Vec3::new(
                elevation.to_radians().cos(),
                elevation.to_radians().sin(),
                0.0,
            );
            state.medium.multiple_scattering = false;
            luts.generate(&state);
            let pixels = luts.pixels(2);
            for y in [130, 144, 180, 230, 255] {
                for x in [0, 256, 383] {
                    let direction = sky_texel_direction(&state, x, y);
                    let reference = reference_radiance(&state, direction);
                    let actual = Vec3::from_slice(&pixels[y * 512 + x][..3]).as_dvec3();
                    let tolerance = if direction.y < 0.05 { 0.10 } else { 0.05 };
                    let relative = ((actual - reference).abs() / reference.max(DVec3::splat(0.1)))
                        .max_element();
                    let category = usize::from(direction.y < 0.05);
                    worst[category] = worst[category].max(relative);
                    assert!(
                        relative <= tolerance,
                        "sun {elevation} texel {x},{y} actual {actual} reference {reference} relative {relative}"
                    );
                }
            }
            state.medium.multiple_scattering = true;
            luts.generate(&state);
            let multiple = luts.pixels(2);
            for (single, multiple) in pixels.iter().zip(&multiple) {
                for c in 0..3 {
                    assert!(
                        multiple[c] + 1e-5 >= single[c],
                        "multiple scattering removed energy"
                    );
                }
            }
            state.sun_illuminance *= 0.5;
            luts.generate(&state);
            let half = luts.pixels(2);
            for (full, half) in multiple.iter().zip(&half) {
                for c in 0..3 {
                    assert!(
                        (half[c] * 2.0 - full[c]).abs() <= 0.001 * full[c].max(0.1),
                        "radiance must be linear in source lux"
                    );
                }
            }
        }
        use std::io::Write as _;
        writeln!(
            std::io::stdout(),
            "{:?}",
            ("atmosphere maximum relative error: away, grazing", worst)
        )
        .expect("report");
        for density in [0.0, 1.0, 8.0] {
            let mut state = AtmosphereState {
                moon_illuminance: 0.0,
                stars: 0.0,
                ..Default::default()
            };
            state.medium.rayleigh = density;
            state.medium.mie = density * 2.0;
            state.medium.ozone = density;
            state.medium.mie_anisotropy = 0.95;
            state.medium.ground_albedo = Vec3::ONE;
            luts.generate(&state);
            for texture in 0..4 {
                let pixels = luts.pixels(texture);
                for pixel in &pixels {
                    assert!(
                        pixel[..3].iter().all(|v| v.is_finite() && *v >= 0.0),
                        "texture {texture} invalid {pixel:?}"
                    );
                    if texture == 0 {
                        assert!(pixel[..3].iter().all(|v| *v <= 1.0));
                    }
                }
                if texture == 0 {
                    for y in [0, 1, 8, 32, 63] {
                        for x in [0, 16, 128, 254] {
                            let bottom = 6_360_000.0_f64;
                            let top = 6_460_000.0_f64;
                            let height = (top * top - bottom * bottom).sqrt();
                            let rho = height * y as f64 / 63.0;
                            let radius = (rho * rho + bottom * bottom).sqrt();
                            let distance =
                                (top - radius) + (rho + height - (top - radius)) * x as f64 / 255.0;
                            let mu = if distance < 1e-6 {
                                1.0
                            } else {
                                ((top * top - radius * radius - distance * distance)
                                    / (2.0 * radius * distance))
                                    .clamp(-1.0, 1.0)
                            };
                            let mut ray = state.clone();
                            ray.sun_direction =
                                Vec3::new((1.0 - mu * mu).sqrt() as f32, mu as f32, 0.0);
                            let reference = solar_transmittance(&ray, DVec3::Y * radius);
                            let actual = Vec3::from_slice(&pixels[y * 256 + x][..3]).as_dvec3();
                            assert!(
                                (actual - reference).abs().max_element() < 0.005,
                                "transmittance LUT density {density}, {x},{y}: {actual} vs {reference}"
                            );
                        }
                    }
                }
                if density == 0.0 && texture == 2 {
                    assert!(
                        pixels[128 * 512..]
                            .iter()
                            .all(|p| p[..3].iter().all(|v| *v == 0.0)),
                        "vacuum scatters no sky radiance"
                    );
                    let nadir = pixels[256];
                    assert!(
                        (nadir[0] - state.sun_illuminance / core::f32::consts::PI).abs() < 0.1,
                        "vacuum ground follows Lambert units: {nadir:?}"
                    );
                }
            }
        }
    });
}

#[test]
#[ignore = "Vulkan GPU; run serially without other builds or captures"]
fn procedural_stars_preserve_catalogue_flux_in_lighting_cube() {
    futures_lite::future::block_on(async {
        let luts = Luts::new().await;
        let mut state = AtmosphereState {
            sun_illuminance: 0.0,
            moon_illuminance: 0.0,
            stars: 1.0,
            ..Default::default()
        };
        state.medium.rayleigh = 0.0;
        state.medium.mie = 0.0;
        state.medium.ozone = 0.0;
        state.medium.ground_albedo = Vec3::ZERO;
        luts.generate(&state);
        let stars = luts.pixels(3);
        state.stars = 0.0;
        luts.generate(&state);
        let dark = luts.pixels(3);
        let mut actual = 0.0_f64;
        for (i, (star, black)) in stars.iter().zip(&dark).enumerate() {
            let x = (i % 256) as f64;
            let y = ((i / 256) % 256) as f64;
            let u = (x + 0.5) / 128.0 - 1.0;
            let v = (y + 0.5) / 128.0 - 1.0;
            let solid_angle = 4.0 / (256.0 * 256.0) * (1.0 + u * u + v * v).powf(-1.5);
            // Remove the known uniform airglow from upper-hemisphere texels.
            let face = i / (256 * 256);
            let sky = face == 2 || (face != 3 && v < 0.0);
            if sky {
                actual += (f64::from(star[0] - black[0]) - 0.00012) * solid_angle;
            }
        }
        let mut expected = 0.0;
        for cy in 64_u32..128 {
            for cx in 0_u32..256 {
                let mut hash = cx
                    .wrapping_mul(1973)
                    .wrapping_add(cy.wrapping_mul(9277))
                    .wrapping_add(89173);
                hash = (hash ^ (hash >> 16)).wrapping_mul(2246822519);
                hash = (hash ^ (hash >> 13)).wrapping_mul(3266489917);
                hash ^= hash >> 16;
                if hash % 1000 >= 15 {
                    continue;
                }
                let magnitude = 1.0 + 5.0 * f64::from(hash & 1023) / 1023.0;
                expected += 2.54e-6 * 10.0_f64.powf(-0.4 * magnitude);
            }
        }
        // 256² cubemap quadrature and the finite footprint may redistribute
        // unresolved stars; select the 15% integrated-flux budget before testing.
        assert!(
            (actual / expected - 1.0).abs() <= 0.15,
            "star flux {actual} lux vs catalogue {expected}"
        );
        use std::io::Write as _;
        writeln!(
            std::io::stdout(),
            "{:?}",
            ("star cube/catalogue lux", actual, expected)
        )
        .expect("report");
    });
}
