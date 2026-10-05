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

fn parameters(state: &AtmosphereState) -> [[f32; 4]; 10] {
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
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [0.0; 4],
        [state.sun_angular_radius, 0.00452, 0.0, 0.0],
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
        self.generate_parameters(&parameters(state));
    }

    fn generate_parameters(&self, parameters: &[[f32; 4]; 10]) {
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(parameters),
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

fn orbit_reference(
    state: &AtmosphereState,
    origin: DVec3,
    dir: DVec3,
    maximum: f64,
) -> (DVec3, DVec3) {
    let interval = |radius: f64| {
        let b = origin.dot(dir);
        let delta = b * b - origin.length_squared() + radius * radius;
        if delta < 0.0 {
            None
        } else {
            Some((-b - delta.sqrt(), -b + delta.sqrt()))
        }
    };
    let Some((entry, exit)) = interval(6_460_000.0) else {
        return (DVec3::ZERO, DVec3::ONE);
    };
    let start = entry.max(0.0);
    let mut end = exit.min(maximum);
    if let Some((ground, _)) = interval(6_360_000.0)
        && ground > 0.0
    {
        end = end.min(ground);
    }
    if end <= start {
        return (DVec3::ZERO, DVec3::ONE);
    }
    let dt = (end - start) / 2048.0;
    let mu = dir.dot(state.sun_direction.as_dvec3());
    let g = f64::from(state.medium.mie_anisotropy);
    let pr = 3.0 * (1.0 + mu * mu) / (16.0 * core::f64::consts::PI);
    let pm = 3.0 * (1.0 - g * g) * (1.0 + mu * mu)
        / (8.0 * core::f64::consts::PI * (2.0 + g * g) * (1.0 + g * g - 2.0 * g * mu).powf(1.5));
    let mut optical = DVec3::ZERO;
    let mut radiance = DVec3::ZERO;
    for i in 0..2048 {
        let point = origin + dir * (start + (f64::from(i) + 0.5) * dt);
        let (r, m, e) = extinction(state, point);
        radiance += exp3(-(optical + e * dt * 0.5))
            * solar_transmittance(state, point)
            * (r * pr + m * pm)
            * f64::from(state.sun_illuminance)
            * dt;
        optical += e * dt;
    }
    (radiance, exp3(-optical))
}

/// Planetary view transport including the cloud layer, as composited.
const VIEW_PROBE: &str = "integrate_view(p,observer_position(p),d.xyz,d.w,u32(p.observer.w)*2u,0.0,trans,multiple,filtering,weather)";

impl Luts {
    /// Clear-air transport, as used by lookup fields.
    fn probe(&self, params: &[[f32; 4]; 10], directions: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.probe_with(
            "integrate_atmosphere(p,observer_position(p),d.xyz,d.w,u32(p.observer.w)*2u,trans,multiple,filtering,false)",
            params,
            directions,
        )
    }

    /// Planetary view transport including the cloud shell, as composited.
    fn probe_view(&self, params: &[[f32; 4]; 10], directions: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.probe_with(VIEW_PROBE, params, directions)
    }

    fn probe_with(
        &self,
        call: &str,
        params: &[[f32; 4]; 10],
        directions: &[[f32; 4]],
    ) -> Vec<[f32; 4]> {
        self.probe_program("", call, [0; 4], params, directions)
    }

    /// Runs `call` once per input vector `d`, with optional WGSL `helpers`
    /// compiled after the production model. A call that names `weather` gets a
    /// cube weather map filled with `weather_texel`.
    fn probe_program(
        &self,
        helpers: &str,
        call: &str,
        weather_texel: [u8; 4],
        params: &[[f32; 4]; 10],
        directions: &[[f32; 4]],
    ) -> Vec<[f32; 4]> {
        let model = include_str!("../src/atmosphere/model.wgsl")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let code = format!(
            r#"{model}
@group(0) @binding(0) var<uniform> p: AtmosphereParams;
@group(0) @binding(1) var filtering: sampler;
@group(0) @binding(2) var trans: texture_2d<f32>;
@group(0) @binding(3) var multiple: texture_2d<f32>;
@group(0) @binding(4) var<storage,read> directions: array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> result: array<vec4<f32>>;
@group(0) @binding(6) var weather: texture_cube<f32>;
{helpers}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {{
 let d=directions[id.x];
 let t={call};
 result[id.x*2u]=vec4(t.radiance,1.0); result[id.x*2u+1u]=vec4(t.transmittance,1.0);
}}
"#
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(directions),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (directions.len() * 32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let weather_view = self
            .device
            .create_texture_with_data(
                &self.queue,
                &wgpu::TextureDescriptor {
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
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &weather_texel.repeat(6),
            )
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.filtering),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&self.views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&self.views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output.as_entire_binding(),
            },
        ];
        // The derived layout only contains bindings the program uses.
        if call.contains("weather") {
            entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&weather_view),
            });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(directions.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        self.queue.submit([encoder.finish()]);
        mapped(&self.device, &readback)
    }
}

#[test]
#[ignore = "Vulkan GPU: run separately from builds and captures"]
#[expect(
    clippy::print_stdout,
    reason = "Reports numerical evidence during explicit GPU validation"
)]
fn orbital_shell_matches_independent_reference_and_depth_limits() {
    futures_lite::future::block_on(async {
        let gpu = Luts::new().await;
        let mut state = AtmosphereState {
            sun_direction: Vec3::Z,
            moon_illuminance: 0.0,
            stars: 0.0,
            ..Default::default()
        };
        state.medium.multiple_scattering = false;
        let mut params = parameters(&state);
        params[5] = [0.0, 0.0, 24000.0, 6360.0];
        params[6] = [6460.0, 0.0, 0.0, 137.0];
        // Cloud shell 2-8 km, built-in noise cover.
        params[8] = [2.0, 8.0, 0.0, 0.0];
        gpu.generate_parameters(&params);
        let mut directions = Vec::new();
        for azimuth in [
            0.0_f32,
            core::f32::consts::FRAC_PI_2,
            core::f32::consts::PI,
            core::f32::consts::TAU - 0.00001,
        ] {
            for impact in [
                0.0_f32, 5000.0, 6359.0, 6361.0, 6370.0, 6390.0, 6440.0, 6480.0,
            ] {
                let angle = (impact / 24000.0).asin();
                directions.push([
                    angle.sin() * azimuth.cos(),
                    angle.sin() * azimuth.sin(),
                    -angle.cos(),
                    1e6,
                ]);
            }
        }
        // A foreground object at 1 metre must remain outside the atmosphere.
        directions.push([0.0, 0.0, -1.0, 0.001]);
        let actual = gpu.probe(&params, &directions);
        let mut max_error = 0.0_f64;
        let mut max_trans = 0.0_f64;
        let mut reference_image = String::from("P3\n8 4\n255\n");
        let mut actual_image = reference_image.clone();
        for (i, d) in directions.iter().enumerate() {
            let dir = DVec3::new(f64::from(d[0]), f64::from(d[1]), f64::from(d[2])).normalize();
            let (l, t) = orbit_reference(
                &state,
                DVec3::new(0.0, 0.0, 24e6),
                dir,
                f64::from(d[3]) * 1000.0,
            );
            let a = DVec3::from_array([
                f64::from(actual[i * 2][0]),
                f64::from(actual[i * 2][1]),
                f64::from(actual[i * 2][2]),
            ]);
            let b = DVec3::from_array([
                f64::from(actual[i * 2 + 1][0]),
                f64::from(actual[i * 2 + 1][1]),
                f64::from(actual[i * 2 + 1][2]),
            ]);
            let error = ((a - l).abs() / l.max(DVec3::splat(0.1))).max_element();
            let trans = (b - t).abs().max_element();
            max_error = max_error.max(error);
            max_trans = max_trans.max(trans);
            assert!(
                a.is_finite() && b.is_finite() && error <= 0.10 && trans <= 0.01,
                "ray{i}: relative={error}, trans={trans}, actual{a} reference{l}"
            );
            if i < 32 {
                for (image, color) in [(&mut reference_image, l), (&mut actual_image, a)] {
                    let rgb = (color / 10000.0).clamp(DVec3::ZERO, DVec3::ONE) * 255.0;
                    image.push_str(&format!(
                        "{} {} {}\n",
                        rgb.x as u32, rgb.y as u32, rgb.z as u32
                    ));
                }
            }
        }
        std::fs::create_dir_all("target/planetary-reference").unwrap();
        std::fs::write("target/planetary-reference/reference.ppm", reference_image).unwrap();
        std::fs::write("target/planetary-reference/gpu.ppm", actual_image).unwrap();
        println!(
            "orbital maximum relative radiance={max_error}, transmittance absolute={max_trans}; 32-ray azimuth/limb reference images + foreground depth"
        );
        // With an opaque source occluder, the central clear-air column receives no source.
        params[7] = [0.0, 0.0, 300000.0, 1737.4];
        let eclipsed = gpu.probe(&params, &[[0.0, 0.0, -1.0, 1e6]]);
        assert!(eclipsed[0][..3].iter().all(|v| v.abs() < 1e-5));
        params[7] = [0.0; 4];
        // Segmenting the ray at the cloud shell must not change clear air:
        // a vanishing cloud extinction has to reproduce the same reference.
        params[6][1] = 0.55;
        params[6][2] = 1e-9;
        let segmented = gpu.probe_view(&params, &directions);
        let mut segmented_error = 0.0_f64;
        let mut segmented_trans = 0.0_f64;
        for (i, d) in directions.iter().enumerate() {
            let dir = DVec3::new(f64::from(d[0]), f64::from(d[1]), f64::from(d[2])).normalize();
            let (l, t) = orbit_reference(
                &state,
                DVec3::new(0.0, 0.0, 24e6),
                dir,
                f64::from(d[3]) * 1000.0,
            );
            for channel in 0..3 {
                let a = f64::from(segmented[i * 2][channel]);
                let b = f64::from(segmented[i * 2 + 1][channel]);
                segmented_error = segmented_error.max((a - l[channel]).abs() / l[channel].max(0.1));
                segmented_trans = segmented_trans.max((b - t[channel]).abs());
            }
        }
        println!(
            "segmented clear-air maximum relative radiance={segmented_error}, transmittance absolute={segmented_trans}"
        );
        assert!(
            segmented_error <= 0.10 && segmented_trans <= 0.01,
            "cloud-shell segmentation must preserve clear-air transport"
        );
        params[6][2] = 0.5;
        let cloudy = gpu.probe_view(&params, &directions);
        let repeated = gpu.probe_view(&params, &directions);
        assert_eq!(
            cloudy, repeated,
            "body-fixed cloud field must repeat exactly"
        );
        params[4][3] = 256.0;
        let dense = gpu.probe_view(&params, &directions);
        let mut cloud_error = 0.0_f64;
        for (i, (a, b)) in cloudy.iter().zip(&dense).enumerate() {
            for channel in 0..3 {
                let error = if i % 2 == 0 {
                    (f64::from(a[channel] - b[channel])).abs() / f64::from(b[channel]).max(1.0)
                } else {
                    f64::from(a[channel] - b[channel]).abs()
                };
                cloud_error = cloud_error.max(error);
            }
        }
        println!(
            "cloud 128/512 integration maximum radiance-relative or transmittance-absolute difference={cloud_error}"
        );
        assert!(cloud_error < 0.05, "cloud-shell march must converge");
        assert!(cloudy.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
        assert!(
            cloudy.iter().zip(&actual).any(|(a, b)| a != b),
            "clouds must alter transport"
        );
        // Weather-map layer: an empty map is clear air, a full map converges.
        params[4][3] = 64.0;
        params[8][2] = 1.0;
        let empty = gpu.probe_program("", VIEW_PROBE, [0; 4], &params, &directions);
        params[6][1] = 0.0;
        let clear = gpu.probe_view(&params, &directions);
        params[6][1] = 0.55;
        let full = gpu.probe_program("", VIEW_PROBE, [255; 4], &params, &directions);
        params[4][3] = 256.0;
        let full_dense = gpu.probe_program("", VIEW_PROBE, [255; 4], &params, &directions);
        let mut empty_error = 0.0_f64;
        let mut map_error = 0.0_f64;
        for i in 0..full.len() {
            for channel in 0..3 {
                let scale = |value: f32| {
                    if i % 2 == 0 {
                        f64::from(value).max(1.0)
                    } else {
                        1.0
                    }
                };
                empty_error = empty_error.max(
                    f64::from(empty[i][channel] - clear[i][channel]).abs()
                        / scale(clear[i][channel]),
                );
                map_error = map_error.max(
                    f64::from(full[i][channel] - full_dense[i][channel]).abs()
                        / scale(full_dense[i][channel]),
                );
            }
        }
        println!(
            "weather map: empty versus clear maximum difference={empty_error}; full 128/512 maximum difference={map_error}"
        );
        assert!(
            empty_error < 0.01,
            "an empty weather map must match clear air"
        );
        assert!(map_error < 0.05, "weather-map cloud march must converge");
        assert!(full.iter().zip(&empty).any(|(a, b)| a != b));
    });
}

// Jendersie & d'Eon (2023) Eq. 4-7 at the droplet diameter the shader declares.
const DROPLET_DIAMETER_UM: f64 = 20.0;

fn droplet_fit() -> (f64, f64, f64, f64) {
    let d = DROPLET_DIAMETER_UM;
    let g = (-2.20679 / (d + 3.91029) - 0.428934).exp();
    let alpha = (3.62489 - 8.29288 / (d + 5.52825)).exp();
    let weight = (-0.599085 / (d - 0.641583) - 0.665888).exp();
    let mean =
        g * (1.0 + alpha * (3.0 + 2.0 * g * g) / 5.0) / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0);
    (weight, g, alpha, mean)
}

fn shader_constant(name: &str) -> f64 {
    let source = include_str!("../src/atmosphere/model.wgsl");
    let declaration = format!("const {name}: f32 = ");
    let start = source.find(&declaration).expect("declared constant") + declaration.len();
    let end = start + source[start..].find(';').expect("constant terminator");
    source[start..end].parse().expect("numeric constant")
}

#[test]
fn cloud_phase_constants_match_the_published_fit() {
    let (weight, g, alpha, mean) = droplet_fit();
    for (name, expected) in [
        ("CLOUD_DRAINE_WEIGHT", weight),
        ("CLOUD_DRAINE_G", g),
        ("CLOUD_DRAINE_ALPHA", alpha),
        ("CLOUD_DRAINE_MEAN_COSINE", mean),
    ] {
        let actual = shader_constant(name);
        assert!(
            (actual - expected).abs() <= expected * 1e-5,
            "{name}: shader {actual}, fit {expected}"
        );
    }
}

fn draine(mu: f64) -> f64 {
    let (_, g, alpha, _) = droplet_fit();
    (1.0 - g * g) * (1.0 + alpha * mu * mu)
        / (4.0
            * core::f64::consts::PI
            * (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0)
            * (1.0 + g * g - 2.0 * g * mu).powf(1.5))
}

/// Deterministic uniform variates in [0, 1).
struct Variates(u64);

impl Variates {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Monte Carlo reflection of a conservative plane-parallel slab with the
/// Draine phase function over a black boundary, for unit beam-normal
/// irradiance. Returns the flux albedo and the radiance leaving the top
/// toward a detector at `mu_view` on the sun's side of the solar plane.
fn slab_reference(depth: f64, mu_sun: f64, mu_view: f64, photons: u32) -> (f64, f64) {
    const TABLE: usize = 8192;
    let mut cdf = vec![0.0; TABLE + 1];
    for i in 0..TABLE {
        let mu = -1.0 + 2.0 * (i as f64 + 0.5) / TABLE as f64;
        cdf[i + 1] = cdf[i] + draine(mu);
    }
    let total = cdf[TABLE];
    let detector = [-(1.0 - mu_view * mu_view).sqrt(), 0.0, -mu_view];
    let mut variates = Variates(0x9E37_79B9_7F4A_7C15);
    let (mut reflected, mut radiance) = (0u32, 0.0);
    for _ in 0..photons {
        let mut direction = [(1.0 - mu_sun * mu_sun).sqrt(), 0.0, mu_sun];
        let mut tau = 0.0;
        loop {
            tau += -(1.0 - variates.next()).ln() * direction[2];
            if tau < 0.0 {
                reflected += 1;
                break;
            }
            if tau > depth {
                break;
            }
            let cosine = direction[0] * detector[0] + direction[2] * detector[2];
            radiance += draine(cosine) * (-tau / mu_view).exp() / mu_view;
            let target = variates.next() * total;
            let bin = cdf.partition_point(|v| *v < target).clamp(1, TABLE);
            let fraction = (target - cdf[bin - 1]) / (cdf[bin] - cdf[bin - 1]);
            let mu = -1.0 + 2.0 * (bin as f64 - 1.0 + fraction) / TABLE as f64;
            let sine = (1.0 - mu * mu).max(0.0).sqrt();
            let phi = variates.next() * core::f64::consts::TAU;
            let [x, y, z] = direction;
            let planar = (1.0 - z * z).max(0.0).sqrt();
            direction = if planar < 1e-6 {
                [sine * phi.cos(), sine * phi.sin(), mu * z.signum()]
            } else {
                [
                    sine * (x * z * phi.cos() - y * phi.sin()) / planar + x * mu,
                    sine * (y * z * phi.cos() + x * phi.sin()) / planar + y * mu,
                    -sine * phi.cos() * planar + z * mu,
                ]
            };
        }
    }
    (
        f64::from(reflected) / f64::from(photons),
        radiance / f64::from(photons) * mu_sun,
    )
}

/// Integrates the production cloud source functions through a homogeneous
/// slab. Inputs are scaled optical depth, solar cosine and view cosine. The transmittance
/// slot carries the diffuse source at the slab top for light leaving upward.
const SLAB_PROBE: &str = r#"
fn slab_probe(d: vec4<f32>) -> AtmosphereTransport {
    // Keeps every binding of the shared probe layout in use.
    let keep = (textureSampleLevel(trans, filtering, vec2(0.5), 0.0).r
        + textureSampleLevel(multiple, filtering, vec2(0.5), 0.0).r + p.sun.w)*0.0;
    let steps = 4096u;
    let dt = d.x/f32(steps);
    let cosine = -d.y*d.z-sqrt(max(0.0, 1.0-d.y*d.y))*sqrt(max(0.0, 1.0-d.z*d.z));
    var radiance = 0.0;
    for (var i = 0u; i < steps; i++) {
        let t = (f32(i)+0.5)*dt;
        let source = phase_draine(cosine)*exp(-t/d.y)+cloud_diffuse(t, d.x, d.y, -d.z, 0.0);
        radiance += source*exp(-t/d.z)/d.z*dt;
    }
    return AtmosphereTransport(vec3(radiance+keep), vec3(cloud_diffuse(0.0, d.x, d.y, -1.0, 0.0)));
}
"#;

#[test]
#[ignore = "Vulkan GPU: run separately from builds and captures"]
#[expect(
    clippy::print_stdout,
    reason = "Reports numerical evidence during explicit GPU validation"
)]
fn cloud_slab_reflection_matches_monte_carlo() {
    futures_lite::future::block_on(async {
        let gpu = Luts::new().await;
        let params = parameters(&AtmosphereState::default());
        gpu.generate_parameters(&params);
        let cases: Vec<[f32; 4]> = [5.0_f32, 15.0, 40.0]
            .into_iter()
            .flat_map(|depth| [1.0_f32, 0.5].map(|mu_sun| [depth, mu_sun, 1.0, 0.0]))
            .collect();
        let actual = gpu.probe_program(SLAB_PROBE, "slab_probe(d)", [0; 4], &params, &cases);
        let (_, _, _, mean) = droplet_fit();
        for (i, case) in cases.iter().enumerate() {
            let (depth, mu_sun) = (f64::from(case[0]), f64::from(case[1]));
            let (albedo, radiance) = slab_reference(depth, mu_sun, 1.0, 40_000);
            let model_radiance = f64::from(actual[i * 2][0]);
            // At the top the downward diffuse flux vanishes, so I1 = -1.5 I0 and
            // the upward flux is 2 pi I0.
            let top = f64::from(actual[i * 2 + 1][0]) / (1.0 + 1.5 * mean);
            let model_albedo = core::f64::consts::TAU * top / mu_sun;
            let ratio = model_radiance / radiance;
            println!(
                "slab depth={depth} mu_sun={mu_sun}: albedo model={model_albedo:.4} reference={albedo:.4}; nadir radiance ratio={ratio:.3}"
            );
            assert!(
                (model_albedo - albedo).abs() <= 0.03,
                "slab albedo outside 0.03 of Monte Carlo"
            );
            let tolerance = if mu_sun > 0.9 { 0.05 } else { 0.25 };
            assert!(
                (ratio - 1.0).abs() <= tolerance,
                "slab nadir radiance outside {tolerance} of Monte Carlo"
            );
        }
    });
}
