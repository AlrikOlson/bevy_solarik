use super::{AtmosphereParams, AtmosphereState};
use bevy_app::{App, Plugin};
use bevy_asset::{
    AssetServer, Assets, Handle, RenderAssetUsages, embedded_asset, load_embedded_asset,
};
use bevy_camera::{CameraMainTextureUsages, Hdr};
use bevy_core_pipeline::{
    core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
    prepass::{DepthPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{prelude::*, schedule::SystemSet};
use bevy_image::Image;
use bevy_render::{
    Render, RenderApp, RenderStartup, RenderSystems,
    diagnostic::RecordDiagnostics as _,
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{
        AddressMode, BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
        FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
        StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
        UniformBuffer,
        binding_types::{
            sampler, texture_2d, texture_3d, texture_depth_2d, texture_storage_2d,
            texture_storage_2d_array, texture_storage_3d, uniform_buffer,
        },
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    texture::GpuImage,
    view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_shader::load_shader_library;

/// Enables the atmosphere sky and aerial perspective on a camera.
/// Requires an HDR, non-MSAA view with storage usage on its main texture.
#[derive(Component, ExtractComponent, Debug, Clone, Copy, Default)]
#[require(Hdr, DepthPrepass, Msaa::Off,
    CameraMainTextureUsages = CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING))]
pub struct AtmosphereCamera;

/// Render ordering boundary: after surface lighting, before temporal postprocessing.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtmosphereComposite;

/// Sky background before transparent and Solarik primary-glass compositing.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtmosphereBackground;

/// Disk-free radiance cubemap for `SolarikSkyLight` and `GeneratedEnvironmentMapLight`.
/// It is allocated once and written on the GPU; its image data is never uploaded per frame.
#[derive(Resource, ExtractResource, Debug, Clone)]
pub struct AtmosphereEnvironment {
    pub image: Handle<Image>,
}

impl AtmosphereEnvironment {
    pub fn new(images: &mut Assets<Image>) -> Self {
        let mut image = Image::new_uninit(
            Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC;
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });
        Self {
            image: images.add(image),
        }
    }
}

/// Optional clear-sky renderer, usable with Solarik or Bevy's raster lighting.
pub struct AtmospherePlugin;

impl Plugin for AtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "model.wgsl");
        embedded_asset!(app, "luts.wgsl");
        embedded_asset!(app, "view.wgsl");
        app.init_resource::<AtmosphereState>().add_plugins((
            ExtractResourcePlugin::<AtmosphereState>::default(),
            ExtractResourcePlugin::<AtmosphereEnvironment>::default(),
            ExtractComponentPlugin::<AtmosphereCamera>::default(),
        ));
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(RenderStartup, initialize_atmosphere)
                // Bevy filters generated environments after PrepareBindGroups.
                // Generate the source first, before either filter or Solarik's CDF.
                .add_systems(
                    Render,
                    generate_atmosphere.in_set(RenderSystems::PrepareBindGroups),
                )
                .configure_sets(
                    Core3d,
                    AtmosphereBackground
                        .after(main_opaque_pass_3d)
                        .before(main_transparent_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                )
                .add_systems(
                    Core3d,
                    compose_atmosphere::<true>.in_set(AtmosphereBackground),
                )
                .configure_sets(
                    Core3d,
                    AtmosphereComposite
                        .after(Core3dSystems::MainPass)
                        .before(Core3dSystems::EarlyPostProcess),
                )
                .add_systems(
                    Core3d,
                    compose_atmosphere::<false>.in_set(AtmosphereComposite),
                );
        }
    }
}

struct Stage {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
struct AtmosphereGpu {
    uniform: UniformBuffer<AtmosphereParams>,
    filtering: Sampler,
    trans: TextureView,
    multiple: TextureView,
    sky: TextureView,
    aerial_scattering: TextureView,
    aerial_transmittance: TextureView,
    stages: Vec<Stage>,
    last: Option<AtmosphereState>,
    last_cube: Option<TextureView>,
}

fn initialize_atmosphere(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    assets: Res<AssetServer>,
) {
    let read = || texture_2d(TextureSampleType::Float { filterable: true });
    let write = || texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly);
    let mut stages = Vec::new();
    for (index, definition) in [
        "TRANSMITTANCE",
        "MULTIPLE",
        "SKY_VIEW",
        "CUBE",
        "AERIAL",
        "COMPOSITE",
        "BACKGROUND",
    ]
    .into_iter()
    .enumerate()
    {
        let mut entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<AtmosphereParams>(false),
                sampler(SamplerBindingType::Filtering),
            ),
        )
        .to_vec();
        entries.extend(match index {
            0 => BindGroupLayoutEntries::with_indices(ShaderStages::COMPUTE, ((2, write()),))
                .to_vec(),
            1 => BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                ((2, read()), (3, write())),
            )
            .to_vec(),
            2 => BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                ((2, read()), (3, read()), (4, write())),
            )
            .to_vec(),
            3 => BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (2, read()),
                    (
                        3,
                        texture_storage_2d_array(
                            TextureFormat::Rgba32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            )
            .to_vec(),
            4 => BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (2, read()),
                    (3, read()),
                    (4, uniform_buffer::<ViewUniform>(true)),
                    (
                        5,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        6,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            )
            .to_vec(),
            _ => BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (2, read()),
                    (3, read()),
                    (4, uniform_buffer::<ViewUniform>(true)),
                    (5, read()),
                    (6, texture_3d(TextureSampleType::Float { filterable: true })),
                    (7, texture_3d(TextureSampleType::Float { filterable: true })),
                    (8, texture_depth_2d()),
                    (
                        9,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                    ),
                ),
            )
            .to_vec(),
        });
        let layout = BindGroupLayoutDescriptor::new(definition, &entries);
        let shader = if index < 4 {
            load_embedded_asset!(assets.as_ref(), "luts.wgsl")
        } else {
            load_embedded_asset!(assets.as_ref(), "view.wgsl")
        };
        let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("atmosphere_{definition}").into()),
            layout: vec![layout.clone()],
            shader,
            shader_defs: if index == 6 {
                vec!["COMPOSITE".into(), "BACKGROUND".into()]
            } else {
                vec![definition.into()]
            },
            ..Default::default()
        });
        stages.push(Stage { layout, pipeline });
    }
    let texture = |label: &'static str, width, height, depth, format| {
        device
            .create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: depth,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: if depth > 1 {
                    TextureDimension::D3
                } else {
                    TextureDimension::D2
                },
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    };
    commands.insert_resource(AtmosphereGpu {
        uniform: UniformBuffer::from(AtmosphereState::default().uniform()),
        filtering: device.create_sampler(&SamplerDescriptor {
            label: Some("atmosphere_sampler"),
            address_mode_u: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        }),
        trans: texture(
            "atmosphere_transmittance",
            256,
            64,
            1,
            TextureFormat::Rgba32Float,
        ),
        multiple: texture(
            "atmosphere_multiple_scattering",
            32,
            32,
            1,
            TextureFormat::Rgba32Float,
        ),
        sky: texture(
            "atmosphere_sky_view",
            512,
            256,
            1,
            TextureFormat::Rgba32Float,
        ),
        aerial_scattering: texture(
            "atmosphere_aerial_scattering",
            32,
            32,
            32,
            TextureFormat::Rgba16Float,
        ),
        aerial_transmittance: texture(
            "atmosphere_aerial_transmittance",
            32,
            32,
            32,
            TextureFormat::Rgba16Float,
        ),
        stages,
        last: None,
        last_cube: None,
    });
}

fn make_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    gpu: &AtmosphereGpu,
    stage: usize,
    views: &[&TextureView],
) -> Option<BindGroup> {
    let mut entries = vec![
        BindGroupEntry {
            binding: 0,
            resource: gpu.uniform.binding()?,
        },
        BindGroupEntry {
            binding: 1,
            resource: bevy_render::render_resource::BindingResource::Sampler(&gpu.filtering),
        },
    ];
    entries.extend(views.iter().enumerate().map(|(i, view)| BindGroupEntry {
        binding: i as u32 + 2,
        resource: bevy_render::render_resource::BindingResource::TextureView(view),
    }));
    Some(device.create_bind_group(
        "atmosphere_lut_group",
        &cache.get_bind_group_layout(&gpu.stages[stage].layout),
        &entries,
    ))
}

fn generate_atmosphere(
    state: Res<AtmosphereState>,
    environment: Option<Res<AtmosphereEnvironment>>,
    images: Res<RenderAssets<GpuImage>>,
    gpu: Option<ResMut<AtmosphereGpu>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut ctx: RenderContext,
) {
    let (Some(environment), Some(mut gpu)) = (environment, gpu) else {
        return;
    };
    let Some(image) = images.get(&environment.image) else {
        return;
    };
    let image_changed = gpu
        .last_cube
        .as_ref()
        .is_none_or(|view| view.id() != image.texture_view.id());
    if gpu.last.as_ref() == Some(state.as_ref()) && !image_changed {
        return;
    }
    if let Err(error) = state.validate() {
        tracing::error!("invalid atmosphere state: {error}");
        return;
    }
    if gpu
        .stages
        .iter()
        .any(|stage| cache.get_compute_pipeline(stage.pipeline).is_none())
    {
        return;
    }
    let medium_changed = gpu
        .last
        .as_ref()
        .is_none_or(|last| last.medium != state.medium);
    *gpu.uniform.get_mut() = state.uniform();
    gpu.uniform.write_buffer(&device, &queue);
    let cube = image.texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });
    let views = [
        &[&gpu.trans][..],
        &[&gpu.trans, &gpu.multiple],
        &[&gpu.trans, &gpu.multiple, &gpu.sky],
        &[&gpu.sky, &cube],
    ];
    let sizes = [(32, 8, 1), (4, 4, 1), (64, 32, 1), (32, 32, 6)];
    let names = [
        "atmosphere/transmittance",
        "atmosphere/multiple_scattering",
        "atmosphere/sky_view",
        "atmosphere/cubemap",
    ];
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    for index in 0..4 {
        if index < 2 && !medium_changed {
            continue;
        }
        let Some(group) = make_group(&device, &cache, &gpu, index, views[index]) else {
            return;
        };
        let Some(pipeline) = cache.get_compute_pipeline(gpu.stages[index].pipeline) else {
            return;
        };
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(names[index]),
                timestamp_writes: None,
            });
        let span = diagnostics.time_span(&mut pass, names[index]);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(sizes[index].0, sizes[index].1, sizes[index].2);
        span.end(&mut pass);
    }
    gpu.last = Some(state.clone());
    gpu.last_cube = Some(image.texture_view.clone());
}

fn compose_atmosphere<const BACKGROUND: bool>(
    view: ViewQuery<(
        &AtmosphereCamera,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewUniformOffset,
    )>,
    gpu: Option<Res<AtmosphereGpu>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    uniforms: Res<ViewUniforms>,
    mut ctx: RenderContext,
) {
    let (_, target, prepass, offset) = view.into_inner();
    let Some(gpu) = gpu else {
        return;
    };
    if gpu.last.is_none() {
        return;
    }
    let (Some(depth), Some(view_binding), Some(parameters)) = (
        prepass.depth_view(),
        uniforms.uniforms.binding(),
        gpu.uniform.binding(),
    ) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let indices: &[usize] = if BACKGROUND { &[6] } else { &[4, 5] };
    for &index in indices {
        let Some(pipeline) = cache.get_compute_pipeline(gpu.stages[index].pipeline) else {
            return;
        };
        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: parameters.clone(),
            },
            BindGroupEntry {
                binding: 1,
                resource: bevy_render::render_resource::BindingResource::Sampler(&gpu.filtering),
            },
            BindGroupEntry {
                binding: 2,
                resource: bevy_render::render_resource::BindingResource::TextureView(&gpu.trans),
            },
            BindGroupEntry {
                binding: 3,
                resource: bevy_render::render_resource::BindingResource::TextureView(&gpu.multiple),
            },
            BindGroupEntry {
                binding: 4,
                resource: view_binding.clone(),
            },
        ];
        let texture_views = if index == 4 {
            vec![&gpu.aerial_scattering, &gpu.aerial_transmittance]
        } else {
            vec![
                &gpu.sky,
                &gpu.aerial_scattering,
                &gpu.aerial_transmittance,
                depth,
                target.main_texture_view(),
            ]
        };
        entries.extend(
            texture_views
                .into_iter()
                .enumerate()
                .map(|(i, texture)| BindGroupEntry {
                    binding: i as u32 + 5,
                    resource: bevy_render::render_resource::BindingResource::TextureView(texture),
                }),
        );
        let group = device.create_bind_group(
            "atmosphere_view_group",
            &cache.get_bind_group_layout(&gpu.stages[index].layout),
            &entries,
        );
        let name = match index {
            4 => "atmosphere/aerial_lut",
            6 => "atmosphere/background",
            _ => "atmosphere/composite",
        };
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
        let span = diagnostics.time_span(&mut pass, name);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[offset.offset]);
        if index == 4 {
            pass.dispatch_workgroups(8, 8, 8);
        } else {
            let size = target.main_texture().size();
            pass.dispatch_workgroups(size.width.div_ceil(8), size.height.div_ceil(8), 1);
        }
        span.end(&mut pass);
    }
}
