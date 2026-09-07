use super::SolarikLighting;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_anti_alias::dlss::{
    Dlss, DlssRayReconstructionFeature, ViewDlssRayReconstructionTextures,
};
use bevy_camera::MainPassResolutionOverride;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_ecs::query::Has;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::texture::CachedTexture;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureView, TextureViewDescriptor,
    },
    renderer::RenderDevice,
};

/// Size of the `LightSample` shader struct in bytes.
const LIGHT_SAMPLE_STRUCT_SIZE: u64 = 8;

/// Size of the `ResolvedLightSamplePacked` shader struct in bytes.
const RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE: u64 = 36;

/// Size of the GI `Reservoir` shader struct in bytes.
const GI_RESERVOIR_STRUCT_SIZE: u64 = 48;

pub const LIGHT_TILE_BLOCKS: u64 = 128;
pub const LIGHT_TILE_SAMPLES_PER_BLOCK: u64 = 1024;

/// Amount of entries in the world cache (must be a power of 2, and >= 2^10)
pub const WORLD_CACHE_SIZE: u64 = 2u64.pow(20);

/// Internal rendering resources used for Solarik lighting.
#[derive(Component)]
pub struct SolarikLightingResources {
    pub sky_distribution: Buffer,
    pub light_tile_samples: Buffer,
    pub light_tile_resolved_samples: Buffer,
    pub di_reservoirs_a: TextureView,
    pub di_reservoirs_b: TextureView,
    pub gi_reservoirs_a: Buffer,
    pub gi_reservoirs_b: Buffer,
    pub world_cache_checksums: Buffer,
    pub world_cache_life: Buffer,
    pub world_cache_radiance: Buffer,
    pub world_cache_geometry_data: Buffer,
    pub world_cache_luminance_deltas: Buffer,
    pub world_cache_active_cells_new_radiance: Buffer,
    pub world_cache_a: Buffer,
    pub world_cache_b: Buffer,
    pub world_cache_active_cell_indices: Buffer,
    pub world_cache_active_cells_dispatch: Buffer,
    pub view_size: UVec2,
}

/// The denoiser guide set a host
/// denoiser reads beside the lit frame — written by the
/// `resolve_denoise_guides` pass (albedos, normal, roughness) and by
/// `specular_gi.wgsl` (the specular hit distance) when
/// [`SolarikLighting::denoise_guides`] is set. Every texture is
/// `Rgba16Float`: one core storage format for the five (roughness and
/// hit distance in `.r`), so the resolve pass needs no adapter-specific
/// single-channel storage format and the host sees one pixel format.
#[derive(Component)]
pub struct SolarikDenoiseGuideTextures {
    pub diffuse_albedo: CachedTexture,
    pub specular_albedo: CachedTexture,
    /// World-space normal in `.xyz`.
    pub normal: CachedTexture,
    /// Perceptual roughness in `.r`.
    pub roughness: CachedTexture,
    /// Distance from the primary surface to the specular path's first
    /// hit in `.r` (metres; `RAY_T_MAX` on a sky miss, 0 where nothing
    /// was traced).
    pub specular_hit_distance: CachedTexture,
}

/// The format every guide texture carries.
pub const DENOISE_GUIDE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub fn prepare_solari_lighting_resources(
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))] query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&SolarikLightingResources>,
            Option<&MainPassResolutionOverride>,
            &SolarikLighting,
        ),
        With<SolarikLighting>,
    >,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] query: Query<
        (
            Entity,
            &ExtractedCamera,
            Option<&SolarikLightingResources>,
            Option<&MainPassResolutionOverride>,
            &SolarikLighting,
            Has<Dlss<DlssRayReconstructionFeature>>,
        ),
        With<SolarikLighting>,
    >,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for query_item in &query {
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let (entity, camera, solarik_lighting_resources, resolution_override, solarik_lighting) =
            query_item;
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let (
            entity,
            camera,
            solarik_lighting_resources,
            resolution_override,
            solarik_lighting,
            has_dlss_rr,
        ) = query_item;

        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }

        if solarik_lighting_resources.map(|r| r.view_size) == Some(view_size) {
            continue;
        }

        let sky_distribution = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_sky_distribution"),
            size: (6 * 128 * 128 + 6 * 128) * size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let light_tile_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_light_tile_samples"),
            size: LIGHT_TILE_BLOCKS * LIGHT_TILE_SAMPLES_PER_BLOCK * LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let light_tile_resolved_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_light_tile_resolved_samples"),
            size: LIGHT_TILE_BLOCKS
                * LIGHT_TILE_SAMPLES_PER_BLOCK
                * RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let di_reservoirs = |name| {
            render_device
                .create_texture(&TextureDescriptor {
                    label: Some(name),
                    size: view_size.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba32Uint,
                    usage: TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        let di_reservoirs_a = di_reservoirs("solarik_lighting_di_reservoirs_a");
        let di_reservoirs_b = di_reservoirs("solarik_lighting_di_reservoirs_b");

        let gi_reservoirs = |name| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: (view_size.x * view_size.y) as u64 * GI_RESERVOIR_STRUCT_SIZE,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let gi_reservoirs_a = gi_reservoirs("solarik_lighting_gi_reservoirs_a");
        let gi_reservoirs_b = gi_reservoirs("solarik_lighting_gi_reservoirs_b");

        let world_cache_checksums = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_checksums"),
            size: WORLD_CACHE_SIZE * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let world_cache_life = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_life"),
            size: WORLD_CACHE_SIZE * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let world_cache_radiance = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_radiance"),
            size: WORLD_CACHE_SIZE * size_of::<[f32; 4]>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let world_cache_geometry_data = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_geometry_data"),
            size: WORLD_CACHE_SIZE * size_of::<[f32; 8]>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let world_cache_luminance_deltas = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_luminance_deltas"),
            size: WORLD_CACHE_SIZE * size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let world_cache_active_cells_new_radiance =
            render_device.create_buffer(&BufferDescriptor {
                label: Some("solarik_lighting_world_cache_active_cells_new_radiance"),
                size: WORLD_CACHE_SIZE * size_of::<[f32; 4]>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

        let world_cache_a = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_a"),
            size: WORLD_CACHE_SIZE * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let world_cache_b = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_b"),
            // Prefix scan slots 0..1024; final slot is the active count.
            size: 1025 * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let world_cache_active_cell_indices = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_active_cell_indices"),
            size: WORLD_CACHE_SIZE * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let world_cache_active_cells_dispatch = render_device.create_buffer(&BufferDescriptor {
            label: Some("solarik_lighting_world_cache_active_cells_dispatch"),
            // A storage vec3 occupies 16 bytes in Metal; indirect dispatch reads the first 12.
            size: size_of::<[u32; 4]>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        commands.entity(entity).insert(SolarikLightingResources {
            sky_distribution,
            light_tile_samples,
            light_tile_resolved_samples,
            di_reservoirs_a,
            di_reservoirs_b,
            gi_reservoirs_a,
            gi_reservoirs_b,
            world_cache_checksums,
            world_cache_life,
            world_cache_radiance,
            world_cache_geometry_data,
            world_cache_luminance_deltas,
            world_cache_active_cells_new_radiance,
            world_cache_a,
            world_cache_b,
            world_cache_active_cell_indices,
            world_cache_active_cells_dispatch,
            view_size,
        });

        if solarik_lighting.denoise_guides {
            let guide = |label: &'static str| {
                let texture = render_device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: view_size.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: DENOISE_GUIDE_FORMAT,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
                let default_view = texture.create_view(&TextureViewDescriptor::default());
                CachedTexture {
                    texture,
                    default_view,
                }
            };
            commands.entity(entity).insert(SolarikDenoiseGuideTextures {
                diffuse_albedo: guide("solarik_denoise_guide_diffuse_albedo"),
                specular_albedo: guide("solarik_denoise_guide_specular_albedo"),
                normal: guide("solarik_denoise_guide_normal"),
                roughness: guide("solarik_denoise_guide_roughness"),
                specular_hit_distance: guide("solarik_denoise_guide_specular_hit_distance"),
            });
        }

        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        if has_dlss_rr {
            let diffuse_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solarik_lighting_diffuse_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let diffuse_albedo_view = diffuse_albedo.create_view(&TextureViewDescriptor::default());

            let specular_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solarik_lighting_specular_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_albedo_view =
                specular_albedo.create_view(&TextureViewDescriptor::default());

            let normal_roughness = render_device.create_texture(&TextureDescriptor {
                label: Some("solarik_lighting_normal_roughness"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let normal_roughness_view =
                normal_roughness.create_view(&TextureViewDescriptor::default());

            let specular_motion_vectors = render_device.create_texture(&TextureDescriptor {
                label: Some("solarik_lighting_specular_motion_vectors"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_motion_vectors_view =
                specular_motion_vectors.create_view(&TextureViewDescriptor::default());

            commands
                .entity(entity)
                .insert(ViewDlssRayReconstructionTextures {
                    diffuse_albedo: CachedTexture {
                        texture: diffuse_albedo,
                        default_view: diffuse_albedo_view,
                    },
                    specular_albedo: CachedTexture {
                        texture: specular_albedo,
                        default_view: specular_albedo_view,
                    },
                    normal_roughness: CachedTexture {
                        texture: normal_roughness,
                        default_view: normal_roughness_view,
                    },
                    specular_motion_vectors: CachedTexture {
                        texture: specular_motion_vectors,
                        default_view: specular_motion_vectors_view,
                    },
                });
        }
    }
}
