//! Explicit single-mip visibility alpha masks, sampled before camera/shadow depth writes.
use bevy_asset::{Assets, Handle};
use bevy_ecs::{component::Component, resource::Resource};
use bevy_image::Image;
use bevy_math::{UVec2, Vec4};
use bevy_render::{
    extract_resource::ExtractResource,
    render_resource::{Extent3d, TextureFormat, TextureViewDimension},
};

/// Shared, single-mip R8/RGBA texture array for opt-in meshlet visibility cutouts.
/// UV transforms must already be baked into the meshlet UVs. Raster materials
/// remain opaque; ray materials must independently retain the matching alpha mask.
#[derive(Clone, Default, Resource, ExtractResource)]
#[extract_app(bevy_render::RenderApp)]
pub struct MeshletCutoutAtlas(pub Handle<Image>);

/// Sample this atlas layer before writing visibility or shadow depth.
/// Invalid/unloaded atlases or invalid metadata suppress the instance.
#[derive(Clone, Copy, Component)]
pub struct MeshletVisibilityCutout {
    pub layer: u32,
    pub cutoff: f32,
    /// Original mask dimensions inside a padded layer. Zero keeps legacy full-layer UVs.
    pub size: UVec2,
}

/// Rasterize both windings of a meshlet surface. Its material must also enable
/// double-sided shading so back-face normals match the ordinary mesh path.
#[derive(Clone, Copy, Component)]
pub struct MeshletDoubleSided;

pub(super) fn metadata(
    cutout: Option<&MeshletVisibilityCutout>,
    atlas: &MeshletCutoutAtlas,
    images: &Assets<Image>,
) -> Option<Vec4> {
    let Some(cutout) = cutout else {
        return Some(Vec4::new(-1.0, 0.0, 0.0, 0.0));
    };
    let image = images.get(&atlas.0)?;
    let desc = &image.texture_descriptor;
    let view = image.texture_view_descriptor.as_ref()?;
    let valid = valid_atlas(desc.size, desc.format, desc.mip_level_count, view.dimension)
        && view.base_array_layer == 0
        && view.array_layer_count.is_none()
        && cutout.layer < desc.size.depth_or_array_layers
        && cutout.cutoff.is_finite()
        && (0.0..=1.0).contains(&cutout.cutoff);
    let full = UVec2::new(desc.size.width, desc.size.height);
    let size = if cutout.size == UVec2::ZERO {
        full
    } else {
        cutout.size
    };
    let valid = valid && size.min_element() > 0 && size.cmple(full).all();
    if !valid {
        return None;
    }
    // Both dimensions fit exactly in one f32 integer (24 bits). No instance
    // buffer/layout growth; z remains the double-sided flag. Negative selects R8.
    let code = if cutout.size == UVec2::ZERO && desc.format != TextureFormat::R8Unorm {
        0.0
    } else {
        (((size.x - 1) << 12) + size.y) as f32
    };
    let code = if desc.format == TextureFormat::R8Unorm {
        -code
    } else {
        code
    };
    valid.then_some(Vec4::new(cutout.layer as f32, cutout.cutoff, 0.0, code))
}

pub(super) fn valid_atlas(
    size: Extent3d,
    format: TextureFormat,
    mips: u32,
    view: Option<TextureViewDimension>,
) -> bool {
    mips == 1
        && size.width > 0
        && size.width <= 4096
        && size.height > 0
        && size.height <= 4096
        && size.depth_or_array_layers > 0
        && size.depth_or_array_layers <= 64
        && u64::from(size.width)
            * u64::from(size.height)
            * u64::from(size.depth_or_array_layers)
            * if format == TextureFormat::R8Unorm {
                1
            } else {
                4
            }
            <= 256 * 1024 * 1024
        && matches!(
            format,
            TextureFormat::R8Unorm | TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
        )
        && view == Some(TextureViewDimension::D2Array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_render::render_resource::TextureViewDescriptor;

    #[test]
    fn native_mask_dimensions_pack_exactly_and_r8_keeps_the_memory_budget() {
        let (mut images, atlas) = atlas();
        {
            let mut image = images.get_mut(&atlas.0).unwrap();
            image.texture_descriptor.size = Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 2,
            };
            image.texture_descriptor.format = TextureFormat::R8Unorm;
        }
        for size in [
            UVec2::ONE,
            UVec2::new(896, 896),
            UVec2::new(2048, 1024),
            UVec2::splat(4096),
        ] {
            let cutout = MeshletVisibilityCutout {
                layer: 1,
                cutoff: 0.5,
                size,
            };
            let value = metadata(Some(&cutout), &atlas, &images).unwrap();
            assert!(value.w < 0.0);
            let packed = value.w.abs() as u32 - 1;
            assert_eq!(UVec2::new((packed >> 12) + 1, (packed & 4095) + 1), size);
        }
        let invalid = MeshletVisibilityCutout {
            layer: 0,
            cutoff: 0.5,
            size: UVec2::new(0, 1),
        };
        assert!(metadata(Some(&invalid), &atlas, &images).is_none());
        let size = Extent3d {
            width: 2048,
            height: 2048,
            depth_or_array_layers: 64,
        };
        assert!(valid_atlas(
            size,
            TextureFormat::R8Unorm,
            1,
            Some(TextureViewDimension::D2Array)
        ));
        assert!(!valid_atlas(
            size,
            TextureFormat::Rgba8Unorm,
            1,
            Some(TextureViewDimension::D2Array)
        ));
    }

    fn atlas() -> (Assets<Image>, MeshletCutoutAtlas) {
        let mut images = Assets::<Image>::default();
        let mut image = Image::default();
        image.texture_descriptor.format = TextureFormat::Rgba8Unorm;
        image.texture_descriptor.size.depth_or_array_layers = 2;
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let handle = images.add(image);
        (images, MeshletCutoutAtlas(handle))
    }

    #[test]
    fn missing_atlas_suppresses_cutouts_but_preserves_opaque_instances() {
        let images = Assets::<Image>::default();
        let atlas = MeshletCutoutAtlas::default();
        let cutout = MeshletVisibilityCutout {
            size: UVec2::ZERO,
            layer: 0,
            cutoff: 0.5,
        };
        assert_eq!(metadata(None, &atlas, &images).unwrap().x, -1.0);
        assert!(metadata(Some(&cutout), &atlas, &images).is_none());
    }

    #[test]
    fn validates_layer_cutoff_and_single_mip_array_contract() {
        let (mut images, atlas) = atlas();
        let cutout = MeshletVisibilityCutout {
            size: UVec2::ZERO,
            layer: 1,
            cutoff: 0.5,
        };
        assert_eq!(
            metadata(Some(&cutout), &atlas, &images).unwrap(),
            Vec4::new(1.0, 0.5, 0.0, 0.0)
        );
        for invalid in [
            MeshletVisibilityCutout {
                size: UVec2::ZERO,
                layer: 2,
                cutoff: 0.5,
            },
            MeshletVisibilityCutout {
                size: UVec2::ZERO,
                layer: 0,
                cutoff: f32::NAN,
            },
            MeshletVisibilityCutout {
                size: UVec2::ZERO,
                layer: 0,
                cutoff: -0.1,
            },
            MeshletVisibilityCutout {
                size: UVec2::ZERO,
                layer: 0,
                cutoff: 1.1,
            },
        ] {
            assert!(metadata(Some(&invalid), &atlas, &images).is_none());
        }
        images
            .get_mut(&atlas.0)
            .unwrap()
            .texture_descriptor
            .mip_level_count = 2;
        assert!(metadata(Some(&cutout), &atlas, &images).is_none());
        images
            .get_mut(&atlas.0)
            .unwrap()
            .texture_descriptor
            .mip_level_count = 1;
        images.get_mut(&atlas.0).unwrap().texture_view_descriptor = None;
        assert!(metadata(Some(&cutout), &atlas, &images).is_none());
    }
}
