//! Explicit single-mip visibility alpha masks, sampled before camera/shadow depth writes.
use bevy_asset::{Assets, Handle};
use bevy_ecs::{component::Component, resource::Resource};
use bevy_image::Image;
use bevy_math::Vec4;
use bevy_render::{
    extract_resource::ExtractResource,
    render_resource::{Extent3d, TextureFormat, TextureViewDimension},
};

/// Shared, single-mip RGBA texture array for opt-in meshlet visibility cutouts.
/// UV transforms must already be baked into the meshlet UVs. Raster materials
/// remain opaque; ray materials must independently retain the matching alpha mask.
#[derive(Clone, Default, Resource, ExtractResource)]
pub struct MeshletCutoutAtlas(pub Handle<Image>);

/// Sample this atlas layer before writing visibility or shadow depth.
/// Invalid/unloaded atlases or invalid metadata suppress the instance.
#[derive(Clone, Copy, Component)]
pub struct MeshletVisibilityCutout {
    pub layer: u32,
    pub cutoff: f32,
}

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
    valid.then_some(Vec4::new(cutout.layer as f32, cutout.cutoff, 0.0, 0.0))
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
            * 4
            <= 256 * 1024 * 1024
        && matches!(
            format,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
        )
        && view == Some(TextureViewDimension::D2Array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_render::render_resource::TextureViewDescriptor;

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
            layer: 1,
            cutoff: 0.5,
        };
        assert_eq!(
            metadata(Some(&cutout), &atlas, &images).unwrap(),
            Vec4::new(1.0, 0.5, 0.0, 0.0)
        );
        for invalid in [
            MeshletVisibilityCutout {
                layer: 2,
                cutoff: 0.5,
            },
            MeshletVisibilityCutout {
                layer: 0,
                cutoff: f32::NAN,
            },
            MeshletVisibilityCutout {
                layer: 0,
                cutoff: -0.1,
            },
            MeshletVisibilityCutout {
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
