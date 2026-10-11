//! Renderer asset ownership for immutable finite canopy regions.
use super::{
    PreparedRegion, SourceProfile,
    arena::{Allocation, Arena},
};
use alloc::sync::Arc;
use bevy_asset::{Asset, AssetId, Handle};
use bevy_ecs::{
    component::Component,
    system::{
        SystemParamItem,
        lifetimeless::{SRes, SResMut},
    },
};
use bevy_math::{Affine3A, Vec3};
use bevy_reflect::TypePath;
use bevy_render::{
    render_asset::{PrepareAssetError, RenderAsset},
    renderer::{RenderDevice, RenderQueue},
    scene_readiness::SceneGeometryReadiness,
    sync_world::SyncToRenderWorld,
};
use bevy_transform::components::Transform;
#[derive(Asset, TypePath, Clone)]
pub struct CanopyRegion {
    pub(super) prepared: Arc<PreparedRegion>,
    pub(super) source_identity: [u8; 32],
}
impl CanopyRegion {
    /// Immutable source recovery for bounded graphics-debug preparation controls.
    #[cfg(feature = "graphics_debug")]
    pub fn prepared(&self) -> &Arc<PreparedRegion> {
        &self.prepared
    }
    pub fn new(
        prepared: Arc<PreparedRegion>,
        profile: &SourceProfile,
    ) -> Result<Self, &'static str> {
        if prepared.source_extents != profile.extents {
            return Err("canopy region/source bounds mismatch");
        }
        Ok(Self {
            prepared,
            source_identity: profile.identity,
        })
    }
}
#[derive(Component, Clone)]
#[require(Transform, SyncToRenderWorld)]
pub struct Canopy3d(pub Handle<CanopyRegion>);
/// Optional diagnostic mask for secondary rays. Primary depth/materials stay visible.
#[derive(Component, Clone, Copy)]
pub struct CanopyRayMask(pub bool);
pub(super) struct GpuRegion {
    pub allocation: Arc<Allocation>,
    pub source_identity: [u8; 32],
    pub minimum: Vec3,
    pub maximum: Vec3,
}
impl RenderAsset for GpuRegion {
    type SourceAsset = CanopyRegion;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<Arena>,
        SRes<SceneGeometryReadiness>,
    );
    fn byte_len(source: &Self::SourceAsset) -> Option<usize> {
        Some(source.prepared.bytes())
    }
    fn prepare_asset(
        source: Self::SourceAsset,
        id: AssetId<Self::SourceAsset>,
        (device, queue, arena, readiness): &mut SystemParamItem<Self::Param>,
        _previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        readiness.revoke(id.untyped());
        let Some(allocation) = arena.upload(source.prepared.rows(), device, queue) else {
            return Err(PrepareAssetError::RetryNextUpdate(source));
        };
        Ok(Self {
            allocation,
            source_identity: source.source_identity,
            minimum: source.prepared.minimum,
            maximum: source.prepared.maximum,
        })
    }
    fn unload_asset(
        id: AssetId<Self::SourceAsset>,
        (_, _, _, readiness): &mut SystemParamItem<Self::Param>,
    ) {
        readiness.revoke(id.untyped());
    }
}
#[derive(Clone, Copy, PartialEq)]
pub(super) struct Instance {
    pub id: AssetId<CanopyRegion>,
    pub transform: Affine3A,
    pub previous: Affine3A,
    pub identity: u64,
    pub rays: bool,
}
