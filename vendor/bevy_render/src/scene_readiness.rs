//! Request-driven prewarming receipt shared between app and render worlds.
//! Ready means queued on the render queue; subsequent rendering uses queue ordering.
use alloc::sync::Arc;
use bevy_asset::{Asset, AssetId, UntypedAssetId};
use bevy_ecs::resource::Resource;
use bevy_platform::collections::HashSet;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Default)]
struct State {
    requested: HashSet<UntypedAssetId>,
    ready: HashSet<UntypedAssetId>,
    alpha: HashSet<UntypedAssetId>,
}
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    revision: AtomicU64,
}
#[derive(Resource, Default, Clone)]
pub struct SceneGeometryReadiness(Arc<Shared>);
impl SceneGeometryReadiness {
    /// Invalidates cached immutable-product proofs on every readiness change.
    pub fn revision(&self) -> u64 {
        self.0.revision.load(Ordering::Acquire)
    }
    pub fn request(&self, id: UntypedAssetId) {
        if let Ok(mut state) = self.0.state.lock() {
            state.requested.insert(id);
        }
    }
    /// Prewarm source alpha geometry before its first scene instance exists.
    pub fn request_alpha(&self, id: UntypedAssetId) {
        if let Ok(mut state) = self.0.state.lock() {
            state.requested.insert(id);
            if state.alpha.insert(id) && state.ready.remove(&id) {
                self.0.revision.fetch_add(1, Ordering::Release);
            }
        }
    }
    pub fn alpha_requested<A: Asset>(&self) -> Vec<AssetId<A>> {
        self.0.state.lock().map_or_else(
            |_| Vec::new(),
            |state| {
                state
                    .alpha
                    .iter()
                    .filter(|id| id.type_id() == core::any::TypeId::of::<A>())
                    .map(|id| id.typed::<A>())
                    .collect()
            },
        )
    }
    pub fn ready(&self, id: UntypedAssetId) -> bool {
        self.0
            .state
            .lock()
            .is_ok_and(|state| state.ready.contains(&id))
    }
    pub fn release(&self, id: UntypedAssetId) {
        if let Ok(mut state) = self.0.state.lock() {
            state.requested.remove(&id);
            state.alpha.remove(&id);
            if state.ready.remove(&id) {
                self.0.revision.fetch_add(1, Ordering::Release);
            }
        }
    }
    pub fn revoke(&self, id: UntypedAssetId) {
        if let Ok(mut state) = self.0.state.lock() {
            if state.ready.remove(&id) {
                self.0.revision.fetch_add(1, Ordering::Release);
            }
        }
    }
    pub fn requested<A: Asset>(&self) -> Vec<AssetId<A>> {
        self.0.state.lock().map_or_else(
            |_| Vec::new(),
            |state| {
                state
                    .requested
                    .iter()
                    .filter(|id| id.type_id() == core::any::TypeId::of::<A>())
                    .map(|id| id.typed::<A>())
                    .collect()
            },
        )
    }
    pub fn publish(&self, id: UntypedAssetId, ready: bool) {
        if let Ok(mut state) = self.0.state.lock() {
            let changed = if ready && state.requested.contains(&id) {
                state.ready.insert(id)
            } else {
                state.ready.remove(&id)
            };
            if changed {
                self.0.revision.fetch_add(1, Ordering::Release);
            }
        }
    }
}
