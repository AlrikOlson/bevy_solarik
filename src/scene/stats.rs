//! Evidence published after a complete scene update, never halfway through it.
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

static SCENE: Mutex<Option<SceneStats>> = Mutex::new(None);
static REALTIME: AtomicU64 = AtomicU64::new(0);
static PRIMARY: AtomicU64 = AtomicU64::new(0);
static REFERENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneStats {
    pub requested: u64,
    pub instances: u64,
    pub missing_blas: u64,
    pub missing_geometry: u64,
    pub missing_material: u64,
    pub triangles: u64,
    pub colour_instances: u64,
    pub uv1_instances: u64,
    pub meshes: u64,
    pub pool_bytes: u64,
    pub lights: u64,
    pub realtime_views: u64,
    pub primary_views: u64,
    pub reference_views: u64,
}
pub fn scene_stats() -> SceneStats {
    let mut stats = SCENE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .unwrap_or_default();
    stats.realtime_views = REALTIME.load(Relaxed);
    stats.primary_views = PRIMARY.load(Relaxed);
    stats.reference_views = REFERENCE.load(Relaxed);
    stats
}

/// Publish on every return path. The main-world report may run concurrently
/// with scene preparation; it must see one finished census, not its reset.
pub(super) struct SceneUpdate(SceneStats);
impl core::ops::Deref for SceneUpdate {
    type Target = SceneStats;
    fn deref(&self) -> &SceneStats {
        &self.0
    }
}
impl core::ops::DerefMut for SceneUpdate {
    fn deref_mut(&mut self) -> &mut SceneStats {
        &mut self.0
    }
}
impl Drop for SceneUpdate {
    fn drop(&mut self) {
        *SCENE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(self.0);
    }
}
pub(super) fn begin_scene(requested: usize) -> SceneUpdate {
    SceneUpdate(SceneStats {
        requested: requested as u64,
        ..Default::default()
    })
}
pub(crate) fn record_dispatch(primary: bool, reference: bool) {
    let counter = if reference {
        &REFERENCE
    } else if primary {
        &PRIMARY
    } else {
        &REALTIME
    };
    counter.fetch_add(1, Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_only_complete_censuses_including_an_empty_scene() {
        let before = scene_stats();
        let mut update = begin_scene(3);
        update.instances = 2;
        update.triangles = 18;
        assert_eq!(
            scene_stats(),
            before,
            "preparing a frame cannot publish a partial census"
        );
        drop(update);
        assert_eq!(
            (
                scene_stats().requested,
                scene_stats().instances,
                scene_stats().triangles
            ),
            (3, 2, 18)
        );
        drop(begin_scene(0));
        assert_eq!(
            scene_stats().instances,
            0,
            "an empty scene must clear the last census"
        );
    }
}
