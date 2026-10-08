//! Exact counts over ready slots; no scene-size rescan after a root delta.
use alloc::collections::BTreeMap;
use bevy_asset::UntypedAssetId;
use bevy_platform::collections::HashMap;

#[derive(Default)]
pub(super) struct SceneMetadata {
    pub(super) materials: HashMap<UntypedAssetId, usize>,
    depths: BTreeMap<u32, usize>,
}
impl SceneMetadata {
    pub(super) fn add(&mut self, material: UntypedAssetId, depth: u32) {
        *self.materials.entry(material).or_default() += 1;
        *self.depths.entry(depth).or_default() += 1;
    }
    pub(super) fn remove(&mut self, material: UntypedAssetId, depth: u32) {
        let count = self
            .materials
            .get_mut(&material)
            .expect("active material count");
        *count -= 1;
        if *count == 0 {
            self.materials.remove(&material);
        }
        let count = self.depths.get_mut(&depth).expect("active depth count");
        *count -= 1;
        if *count == 0 {
            self.depths.remove(&depth);
        }
    }
    pub(super) fn max_depth(&self) -> u32 {
        self.depths.last_key_value().map_or(0, |(&depth, _)| depth)
    }
}
