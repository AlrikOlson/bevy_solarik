//! Ownership needed to update exact scene rows without walking unrelated roots.
use super::binder::RayInstanceInput;
use bevy_ecs::entity::Entity;
use bevy_platform::collections::HashMap;
use bevy_render::scene_slots::SceneInstance;

#[derive(Default)]
pub(super) struct InputRoots {
    pub(super) initialized: bool,
    parts: HashMap<Entity, Vec<SceneInstance>>,
}
impl InputRoots {
    pub(super) fn clear(&mut self) {
        self.parts.clear();
        self.initialized = true;
    }
    pub(super) fn observe(&mut self, input: &RayInstanceInput) {
        self.parts
            .entry(input.entity.entity)
            .or_default()
            .push(input.entity);
    }
    pub(super) fn len(&self) -> usize {
        self.parts.len()
    }
    pub(super) fn replace(
        &mut self,
        entity: Entity,
        rows: &[RayInstanceInput],
    ) -> Vec<SceneInstance> {
        let next: Vec<_> = rows.iter().map(|row| row.entity).collect();
        let old = if next.is_empty() {
            self.parts.remove(&entity)
        } else {
            self.parts.insert(entity, next.clone())
        };
        old.into_iter()
            .flatten()
            .filter(|part| !next.contains(part))
            .collect()
    }
}
