//! A view requests its own lighting variants; unused variants never enter the queue.
use bevy_render::render_resource::{
    CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
};
use std::sync::OnceLock;

#[derive(Clone, Copy, Default)]
pub(super) enum ViewLightingMode {
    #[default]
    Standard,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    RayReconstruction,
}

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
impl ViewLightingMode {
    pub fn from_view(
        textures: Option<&bevy_anti_alias::dlss::ViewDlssRayReconstructionTextures>,
    ) -> Self {
        if textures.is_some() {
            Self::RayReconstruction
        } else {
            Self::Standard
        }
    }
}

pub(super) struct DeferredPipeline {
    descriptor: ComputePipelineDescriptor,
    id: OnceLock<CachedComputePipelineId>,
}

impl DeferredPipeline {
    pub fn new(descriptor: ComputePipelineDescriptor) -> Self {
        Self {
            descriptor,
            id: OnceLock::new(),
        }
    }

    pub fn id(&self, cache: &PipelineCache) -> CachedComputePipelineId {
        self.queue_with(|descriptor| cache.queue_compute_pipeline(descriptor))
    }

    fn queue_with(
        &self,
        queue: impl FnOnce(ComputePipelineDescriptor) -> CachedComputePipelineId,
    ) -> CachedComputePipelineId {
        // Keep the registered ID across shader reloads. Bevy owns its readiness.
        *self.id.get_or_init(|| queue(self.descriptor.clone()))
    }
}

pub(super) struct ViewPipelineSet {
    pub foliage: DeferredPipeline,
    pub glass: DeferredPipeline,
    pub specular: DeferredPipeline,
    pub guides: Option<DeferredPipeline>,
}

impl ViewPipelineSet {
    pub fn base_ids(
        &self,
        cache: &PipelineCache,
    ) -> (CachedComputePipelineId, Option<CachedComputePipelineId>) {
        (
            self.specular.id(cache),
            self.guides.as_ref().map(|pipeline| pipeline.id(cache)),
        )
    }

    pub fn glass_ready(&self, cache: &PipelineCache) -> bool {
        self.glass_ready_with(
            |descriptor| cache.queue_compute_pipeline(descriptor),
            |id| cache.get_compute_pipeline(id).is_some(),
        )
    }

    fn glass_ready_with(
        &self,
        mut queue: impl FnMut(ComputePipelineDescriptor) -> CachedComputePipelineId,
        mut ready: impl FnMut(CachedComputePipelineId) -> bool,
    ) -> bool {
        let ids = [
            Some(self.specular.queue_with(&mut queue)),
            Some(self.glass.queue_with(&mut queue)),
            self.guides
                .as_ref()
                .map(|pipeline| pipeline.queue_with(&mut queue)),
        ];
        ids.into_iter().flatten().all(&mut ready)
    }
}

pub(super) struct ViewPipelineVariants {
    pub standard: ViewPipelineSet,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    pub ray_reconstruction: ViewPipelineSet,
}

impl ViewPipelineVariants {
    pub fn for_view(&self, mode: ViewLightingMode) -> &ViewPipelineSet {
        match mode {
            ViewLightingMode::Standard => &self.standard,
            #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
            ViewLightingMode::RayReconstruction => &self.ray_reconstruction,
        }
    }
}

#[cfg(test)]
#[path = "view_pipelines.tests.rs"]
mod tests;
