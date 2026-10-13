use super::*;
use bevy_utils::default;

fn deferred(label: &'static str) -> DeferredPipeline {
    DeferredPipeline::new(ComputePipelineDescriptor {
        label: Some(label.into()),
        ..default()
    })
}

fn variants() -> ViewPipelineVariants {
    ViewPipelineVariants {
        standard: ViewPipelineSet {
            foliage: deferred("foliage"),
            glass: deferred("glass"),
            specular: deferred("specular"),
            guides: None,
        },
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        ray_reconstruction: ViewPipelineSet {
            foliage: deferred("rr foliage"),
            glass: deferred("rr glass"),
            specular: deferred("rr specular"),
            guides: Some(deferred("rr guides")),
        },
    }
}

fn record(
    labels: &mut Vec<String>,
    descriptor: ComputePipelineDescriptor,
) -> CachedComputePipelineId {
    labels.push(descriptor.label.unwrap().into_owned());
    CachedComputePipelineId::new(labels.len() - 1)
}

#[test]
fn unused_variants_are_not_queued() {
    let variants = variants();
    assert!(variants.standard.foliage.id.get().is_none());
    assert!(variants.standard.glass.id.get().is_none());
    assert!(variants.standard.specular.id.get().is_none());
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
    assert!(variants.ray_reconstruction.foliage.id.get().is_none());
}

#[test]
fn repeated_views_reuse_the_same_id_without_queuing_glass() {
    let variants = variants();
    let mut labels = Vec::new();
    let view = variants.for_view(ViewLightingMode::Standard);
    let first = view
        .foliage
        .queue_with(|descriptor| record(&mut labels, descriptor));
    let next = view.foliage.queue_with(|_| panic!("duplicate compilation"));
    assert_eq!(first, next);
    assert_eq!(labels, ["foliage"]);
    assert!(view.glass.id.get().is_none());
}

#[test]
fn glass_fallback_tracks_live_readiness_after_shader_reload() {
    let variants = variants();
    let view = variants.for_view(ViewLightingMode::Standard);
    let mut labels = Vec::new();
    assert!(!view.glass_ready_with(|descriptor| record(&mut labels, descriptor), |_| false));
    assert_eq!(labels, ["specular", "glass"]);
    assert!(view.glass_ready_with(|_| panic!("requeued"), |_| true));
    assert!(!view.glass_ready_with(|_| panic!("requeued"), |id| id.id() != 1));
    assert!(view.glass_ready_with(|_| panic!("requeued"), |_| true));
    assert!(view.foliage.id.get().is_none());
}

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
#[test]
fn rr_readiness_does_not_require_or_queue_standard_pipelines() {
    let variants = variants();
    let mut labels = Vec::new();
    let rr = variants.for_view(ViewLightingMode::RayReconstruction);
    assert!(rr.glass_ready_with(|descriptor| record(&mut labels, descriptor), |_| true));
    assert_eq!(labels, ["rr specular", "rr glass", "rr guides"]);
    assert!(variants.standard.glass.id.get().is_none());
    assert!(variants.standard.specular.id.get().is_none());
    assert!(variants.standard.foliage.id.get().is_none());
    assert!(!rr.glass_ready_with(|_| panic!("requeued"), |id| id.id() != 2));
}

#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
#[test]
fn switching_modes_and_mixed_views_retain_independent_pipeline_ids() {
    let variants = variants();
    let mut labels = Vec::new();
    let standard = variants.for_view(ViewLightingMode::Standard);
    let rr = variants.for_view(ViewLightingMode::RayReconstruction);
    let a = standard
        .foliage
        .queue_with(|descriptor| record(&mut labels, descriptor));
    let b = rr
        .foliage
        .queue_with(|descriptor| record(&mut labels, descriptor));
    assert_ne!(a, b);
    assert_eq!(standard.foliage.queue_with(|_| panic!("requeued")), a);
    assert_eq!(rr.foliage.queue_with(|_| panic!("requeued")), b);
    assert_eq!(labels, ["foliage", "rr foliage"]);
}
