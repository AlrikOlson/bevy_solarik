use super::MeshletPlugin;
use bevy_render::settings::WgpuLimits;

#[test]
fn queue_budget_respects_each_device_limit_and_packed_visibility_ids() {
    let baseline = WgpuLimits::default();
    assert_eq!(
        MeshletPlugin::validate_queue_capacity(1 << 24, &baseline),
        Ok(512 << 20)
    );
    assert!(MeshletPlugin::validate_queue_capacity(1 << 25, &baseline).is_err());
    let enthusiast = WgpuLimits {
        max_storage_buffer_binding_size: 256 << 20,
        max_buffer_size: 256 << 20,
        ..baseline
    };
    assert_eq!(
        MeshletPlugin::validate_queue_capacity(1 << 25, &enthusiast),
        Ok(1 << 30)
    );
    assert!(MeshletPlugin::validate_queue_capacity(0, &enthusiast).is_err());
    assert!(MeshletPlugin::validate_queue_capacity((1 << 25) + 1, &enthusiast).is_err());
    let small_buffer = WgpuLimits {
        max_buffer_size: 128 << 20,
        ..enthusiast.clone()
    };
    assert!(MeshletPlugin::validate_queue_capacity(1 << 25, &small_buffer).is_err());
    let small_binding = WgpuLimits {
        max_storage_buffer_binding_size: 128 << 20,
        ..enthusiast
    };
    assert!(MeshletPlugin::validate_queue_capacity(1 << 25, &small_binding).is_err());
}
