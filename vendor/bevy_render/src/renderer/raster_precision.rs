use super::RenderAdapter;

impl RenderAdapter {
    /// Fractional framebuffer-coordinate bits retained by the rasterizer.
    /// Returns `None` when the backend does not expose this property here.
    pub fn raster_subpixel_precision_bits(&self) -> Option<u32> {
        #[cfg(any(
            target_os = "windows",
            target_os = "linux",
            target_os = "android",
            feature = "vulkan-portability"
        ))]
        {
            // SAFETY: The borrowed HAL adapter keeps its instance and physical
            // device alive. This immutable property query changes no GPU state
            // and neither destroys nor retains any raw handle.
            unsafe {
                if let Some(adapter) = self.as_hal::<wgpu::hal::api::Vulkan>() {
                    let properties = adapter
                        .shared_instance()
                        .raw_instance()
                        .get_physical_device_properties(adapter.raw_physical_device());
                    return Some(properties.limits.sub_pixel_precision_bits);
                }
            }
        }
        None
    }
}
