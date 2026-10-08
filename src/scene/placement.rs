//! Experimental render-only placement formats; no scene consumer is enabled yet.
//! Scientific identity and exact body coordinates remain owned by the caller.

/// Plain WGSL functions, without bindings, for use by bounded scene experiments.
pub const SHADER: &str = include_str!("placement.wgsl");

/// Three scalar words, tightly packed with a 12-byte storage stride.
/// Never upload this as a WGSL array of vec3<u32>, which has a 16-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Placement12 {
    /// XY, Z/height/prototype, and smallest-three quaternion.
    pub words: [u32; 3],
}

/// Four scalar words with finer height/orientation and a larger prototype table.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Placement16 {
    /// XY, Z/height, two quaternion components, and component/prototype/index.
    pub words: [u32; 4],
}

/// Exact page origin and finite outward-rounded decode ranges.
/// This host structure is not copied verbatim to GPU storage.
#[derive(Clone, Copy, Debug)]
pub struct PlacementPage {
    /// Body-local minimum; camera subtraction must happen before conversion.
    pub origin: [f64; 3],
    /// Nonnegative page-local XYZ spans.
    pub extent: [f32; 3],
    /// Positive minimum decoded uniform scale.
    pub height_base: f32,
    /// Nonnegative uniform-scale span.
    pub height_span: f32,
}

/// Camera-relative upload layout, exactly two vec4<f32> values.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPlacementPage {
    /// Camera-relative page origin and minimum scale.
    pub origin_height: [f32; 4],
    /// Local XYZ spans and scale span.
    pub extent_height: [f32; 4],
}

impl PlacementPage {
    /// Reject invalid or overflowing input instead of publishing an unusable page.
    pub fn relative_to(self, camera: [f64; 3]) -> Option<GpuPlacementPage> {
        let relative: [f32; 3] = core::array::from_fn(|i| (self.origin[i] - camera[i]) as f32);
        let valid = self.origin.into_iter().chain(camera).all(f64::is_finite)
            && relative.into_iter().all(f32::is_finite)
            && self.extent.into_iter().all(|v| v.is_finite() && v >= 0.0)
            && self.height_base.is_finite()
            && self.height_base > 0.0
            && self.height_span.is_finite()
            && self.height_span >= 0.0
            && (self.height_base + self.height_span).is_finite()
            && relative
                .into_iter()
                .zip(self.extent)
                .all(|(a, b)| (a + b).is_finite());
        valid.then_some(GpuPlacementPage {
            origin_height: [relative[0], relative[1], relative[2], self.height_base],
            extent_height: [
                self.extent[0],
                self.extent[1],
                self.extent[2],
                self.height_span,
            ],
        })
    }
}
