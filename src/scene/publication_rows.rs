//! Retained shader rows keep matrix bytes until their authoritative input changes.
use super::{GpuInstanceGeometryIds, Mat4, RayInstanceInput};

type Buffers<'a> = (
    &'a mut [Mat4],
    &'a mut [Mat4],
    &'a mut [GpuInstanceGeometryIds],
    &'a mut [u32],
);

#[inline]
pub(super) fn write(
    index: usize,
    input: &RayInstanceInput,
    (geometry, material): (&GpuInstanceGeometryIds, u32),
    (transforms, previous, geometries, materials): Buffers<'_>,
    transform_changed: bool,
) -> bool {
    if transform_changed {
        transforms[index] = Mat4::from(input.transform);
        previous[index] = Mat4::from(input.previous);
    }
    let changed = geometries[index] != *geometry || materials[index] != material;
    if changed {
        geometries[index] = geometry.clone();
        materials[index] = material;
    }
    changed
}

#[cfg(test)]
#[path = "publication_rows_tests.rs"]
mod tests;
