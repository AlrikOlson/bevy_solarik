//! Retained shader rows keep matrix bytes until their authoritative input changes.
use super::{GpuInstanceGeometryIds, Mat4, RayInstanceInput};

type Buffers<'a> = (
    &'a mut [Mat4],
    &'a mut [Mat4],
    &'a mut [GpuInstanceGeometryIds],
    &'a mut [u32],
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Changes {
    pub geometry: bool,
    pub material: bool,
}

#[inline]
pub(super) fn write(
    index: usize,
    input: &RayInstanceInput,
    (geometry, material): (&GpuInstanceGeometryIds, u32),
    (transforms, previous, geometries, materials): Buffers<'_>,
    transform_changed: bool,
) -> Changes {
    if transform_changed {
        transforms[index] = Mat4::from(input.transform);
        previous[index] = Mat4::from(input.previous);
    }
    let changed = Changes {
        geometry: geometries[index] != *geometry,
        material: materials[index] != material,
    };
    if changed.geometry {
        geometries[index] = geometry.clone();
    }
    if changed.material {
        materials[index] = material;
    }
    changed
}

pub(super) fn set_light_probability(
    index: usize,
    probability: f32,
    geometries: &mut [GpuInstanceGeometryIds],
    dirty: &mut Vec<u32>,
) {
    if geometries[index].light_probability.to_bits() != probability.to_bits() {
        geometries[index].light_probability = probability;
        dirty.push(index as u32);
    }
}

#[cfg(test)]
#[path = "publication_rows_tests.rs"]
mod tests;
