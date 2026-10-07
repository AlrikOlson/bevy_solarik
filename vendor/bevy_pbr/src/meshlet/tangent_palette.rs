//! Lossless reuse of authored tangents across meshlets and LODs.
use bevy_math::Vec4;
use bevy_platform::collections::HashMap;

pub(in crate::meshlet) fn intern(values: impl IntoIterator<Item = Vec4>) -> (Vec<Vec4>, Vec<u32>) {
    let mut palette = Vec::new();
    let mut ids = HashMap::<[u32; 4], u32>::default();
    let indices = values
        .into_iter()
        .map(|value| {
            let key = value.to_array().map(f32::to_bits);
            *ids.entry(key).or_insert_with(|| {
                let id = palette.len() as u32;
                palette.push(value);
                id
            })
        })
        .collect();
    (palette, indices)
}

pub(in crate::meshlet) fn rebase(indices: &[u32], palette_byte_offset: u64) -> Vec<u32> {
    assert!(palette_byte_offset.is_multiple_of(16));
    let base = u32::try_from(palette_byte_offset / 16).expect("tangent palette offset overflow");
    indices
        .iter()
        .map(|index| index.checked_add(base).expect("tangent index overflow"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_values_share_storage_without_changing_any_bits() {
        let a = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let b = Vec4::new(1.0, -0.0, 0.0, -1.0);
        let original = [a, b, a, Vec4::ZERO, b, a];
        let (palette, indices) = intern(original);
        assert_eq!(palette.len(), 3);
        assert_eq!(indices, [0, 1, 0, 2, 1, 0]);
        for (expected, index) in original.into_iter().zip(indices) {
            assert_eq!(
                expected.to_array().map(f32::to_bits),
                palette[index as usize].to_array().map(f32::to_bits)
            );
        }
    }

    #[test]
    fn independent_assets_rebase_into_their_own_palette() {
        assert_eq!(rebase(&[1, 0, 1], 0), [1, 0, 1]);
        assert_eq!(rebase(&[1, 0, 1], 48), [4, 3, 4]);
    }
}
