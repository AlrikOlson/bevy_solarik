//! Attribute error for textured surfaces, including planar alpha foliage.
pub(super) fn weights(vertices: &[f32], stride: usize, indices: &[u32]) -> [f32; 5] {
    // meshoptimizer's documented automatic weight: inverse sqrt(mean UV area).
    // UVs follow position and normal in the validated packed vertex layout.
    let uv = |i: u32| {
        let offset = i as usize * stride + 6;
        bevy_math::DVec2::new(vertices[offset] as f64, vertices[offset + 1] as f64)
    };
    let area: f64 = indices
        .chunks_exact(3)
        .map(|t| {
            let a = uv(t[1]) - uv(t[0]);
            let b = uv(t[2]) - uv(t[0]);
            a.perp_dot(b).abs() * 0.5
        })
        .sum();
    let weight = if area > 0.0 && area.is_finite() {
        (((indices.len() / 3) as f64 / area)
            .sqrt()
            .min(f32::MAX as f64)) as f32
    } else {
        0.0
    };
    [0.5, 0.5, 0.5, weight, weight]
}

#[cfg(test)]
mod tests {
    use super::*;
    use meshopt::{SimplifyOptions, VertexDataAdapter, simplify_with_attributes_and_locks};

    #[test]
    fn uv_weight_tracks_chart_density_and_degenerate_charts_are_finite() {
        let mut vertices = [0.0; 24];
        vertices[14] = 1.0;
        vertices[23] = 1.0;
        let base = weights(&vertices, 8, &[0, 1, 2])[3];
        assert!((base - 2.0_f32.sqrt()).abs() < 1e-6);
        vertices[14] = 4.0;
        vertices[23] = 4.0;
        assert!((weights(&vertices, 8, &[0, 1, 2])[3] * 4.0 - base).abs() < 1e-6);
        assert_eq!(weights(&[0.0; 24], 8, &[0, 1, 2])[3], 0.0);
    }

    #[test]
    fn attribute_error_is_bounded_by_spatial_support() {
        let vertices: [[f32; 8]; 5] = [
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0., 1., 1., 0.],
            [1., 1., 0., 0., 0., 1., 1., 1.],
            [0., 1., 0., 0., 0., 1., 0., 1.],
            [0.5, 0.5, 0., 0., 0., 1., 1000., 0.5],
        ];
        let indices = [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let bytes = bytemuck::cast_slice(&vertices);
        let adapter = VertexDataAdapter::new(bytes, 32, 0).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        let mut error = 0.;
        let reduced = simplify_with_attributes_and_locks(
            &indices,
            &adapter,
            &floats[3..],
            &[0.5, 0.5, 0.5, 100., 100.],
            32,
            &[true, true, true, true, false],
            6,
            f32::MAX,
            SimplifyOptions::ErrorAbsolute | SimplifyOptions::ErrorClamped,
            Some(&mut error),
        );
        assert_eq!(reduced.len(), 6);
        assert!(
            error <= 1.001,
            "unit-area attribute error escaped support: {error}"
        );
    }

    #[test]
    fn planar_texture_distortion_is_not_free() {
        let vertices: [[f32; 8]; 5] = [
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0., 1., 1., 0.],
            [1., 1., 0., 0., 0., 1., 1., 1.],
            [0., 1., 0., 0., 0., 1., 0., 1.],
            [0.5, 0.5, 0., 0., 0., 1., 0.8, 0.5],
        ];
        let indices = [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
        let bytes = bytemuck::cast_slice(&vertices);
        let adapter = VertexDataAdapter::new(bytes, 32, 0).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(bytes);
        let mut error = 0.;
        let reduced = simplify_with_attributes_and_locks(
            &indices,
            &adapter,
            &floats[3..],
            &weights(floats, 8, &indices),
            32,
            &[false; 5],
            6,
            0.001,
            SimplifyOptions::ErrorAbsolute,
            Some(&mut error),
        );
        assert!(
            reduced.contains(&4),
            "UV kink removed as zero-error planar geometry: {reduced:?}, error={error}"
        );
    }
}
