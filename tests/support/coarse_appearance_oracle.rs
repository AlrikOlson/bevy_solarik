use super::appearance_input::{Appearance, Request, Vertex};
use bevy_math::DVec3;
use bevy_solarik::coarse_appearance::Material;
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Sample {
    pub textures: [[f32; 4]; 3],
    pub color: [f32; 4],
    pub normal: [f32; 4],
    pub surface: [f32; 4],
}
pub fn surface(material: Material, vertex: Vertex, textures: [[f64; 4]; 3]) -> [[f64; 4]; 3] {
    let vector = |v: [f32; 4]| DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
    let n = vector(vertex.normal).normalize();
    let t = vector(vertex.tangent).normalize();
    let b = n.cross(t) * vertex.tangent[3] as f64;
    let x = textures[1][0] * 2. - 1.;
    let y = (textures[1][1] * 2. - 1.) * if material.optical[1] == 0. { 1. } else { -1. };
    let z = (1. - x * x - y * y).max(0.).sqrt();
    let normal = (t * x + b * y + n * z).normalize();
    let color = core::array::from_fn(|i| textures[0][i] * material.base_color[i] as f64);
    let roughness = material.surface[0] as f64 * textures[2][1];
    [
        color,
        [normal.x, normal.y, normal.z, 0.],
        [
            roughness.powi(4),
            material.surface[1] as f64 * textures[2][2],
            material.surface[2] as f64,
            material.surface[3] as f64,
        ],
    ]
}
pub fn compare(input: &Appearance, rows: &[Sample]) -> (f64, f64) {
    assert_eq!(rows.len(), input.requests.len());
    let mut texture_error = 0f64;
    let mut math_error = 0f64;
    for (i, (row, request)) in rows.iter().zip(&input.requests).enumerate() {
        for (actual, expected) in row
            .textures
            .iter()
            .flatten()
            .zip(input.expected[i].iter().flatten())
        {
            let error = (*actual as f64 - expected).abs();
            texture_error = texture_error.max(error);
            assert!(
                actual.is_finite() && error <= 0.0041,
                "sample{i} texture error{error}"
            );
        }
        let sampled = row.textures.map(|v| v.map(f64::from));
        let expected = surface(
            input.materials[request.part[0] as usize],
            input.vertices[request.part[1] as usize],
            sampled,
        );
        for (actual, expected) in [row.color, row.normal, row.surface]
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
        {
            let error = (*actual as f64 - expected).abs();
            math_error = math_error.max(error);
            assert!(
                actual.is_finite() && error <= 1e-5,
                "sample{i} material math error{error}"
            );
        }
    }
    (texture_error, math_error)
}
fn sample(pixels: &[u8], uv: [f32; 4], srgb: bool) -> [f64; 4] {
    let p = [uv[0] as f64 * 2. - 0.5, uv[1] as f64 * 2. - 0.5];
    let low = p.map(f64::floor);
    let w = [p[0] - low[0], p[1] - low[1]];
    let mut result = [0.; 4];
    for y in 0..2 {
        for x in 0..2 {
            let address = (((low[1] as i32 + y).rem_euclid(2) * 2
                + (low[0] as i32 + x).rem_euclid(2))
                * 4) as usize;
            let weight =
                if x == 0 { 1. - w[0] } else { w[0] } * if y == 0 { 1. - w[1] } else { w[1] };
            for i in 0..4 {
                let mut v = pixels[address + i] as f64 / 255.;
                if srgb && i < 3 {
                    v = if v <= 0.04045 {
                        v / 12.92
                    } else {
                        ((v + 0.055) / 1.055).powf(2.4)
                    };
                }
                result[i] += v * weight;
            }
        }
    }
    result
}
pub fn analytic() -> Appearance {
    let material = Material {
        base_color: [0.8, 0.6, 0.4, 1.],
        surface: [0.5, 0.7, 0.3, 0.25],
        optical: [1.45, 0., 0., 0.],
    };
    let mut result = Appearance {
        width: 2,
        height: 2,
        materials: vec![
            material,
            Material {
                optical: [1.45, 1., 0., 0.],
                ..material
            },
        ],
        vertices: vec![
            Vertex {
                normal: [0., 0., 1., 0.],
                tangent: [1., 0., 0., 1.],
            },
            Vertex {
                normal: [0., 0., 1., 0.],
                tangent: [1., 0., 0., -1.],
            },
        ],
        textures: [
            vec![
                0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 128, 128, 64, 255, 0, 255,
            ],
            vec![
                160, 144, 255, 255, 96, 160, 255, 255, 144, 96, 255, 255, 128, 128, 255, 255,
            ],
            vec![
                0, 0, 255, 255, 0, 255, 0, 255, 0, 128, 64, 255, 0, 64, 128, 255,
            ],
        ],
        requests: Vec::new(),
        expected: Vec::new(),
    };
    for texture in &mut result.textures {
        texture.extend_from_within(..);
    }
    for part in 0..2 {
        for vertex in 0..2 {
            for uv in [
                [0.25, 0.25, 0., 0.],
                [0.5, 0.5, 0., 0.],
                [-0.5, 1.5, 0., 0.],
            ] {
                result.requests.push(Request {
                    part: [part, vertex, 0, 0],
                    uv,
                });
                result.expected.push(core::array::from_fn(|role| {
                    sample(&result.textures[role][..16], uv, role == 0)
                }));
            }
        }
    }
    result
}
