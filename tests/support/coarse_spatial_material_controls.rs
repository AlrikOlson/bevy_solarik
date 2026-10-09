use super::{
    Sample, acceleration, appearance_gpu as gpu,
    appearance_input::{Appearance, Vertex},
    appearance_oracle as oracle, input,
};
use bevy_solarik::coarse_appearance::Material;
pub fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
) -> usize {
    let material = Material {
        base_color: [1.; 4],
        surface: [0.5, 0., 0.3, 0.25],
        optical: [1.45, 0., 0., 0.],
    };
    let vertex = Vertex {
        normal: [0., 0., 1., 0.],
        tangent: [1., 0., 0., 1.],
    };
    let mut appearance = Appearance {
        width: 1,
        height: 1,
        materials: vec![material],
        vertices: vec![vertex; 4],
        textures: [
            vec![64, 128, 192, 255],
            vec![128, 128, 255, 255],
            vec![0, 128, 64, 255],
        ],
        requests: Vec::new(),
        expected: Vec::new(),
    };
    let mut textures = core::array::from_fn::<_, 3, _>(|role| {
        core::array::from_fn(|i| f64::from(appearance.textures[role][i]) / 255.)
    });
    for v in &mut textures[0][..3] {
        *v = ((*v + 0.055) / 1.055).powf(2.4);
    }
    let expected = oracle::surface(material, vertex, textures);
    let tasks: [[f32; 8]; 2] = [
        [0., 0., 0., 0.5, 0., 0., 1., 8.],
        [0., 0., 0., 0.5, 0., 0., -1., 8.],
    ];
    for (cutoff, two_sided, invalid) in [
        (-1_f32, 1, false),
        (-1., 0, false),
        (0.5, 1, false),
        (-1., 1, true),
    ] {
        let source = input::Input {
            width: 2,
            height: 1,
            alpha: vec![255, 255, 255, 0, 255, 255, 255, 255],
            positions: vec![
                [-0.5, -0.5, 0., 0.],
                [0.5, -0.5, 0., 0.],
                [0.5, 0.5, 0., 0.],
                [-0.5, 0.5, 0., 0.],
            ],
            uvs: vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            indices: vec![0, 1, 2, 0, 2, 3],
            parts: vec![[0, 6, cutoff.to_bits(), two_sided]],
            rays: vec![[0.; 8]; 2],
            ..Default::default()
        };
        let scene = acceleration::build(device, queue, &source);
        for v in &mut appearance.vertices {
            v.tangent = if invalid {
                [0., 0., 0., 1.]
            } else {
                vertex.tangent
            };
        }
        let uploaded = gpu::upload(device, queue, &appearance);
        queue.write_buffer(&scene.buffers[4], 0, bytemuck::cast_slice(&tasks));
        let (rows, _) = gpu::run::<Sample>(device, queue, pipeline, &scene, &uploaded, 128, None);
        for (face, row) in rows.chunks_exact(64).enumerate() {
            let count = row.iter().filter(|v| v.meta[0] != 0).count();
            assert_eq!(
                count,
                if two_sided == 0 && face == 0 {
                    0
                } else if cutoff < 0. {
                    64
                } else {
                    32
                }
            );
            for sample in row {
                if sample.meta[0] == 0 {
                    assert_eq!(sample.meta, [0; 4]);
                    continue;
                }
                assert_eq!(sample.meta[0], 0x8040_0800);
                if invalid {
                    assert_eq!(sample.meta[1], 4);
                    continue;
                }
                assert_eq!(sample.meta[1], 0);
                assert_eq!(sample.meta[2], 1);
                assert!(sample.surface.valid());
                for (actual, expected) in [
                    sample.surface.color,
                    sample.surface.normal,
                    sample.surface.surface,
                ]
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                {
                    assert!((f64::from(*actual) - expected).abs() < 0.0041);
                }
            }
        }
    }
    512
}
