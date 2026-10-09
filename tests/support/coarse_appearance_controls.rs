use super::{
    acceleration, appearance_gpu as gpu,
    appearance_input::{Appearance, Vertex},
    appearance_oracle as oracle, input,
};
use bevy_solarik::coarse_appearance::{Material, Moment};
pub fn scene_input() -> input::Input {
    input::Input {
        prototype: 1,
        width: 1,
        height: 1,
        alpha: vec![255; 4],
        positions: vec![
            [-1., -1., 0., 0.],
            [1., -1., 0., 0.],
            [1., 1., 0., 0.],
            [-1., 1., 0., 0.],
        ],
        uvs: vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        indices: vec![0, 1, 2, 0, 2, 3],
        parts: vec![[0, 6, (-1f32).to_bits(), 1]],
        rays: vec![[0.; 8]; 2],
        expected: Vec::new(),
    }
}
pub fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cook: &wgpu::ComputePipeline,
    probe: &wgpu::ComputePipeline,
) -> usize {
    let source = scene_input();
    let scene = acceleration::build(device, queue, &source);
    let appearance = oracle::analytic();
    let uploaded = gpu::upload(device, queue, &appearance);
    let (rows, _) = gpu::run::<oracle::Sample>(
        device,
        queue,
        probe,
        &scene,
        &uploaded,
        appearance.requests.len(),
        Some(&appearance.requests),
    );
    oracle::compare(&appearance, &rows);
    let material = Material {
        base_color: [1.; 4],
        surface: [0.5, 0., 0.3, 0.],
        optical: [1.45, 0., 0., 0.],
    };
    let vertex = Vertex {
        normal: [0., 0., 1., 0.],
        tangent: [1., 0., 0., 1.],
    };
    let mut input = Appearance {
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
    let tasks: [[f32; 8]; 2] = [
        [0., 0., 0., 0.5, 0., 0., 1., 8.],
        [0., 0., 0., 0.5, 0., 0., -1., 8.],
    ];
    queue.write_buffer(&scene.buffers[4], 0, bytemuck::cast_slice(&tasks));
    let mut textures = core::array::from_fn::<_, 3, _>(|role| {
        core::array::from_fn(|i| input.textures[role][i] as f64 / 255.)
    });
    for v in &mut textures[0][..3] {
        *v = ((*v + 0.055) / 1.055).powf(2.4);
    }
    let expected = oracle::surface(material, vertex, textures);
    for zero_corners in [0, 1, 4] {
        // A zero raw corner must interpolate with its neighbors unchanged.
        // An entirely undefined frame must fail explicitly, never invent a normal.
        for (i, v) in input.vertices.iter_mut().enumerate() {
            v.tangent = if i < zero_corners {
                [0., 0., 0., 1.]
            } else {
                vertex.tangent
            };
        }
        let uploaded = gpu::upload(device, queue, &input);
        let (rows, _) = gpu::run::<[Moment; 8]>(device, queue, cook, &scene, &uploaded, 2, None);
        for row in &rows {
            let m = row[0];
            if zero_corners == 4 {
                assert_eq!(m.counts, [0, 4, 0, 0]);
                assert!(!m.valid(8));
            } else {
                assert!(m.valid(8));
                assert_eq!(m.counts[0], 64);
                for (values, reference) in [m.color, m.normal, m.surface].iter().zip(&expected) {
                    for (&value, &reference) in values.iter().zip(reference) {
                        assert!((value as f64 / 64. - reference).abs() < 0.0041);
                    }
                }
            }
            assert!(row[1..].iter().all(|m| m.counts == [0; 4] && m.valid(8)));
        }
    }
    appearance.requests.len() + tasks.len() * 3
}
