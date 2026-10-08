use super::{Probe, acceleration, gpu, input::Input};

fn quad(input: &mut Input, z: f32, cutoff: f32, sided: bool) {
    let base = input.positions.len() as u32;
    input.positions.extend([
        [-1., -1., z, 0.],
        [1., -1., z, 0.],
        [1., 1., z, 0.],
        [-1., 1., z, 0.],
    ]);
    input.uvs.extend([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]);
    input.parts.push([
        input.indices.len() as u32,
        6,
        cutoff.to_bits(),
        u32::from(sided),
    ]);
    input.indices.extend([0, 1, 2, 0, 2, 3].map(|i| i + base));
}
fn scene() -> Input {
    Input {
        width: 2,
        height: 1,
        alpha: vec![255, 255, 255, 0, 255, 255, 255, 255],
        ..Default::default()
    }
}
fn task(z: f32, direction: f32) -> [f32; 8] {
    [0., 0., z, 1., 0., 0., direction, 8.]
}
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    input: &Input,
) -> Vec<Probe> {
    let scene = acceleration::build(device, queue, input);
    gpu::trace(device, queue, pipeline, &scene, input.rays.len())
}
fn near(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
}
fn coverage(row: &Probe, expected: u32) {
    assert!(row.validate(8), "{row:?}");
    assert_eq!(row.counts[0], 64);
    assert_eq!(row.counts[1], expected);
    near(row.diagonal[2], expected as f32);
}
pub fn controls(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
) -> usize {
    let mut input = scene();
    quad(&mut input, 3., -1., true);
    input.rays = vec![task(0., 1.)];
    coverage(&run(device, queue, pipeline, &input)[0], 0);

    let mut input = scene();
    quad(&mut input, 0., -1., true);
    input.rays = vec![task(0., 1.), task(0., -1.)];
    for row in run(device, queue, pipeline, &input) {
        coverage(&row, 64);
        near(row.depth[0], 0.5);
        near(row.depth[1], 0.5);
        assert_eq!(row.materials[0], 64);
        near(row.normal[2], 64.);
    }

    input.parts[0][2] = 0.5f32.to_bits();
    for row in run(device, queue, pipeline, &input) {
        coverage(&row, 32);
    }

    input.parts[0][2] = (-1f32).to_bits();
    input.parts[0][3] = 0;
    let rows = run(device, queue, pipeline, &input);
    coverage(&rows[0], 0);
    coverage(&rows[1], 64);

    let mut input = scene();
    quad(&mut input, -0.5, -1., true);
    quad(&mut input, 0.5, -1., true);
    input.rays = vec![task(0., 1.), task(0., -1.)];
    for (i, row) in run(device, queue, pipeline, &input).iter().enumerate() {
        coverage(row, 64);
        near(row.depth[0], 0.25);
        near(row.depth[1], 0.75);
        near(row.depth[2], 16.);
        near(row.depth[3], 48.);
        assert_eq!(row.materials[i], 64);
    }

    let mut input = scene();
    quad(&mut input, 1., -1., true);
    input.rays = vec![task(0., 1.), task(2., 1.), task(2., -1.)];
    let rows = run(device, queue, pipeline, &input);
    coverage(&rows[0], 0);
    coverage(&rows[1], 64);
    coverage(&rows[2], 64);
    near(rows[1].depth[0], 0.);
    near(rows[2].depth[0], 1.);

    let mut input = scene();
    input.rays = vec![[0., 0., 0., 1., 0., 0., 1., 1.]];
    for i in 0..4097 {
        let z = -0.9 + i as f32 / 4097.;
        let base = input.positions.len() as u32;
        input
            .positions
            .extend([[-1., -1., z, 0.], [1., -1., z, 0.], [0., 1., z, 0.]]);
        input.uvs.extend([[0., 0.]; 3]);
        input.indices.extend([base, base + 1, base + 2]);
    }
    input
        .parts
        .push([0, input.indices.len() as u32, (-1f32).to_bits(), 1]);
    assert_eq!(run(device, queue, pipeline, &input)[0].counts[3], 1);
    13
}
