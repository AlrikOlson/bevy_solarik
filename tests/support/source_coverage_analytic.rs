use super::{
    Probe,
    input::{Expected, Input},
};

fn quad(input: &mut Input, z: f32, cutoff: f32, sided: bool, reverse: bool) {
    let base = input.positions.len() as u32;
    input.positions.extend([
        [-1., -1., z, 0.],
        [1., -1., z, 0.],
        [1., 1., z, 0.],
        [-1., 1., z, 0.],
    ]);
    input.uvs.extend([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]);
    let indices = if reverse {
        [0, 2, 1, 0, 3, 2]
    } else {
        [0, 1, 2, 0, 2, 3]
    };
    input.parts.push([
        input.indices.len() as u32,
        6,
        cutoff.to_bits(),
        u32::from(sided),
    ]);
    input.indices.extend(indices.map(|i| i + base));
}
fn ray(x: f32, reverse: bool) -> [f32; 8] {
    [
        x,
        0.123,
        if reverse { 2. } else { -1. },
        0.,
        0.,
        0.,
        if reverse { -1. } else { 1. },
        3.,
    ]
}
pub fn scene() -> Input {
    let mut input = Input {
        width: 2,
        height: 1,
        alpha: vec![255, 255, 255, 0, 255, 255, 255, 255],
        ..Default::default()
    };
    quad(&mut input, 0., 0.5, true, false);
    quad(&mut input, 0.4, -1., true, true);
    quad(&mut input, 0.8, -1., false, false);
    for reverse in [false, true] {
        for x in [-0.5, 0., 0.5, 1.5] {
            input.rays.push(ray(x, reverse));
            let leaf = u32::from((0. ..=1.).contains(&x));
            let (accepted, near, far) = if x > 1. {
                (0, 0., 0.)
            } else if reverse {
                (2 + leaf, 1.2, if leaf == 1 { 2. } else { 1.6 })
            } else {
                (1 + leaf, if leaf == 1 { 1. } else { 1.4 }, 1.4)
            };
            input.expected.push(Expected {
                counts: [if x > 1. { 0 } else { 3 }, accepted, leaf, 0],
                depth: [near, far],
            });
        }
    }
    input
}
pub fn overflow() -> Input {
    let mut input = Input {
        width: 1,
        height: 1,
        alpha: vec![255; 4],
        rays: vec![ray(0.123, false)],
        ..Default::default()
    };
    for i in 0..4097 {
        let z = i as f32 / 4097.;
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
    input
}
pub fn check(input: &Input, rows: &[Probe], tolerance: f64) {
    assert_eq!(input.rays.len(), rows.len());
    for (index, (expected, row)) in input.expected.iter().zip(rows).enumerate() {
        assert_eq!(
            row.counts, expected.counts,
            "source{} control{index}",
            input.prototype
        );
        for (actual, expected) in row.depth[..2].iter().zip(expected.depth) {
            assert!(
                (f64::from(*actual) - expected).abs() < tolerance,
                "source{} control{index}: {actual} != {expected}",
                input.prototype
            );
        }
    }
}
pub fn validate(input: &Input, rows: &[Probe]) {
    for (index, (ray, row)) in input.rays.iter().zip(rows).enumerate() {
        let [candidates, accepted, leaves, overflow] = row.counts;
        assert_eq!(overflow, 0, "source{} ray{index} overflow", input.prototype);
        assert!(leaves <= accepted && accepted <= candidates);
        assert!(
            row.depth
                .iter()
                .chain(row.normal.iter())
                .all(|x| x.is_finite())
        );
        if accepted == 0 {
            assert_eq!(row.first, [0; 4]);
            continue;
        }
        assert!(row.depth[0] >= ray[3] - 1e-4 && row.depth[1] <= ray[7] + 1e-4);
        assert!(row.depth[0] <= row.depth[1] && row.depth[2] <= accepted as f32 * 1.0001);
        let part = input.parts[(row.first[0] - 1) as usize];
        assert!(row.first[1] < part[1] / 3 && row.first[2] <= 1);
        assert!((row.normal[..3].iter().map(|x| x * x).sum::<f32>() - 1.).abs() < 1e-4);
    }
}
