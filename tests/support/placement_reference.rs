use bevy_math::{DMat3, DQuat, DVec3};
use bevy_solarik::scene::placement::{GpuPlacementPage, PlacementPage};

pub struct Case {
    pub page: PlacementPage,
    pub camera: [f64; 3],
    pub root: DVec3,
    pub height: f64,
    pub quaternion: [f64; 4],
    pub radius: f64,
    pub prototype: u32,
    pub words: [u32; 7],
}

struct Cursor<'a>(&'a [u8]);
impl Cursor<'_> {
    fn u32(&mut self) -> u32 {
        let (head, tail) = self.0.split_at(4);
        self.0 = tail;
        u32::from_le_bytes(head.try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn f64(&mut self) -> f64 {
        let (head, tail) = self.0.split_at(8);
        self.0 = tail;
        f64::from_le_bytes(head.try_into().unwrap())
    }
}

pub fn fixture() -> (Vec<Case>, usize) {
    let path = std::env::var("SOLARIK_PLACEMENT_FIXTURE").expect("export placement fixture first");
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() >= 24 && &bytes[..8] == b"SSGP0001");
    let mut header = Cursor(&bytes[8..24]);
    let (stride, count, synthetic, real) = (header.u32(), header.u32(), header.u32(), header.u32());
    assert_eq!(stride, 172);
    assert!(synthetic >= 1000 && real > 0 && count == synthetic + real);
    assert_eq!(bytes.len(), 24 + count as usize * stride as usize);
    let cases = bytes[24..].chunks_exact(172).map(case).collect();
    (cases, synthetic as usize)
}

fn case(bytes: &[u8]) -> Case {
    let mut data = Cursor(bytes);
    let origin = core::array::from_fn(|_| data.f64());
    let camera = core::array::from_fn(|_| data.f64());
    let extent = core::array::from_fn(|_| data.f32());
    let height_base = data.f32();
    let height_span = data.f32();
    Case {
        page: PlacementPage {
            origin,
            extent,
            height_base,
            height_span,
        },
        camera,
        root: DVec3::from_array(core::array::from_fn(|_| data.f64())),
        height: data.f64(),
        quaternion: core::array::from_fn(|_| data.f64()),
        radius: data.f64(),
        prototype: data.u32(),
        words: core::array::from_fn(|_| data.u32()),
    }
}

pub fn matrix(q: [f64; 4]) -> DMat3 {
    DMat3::from_quat(DQuat::from_array(q))
}

/// A high-precision mathematical decoder independent of the production WGSL.
pub fn reference(case: &Case, stride: usize) -> (DVec3, f64, [f64; 4]) {
    let w = if stride == 3 {
        &case.words[..3]
    } else {
        &case.words[3..]
    };
    let p = [w[0] & 65535, w[0] >> 16, w[1] & 65535];
    let page = case.page.relative_to(case.camera).unwrap();
    let root = DVec3::from_array(core::array::from_fn(|i| {
        f64::from(page.origin_height[i])
            + f64::from(p[i]) / 65535.0 * f64::from(case.page.extent[i])
    }));
    let (h, values, largest, maximum) = if stride == 3 {
        (
            (w[1] >> 16) & 1023,
            [w[2] & 1023, (w[2] >> 10) & 1023, (w[2] >> 20) & 1023],
            w[2] >> 30,
            1023.0,
        )
    } else {
        (
            w[1] >> 16,
            [w[2] & 65535, w[2] >> 16, w[3] & 65535],
            w[3] >> 30,
            65535.0,
        )
    };
    let small = values.map(|v| {
        f64::from(v) / maximum * core::f64::consts::SQRT_2 - core::f64::consts::FRAC_1_SQRT_2
    });
    let omitted = (1.0 - small.iter().map(|v| v * v).sum::<f64>())
        .max(0.0)
        .sqrt();
    let mut j = 0;
    let q = core::array::from_fn(|i| {
        if i == largest as usize {
            omitted
        } else {
            let value = small[j];
            j += 1;
            value
        }
    });
    let height = f64::from(case.page.height_base)
        + f64::from(h) / maximum * f64::from(case.page.height_span);
    (root, height, q)
}

pub fn bound(case: &Case, root: DVec3, height: f64, q: [f64; 4]) -> f64 {
    let difference = (matrix(q) * height - matrix(case.quaternion) * case.height).to_cols_array();
    let frobenius = difference.iter().map(|v| v * v).sum::<f64>().sqrt();
    root.distance(case.root - DVec3::from_array(case.camera)) + frobenius * case.radius
}

pub fn arithmetic_allowance(case: &Case, page: GpuPlacementPage) -> f64 {
    // Conservative operation-count allowance, selected before GPU execution.
    // This fixture tolerance is not a complete representation-error proof.
    let magnitude = page.origin_height[..3]
        .iter()
        .map(|v| f64::from(v.abs()))
        .sum::<f64>()
        + case.page.extent.iter().map(|v| f64::from(*v)).sum::<f64>();
    64.0 * f64::from(f32::EPSILON) * (magnitude + case.height * (4.0 * case.radius + 1.0) + 1.0)
}

pub fn point(index: usize, radius: f32) -> DVec3 {
    let values = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
    ];
    DVec3::from_array(values[index].map(|v| f64::from(v * radius)))
}
