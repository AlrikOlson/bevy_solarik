use super::input;
use bevy_solarik::coarse_appearance::Material;
use std::{io::Read, path::Path};
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
}
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Request {
    pub part: [u32; 4],
    pub uv: [f32; 4],
}
pub struct Appearance {
    pub width: u32,
    pub height: u32,
    pub materials: Vec<Material>,
    pub vertices: Vec<Vertex>,
    pub textures: [Vec<u8>; 3],
    pub requests: Vec<Request>,
    pub expected: Vec<[[f64; 4]; 3]>,
}
fn read<T: bytemuck::Pod + bytemuck::Zeroable>(file: &mut impl Read, count: usize) -> Vec<T> {
    let mut values = vec![T::zeroed(); count];
    file.read_exact(bytemuck::cast_slice_mut(&mut values))
        .unwrap();
    values
}
pub fn load(path: &Path, source: &input::Input) -> Appearance {
    let mut file = std::fs::File::open(path).unwrap();
    assert!(file.metadata().unwrap().len() < 512 << 20);
    let magic = read::<u8>(&mut file, 8);
    assert_eq!(&magic, b"SSAM0001");
    let head = read::<u32>(&mut file, 6);
    assert_eq!(head[0], source.prototype);
    assert_eq!(head[1] as usize, source.parts.len());
    assert!((1..=4096).contains(&head[2]) && (1..=4096).contains(&head[3]));
    assert!((1..=64).contains(&head[4]) && head[5] == 0);
    let mut result = Appearance {
        width: head[2],
        height: head[3],
        materials: read(&mut file, head[1] as usize),
        vertices: Vec::new(),
        textures: Default::default(),
        requests: Vec::new(),
        expected: Vec::new(),
    };
    assert!(result.materials.iter().all(Material::valid));
    let mut original = Vec::new();
    for part in &source.parts {
        let lengths = read::<u32>(&mut file, 2);
        let count = lengths[0] as usize;
        assert!(count > 0 && original.len() + count <= source.positions.len());
        assert_eq!(lengths[1], part[1]);
        let p = read::<[f32; 3]>(&mut file, count);
        let n = read::<[f32; 3]>(&mut file, count);
        let uv = read::<[f32; 2]>(&mut file, count);
        let indices = read::<u32>(&mut file, lengths[1] as usize);
        assert_eq!(&uv, &source.uvs[original.len()..original.len() + count]);
        assert!(p.iter().chain(&n).flatten().all(|v| v.is_finite()));
        assert!(indices.iter().all(|&i| (i as usize) < count));
        for (local, &global) in indices
            .iter()
            .zip(&source.indices[part[0] as usize..(part[0] + part[1]) as usize])
        {
            assert_eq!(*local + original.len() as u32, global);
        }
        let tangents = super::tangents::prepare(&p, &n, &uv, &indices);
        for (n, t) in n.iter().zip(&tangents) {
            assert!(t.iter().all(|v| v.is_finite()));
            assert!((n.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 0.001);
            let length = t[..3].iter().map(|v| v * v).sum::<f32>();
            // Mikk can leave zero corner tangents. Preserve them exactly:
            // Solarik normalizes the interpolated frame, which the cook checks.
            assert!(length == 0. || (length - 1.).abs() < 0.001);
            assert!(t[3].abs() == 1.);
            result.vertices.push(Vertex {
                normal: [n[0], n[1], n[2], 0.],
                tangent: *t,
            });
        }
        original.extend(p);
    }
    assert_eq!(original.len(), source.positions.len());
    let low = core::array::from_fn::<_, 3, _>(|i| {
        original
            .iter()
            .map(|p| f64::from(p[i]))
            .fold(f64::INFINITY, f64::min)
    });
    let high = core::array::from_fn::<_, 3, _>(|i| {
        original
            .iter()
            .map(|p| f64::from(p[i]))
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let height = high[1] - low[1];
    let offset = [
        -(low[0] + high[0]) * 0.5,
        -low[1],
        -(low[2] + high[2]) * 0.5,
    ];
    for (p, normalized) in original.iter().zip(&source.positions) {
        for i in 0..3 {
            assert_eq!(
                ((f64::from(p[i]) + offset[i]) / height) as f32,
                normalized[i]
            );
        }
    }
    let texture_bytes = result.width as usize * result.height as usize * 4 * result.materials.len();
    assert!(texture_bytes * 3 < 384 << 20);
    for texture in &mut result.textures {
        *texture = read(&mut file, texture_bytes);
    }
    for _ in 0..head[4] {
        let request = read::<Request>(&mut file, 1)[0];
        assert!((request.part[0] as usize) < result.materials.len());
        result.requests.push(request);
        result.expected.push(read::<[[f64; 4]; 3]>(&mut file, 1)[0]);
    }
    assert_eq!(file.read(&mut [0]).unwrap(), 0);
    result
}
