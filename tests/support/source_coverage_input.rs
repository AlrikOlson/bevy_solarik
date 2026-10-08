use std::io::{Read, Write};
use std::path::Path;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Expected {
    pub counts: [u32; 4],
    pub depth: [f64; 2],
}
#[derive(Default)]
pub struct Input {
    pub prototype: u32,
    pub width: u32,
    pub height: u32,
    pub alpha: Vec<u8>,
    pub positions: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    // First index, index count, cutoff bits, double-sided.
    pub parts: Vec<[u32; 4]>,
    pub rays: Vec<[f32; 8]>,
    pub expected: Vec<Expected>,
}

fn read<T: bytemuck::Pod + bytemuck::Zeroable>(source: &mut impl Read, count: usize) -> Vec<T> {
    let mut result = vec![T::zeroed(); count];
    source
        .read_exact(bytemuck::cast_slice_mut(&mut result))
        .unwrap();
    result
}

pub fn load(path: &Path) -> Input {
    const { assert!(cfg!(target_endian = "little")) };
    let mut file = std::fs::File::open(path).unwrap();
    assert!(file.metadata().unwrap().len() <= 512 << 20);
    let mut magic = [0; 8];
    file.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"SSSC0001");
    let head = read::<u32>(&mut file, 6);
    assert!((1..=8).contains(&head[0]));
    assert!((1..=4096).contains(&head[1]) && (1..=4096).contains(&head[2]));
    assert!((1..=262144).contains(&head[3]) && head[4] <= 32 && head[4] <= head[3]);
    let mut result = Input {
        prototype: head[5],
        width: head[1],
        height: head[2],
        alpha: read(&mut file, (head[1] * head[2] * 4) as usize),
        ..Default::default()
    };
    for _ in 0..head[0] {
        load_part(&mut file, &mut result);
    }
    result.rays = read(&mut file, head[3] as usize);
    result.expected = read(&mut file, head[4] as usize);
    assert_eq!(file.read(&mut [0]).unwrap(), 0, "trailing fixture bytes");
    for r in &result.rays {
        assert!(r.iter().all(|x| x.is_finite()) && r[3] >= 0. && r[7] > r[3]);
        assert!((r[4..7].iter().map(|x| x * x).sum::<f32>() - 1.).abs() < 1e-5);
    }
    result
}

fn load_part(file: &mut impl Read, input: &mut Input) {
    let header = read::<u32>(file, 4);
    let (vertices, indices) = (header[0] as usize, header[1] as usize);
    assert!(vertices > 0 && vertices + input.positions.len() <= 8_000_000);
    assert!(indices > 0 && indices % 3 == 0 && indices + input.indices.len() <= 18_000_000);
    let cutoff = f32::from_bits(header[2]);
    assert!(cutoff == -1. || (0. ..=1.).contains(&cutoff));
    assert!(header[3] <= 1);
    let p = read::<[f32; 4]>(file, vertices);
    let uv = read::<[f32; 2]>(file, vertices);
    let index = read::<u32>(file, indices);
    assert!(
        p.iter()
            .flatten()
            .chain(uv.iter().flatten())
            .all(|x| x.is_finite())
    );
    assert!(index.iter().all(|&i| (i as usize) < vertices));
    input
        .parts
        .push([input.indices.len() as u32, header[1], header[2], header[3]]);
    let base = input.positions.len() as u32;
    input.indices.extend(index.into_iter().map(|i| i + base));
    input.positions.extend(p);
    input.uvs.extend(uv);
}

pub fn write(path: &Path, bytes: &[u8]) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(bytes).unwrap();
}
