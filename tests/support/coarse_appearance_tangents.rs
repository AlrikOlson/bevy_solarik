//! Exact Bevy-generated tangents, cached independently of material shaders.
use bevy_asset::RenderAssetUsages;
use bevy_mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues};
use std::{io::Write, path::Path, time::Instant};
const HEADER: usize = 76;
fn record(kind: &str, vertices: usize, seconds: f64) {
    let root = std::env::var("SOLARIK_COARSE_OUTPUT").unwrap();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(&root).join("tangent-preparation.jsonl"))
        .unwrap();
    writeln!(
        file,
        r#"{{"cache":"{kind}","vertices":{vertices},"seconds":{seconds}}}"#
    )
    .unwrap();
}
fn key(p: &[[f32; 3]], n: &[[f32; 3]], uv: &[[f32; 2]], indices: &[u32]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"source-appearance-tangents-v1");
    hash.update(include_bytes!("coarse_appearance_tangents.rs"));
    hash.update(include_bytes!("../../Cargo.toml"));
    hash.update(include_bytes!("../../Cargo.lock"));
    hash.update(
        std::env::var("SOLARIK_TANGENT_TOOLCHAIN")
            .unwrap_or_default()
            .as_bytes(),
    );
    hash.update(&[
        u8::from(cfg!(target_feature = "sse2")),
        u8::from(cfg!(target_feature = "avx2")),
        u8::from(cfg!(target_feature = "fma")),
    ]);
    for bytes in [
        bytemuck::cast_slice(p),
        bytemuck::cast_slice(n),
        bytemuck::cast_slice(uv),
        bytemuck::cast_slice(indices),
    ] {
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    *hash.finalize().as_bytes()
}
fn encode(key: [u8; 32], values: &[[f32; 4]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"SFAT0001");
    bytes.extend_from_slice(&key);
    let payload = bytemuck::cast_slice(values);
    bytes.extend_from_slice(blake3::hash(payload).as_bytes());
    bytes.extend_from_slice(&(values.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}
fn decode(bytes: &[u8], key: [u8; 32], count: usize) -> Option<Vec<[f32; 4]>> {
    if bytes.len() != HEADER + count.checked_mul(16)?
        || bytes.len() > 256 << 20
        || bytes.get(..8)? != b"SFAT0001"
        || bytes.get(8..40)? != key
        || u32::from_le_bytes(bytes.get(72..76)?.try_into().ok()?) as usize != count
        || blake3::hash(bytes.get(HEADER..)?).as_bytes() != bytes.get(40..72)?
    {
        return None;
    }
    let values: Vec<[f32; 4]> = bytes[HEADER..]
        .chunks_exact(16)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    values
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(values)
}
pub fn prepare(p: &[[f32; 3]], n: &[[f32; 3]], uv: &[[f32; 2]], indices: &[u32]) -> Vec<[f32; 4]> {
    let started = Instant::now();
    let key = key(p, n, uv, indices);
    let root = std::env::var("SOLARIK_TANGENT_CACHE").expect("separate tangent cache");
    let path = Path::new(&root).join(format!("{}.sfat", blake3::Hash::from(key).to_hex()));
    if path.exists() {
        assert!(path.metadata().unwrap().len() <= 256 << 20);
        let values = decode(&std::fs::read(&path).unwrap(), key, p.len())
            .expect("corrupt exact tangent cache");
        record("hit", p.len(), started.elapsed().as_secs_f64());
        return values;
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, p.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, n.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv.to_vec());
    mesh.insert_indices(Indices::U32(indices.to_vec()));
    mesh.generate_tangents()
        .expect("original Bevy tangent generation");
    let Some(VertexAttributeValues::Float32x4(values)) = mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
    else {
        panic!("missing tangents")
    };
    assert_eq!(values.len(), p.len());
    assert!(values.iter().flatten().all(|v| v.is_finite()));
    let bytes = encode(key, values);
    assert!(bytes.len() <= 256 << 20);
    std::fs::create_dir_all(&root).unwrap();
    let pending = path.with_extension(format!("pending-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&pending)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    drop(file);
    assert!(decode(&std::fs::read(&pending).unwrap(), key, p.len()).is_some());
    std::fs::rename(pending, path).unwrap();
    record("cold", p.len(), started.elapsed().as_secs_f64());
    values.clone()
}
#[test]
fn exact_tangent_cache_rejects_foreign_corrupt_or_truncated_data() {
    let key = [7; 32];
    let values = [[1., 0., 0., -1.], [0., 0., 0., 0.]];
    let mut bytes = encode(key, &values);
    assert_eq!(decode(&bytes, key, 2).unwrap(), values);
    assert!(decode(&bytes, [8; 32], 2).is_none());
    assert!(decode(&bytes, key, 1).is_none());
    assert!(decode(&bytes[..bytes.len() - 1], key, 2).is_none());
    *bytes.last_mut().unwrap() ^= 1;
    assert!(decode(&bytes, key, 2).is_none());
}
#[test]
fn exact_tangent_key_tracks_geometry_normal_uv_and_topology() {
    let p = [[0.; 3], [1., 0., 0.], [0., 1., 0.]];
    let mut n = [[0., 0., 1.]; 3];
    let uv = [[0., 0.], [1., 0.], [0., 1.]];
    let indices = [0, 1, 2];
    let old = key(&p, &n, &uv, &indices);
    n[0] = [0., 1., 0.];
    assert_ne!(old, key(&p, &n, &uv, &indices));
    assert_ne!(old, key(&p, &[[0., 0., 1.]; 3], &[[0.; 2]; 3], &indices));
    assert_ne!(old, key(&p, &[[0., 0., 1.]; 3], &uv, &[0, 2, 1]));
}
