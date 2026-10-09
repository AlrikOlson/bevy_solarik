//! Compare all compiled output bytes and exercise invalidation/fallback paths.
use super::*;
use bevy_asset::RenderAssetUsages;
use bevy_mesh::VertexAttributeValues;
use bevy_render::render_resource::PrimitiveTopology;

fn source() -> Mesh {
    bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::new);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            let first = positions.len() as u32;
            for (dx, dy) in [(0., 0.), (0.1, 0.), (0.1, 0.1), (0., 0.1)] {
                positions.push([x as f32 * 0.2 + dx, y as f32 * 0.2 + dy, 0.]);
                normals.push([0., 0., 1.]);
                uvs.push([dx * 10., dy * 10.]);
            }
            indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh.generate_tangents().unwrap();
    mesh
}

fn directory() -> std::path::PathBuf {
    let parent = std::env::temp_dir().canonicalize().unwrap();
    let path = parent.join(format!(
        "space-sim-mvg-ddc-{}-{}",
        std::process::id(),
        NEXT_WRITE.fetch_add(1, Ordering::Relaxed)
    ));
    assert!(path.is_absolute() && path.starts_with(&parent));
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn derived_cache_preserves_all_buffers_tangent_palette_and_dilation_errors_and_warm_hit() {
    // This native acceptance fixture must exercise cache hits, not silently
    // pass by recompiling after failed Windows toolchain discovery.
    assert_eq!(env!("MESHLET_CACHE_COMPILER_KNOWN"), "1");
    let source = source();
    let directory = directory();
    for preserve in [false, true] {
        let expected = convert(&source, 4, preserve).unwrap();
        let key = key(&source, 4, preserve, RECIPE);
        let path = directory.join(format!("{}.mvg", key.to_hex()));
        publish(&directory, &path, key.as_bytes(), &expected).unwrap();
        let loaded = load(&path, key.as_bytes()).unwrap();
        assert_eq!(
            encode(&loaded),
            encode(&expected),
            "every typed buffer and bound must remain exact"
        );
        let hit = cached(&source, 4, preserve, &directory, || {
            panic!("a verified warm hit must not run the geometry compiler")
        })
        .unwrap();
        assert_eq!(encode(&hit), encode(&expected));
        fs::remove_file(path).unwrap();
    }
    fs::remove_file(directory.join(retention::LOCK)).unwrap();
    assert!(fs::read_dir(&directory).unwrap().next().is_none());
    fs::remove_dir(directory).unwrap();
}

#[test]
fn derived_cache_key_covers_all_attributes_indices_topology_policy_and_recipe() {
    let source = source();
    let original = key(&source, 4, false, RECIPE);
    for attribute in [
        Mesh::ATTRIBUTE_POSITION,
        Mesh::ATTRIBUTE_NORMAL,
        Mesh::ATTRIBUTE_UV_0,
        Mesh::ATTRIBUTE_TANGENT,
    ] {
        let mut changed = source.clone();
        match changed.attribute_mut(attribute).unwrap() {
            VertexAttributeValues::Float32x2(values) => values[0][0] += 0.125,
            VertexAttributeValues::Float32x3(values) => values[0][0] += 0.125,
            VertexAttributeValues::Float32x4(values) => values[0][0] += 0.125,
            _ => panic!("source layout"),
        }
        assert_ne!(original, key(&changed, 4, false, RECIPE));
    }
    let mut changed = source.clone();
    if let Some(Indices::U32(indices)) = changed.indices_mut() {
        indices.swap(0, 1);
    }
    assert_ne!(original, key(&changed, 4, false, RECIPE));
    assert_ne!(original, key(&source, 5, false, RECIPE));
    assert_ne!(original, key(&source, 4, true, RECIPE));
    assert_ne!(
        original,
        key(&source, 4, false, b"different compiler/toolchain recipe")
    );
    let mut changed = Mesh::new(
        PrimitiveTopology::TriangleStrip,
        RenderAssetUsages::default(),
    );
    for (attribute, values) in source.attributes() {
        changed.insert_attribute(*attribute, values.clone());
    }
    changed.insert_indices(source.indices().unwrap().clone());
    assert_ne!(original, key(&changed, 4, false, RECIPE));
}

#[test]
fn derived_cache_corruption_wrong_key_version_lengths_and_trailing_data_are_safe_misses() {
    let mesh = convert(&source(), 4, true).unwrap();
    let directory = directory();
    let path = directory.join("entry.mvg");
    let key = blake3::hash(b"test entry");
    publish(&directory, &path, key.as_bytes(), &mesh).unwrap();
    let valid = fs::read(&path).unwrap();
    for length in [0, 7, HEADER - 1, HEADER, valid.len() - 1] {
        fs::write(&path, &valid[..length]).unwrap();
        assert!(load(&path, key.as_bytes()).is_err());
    }
    for offset in [0, 7, 8, 40, 48, HEADER, valid.len() - 1] {
        let mut changed = valid.clone();
        changed[offset] ^= 1;
        fs::write(&path, changed).unwrap();
        assert!(load(&path, key.as_bytes()).is_err());
    }
    fs::write(&path, &valid).unwrap();
    assert!(load(&path, blake3::hash(b"wrong source").as_bytes()).is_err());
    // Valid checksum cannot turn an impossible advertised slice into an allocation.
    let mut changed = valid.clone();
    let first_length = HEADER + size_of::<MeshletAabb>() + 4;
    changed[first_length..first_length + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let checksum = blake3::hash(&changed[HEADER..]);
    changed[48..80].copy_from_slice(checksum.as_bytes());
    fs::write(&path, changed).unwrap();
    assert!(load(&path, key.as_bytes()).is_err());
    let mut changed = valid.clone();
    changed.push(0);
    let size = (changed.len() - HEADER) as u64;
    changed[40..48].copy_from_slice(&size.to_le_bytes());
    let checksum = blake3::hash(&changed[HEADER..]);
    changed[48..80].copy_from_slice(checksum.as_bytes());
    fs::write(&path, changed).unwrap();
    assert!(load(&path, key.as_bytes()).is_err());
    // Reject metadata size before reading or allocating a oversized payload.
    fs::File::create(&path)
        .unwrap()
        .set_len((MAX_ENTRY + 1) as u64)
        .unwrap();
    assert!(load(&path, key.as_bytes()).is_err());
    fs::remove_file(path).unwrap();
    fs::remove_file(directory.join(retention::LOCK)).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn derived_cache_unwritable_directory_keeps_the_exact_compiled_result() {
    let source = source();
    let expected = convert(&source, 4, false).unwrap();
    let directory = directory();
    let file = directory.join("ordinary-file");
    fs::write(&file, b"keep").unwrap();
    let actual = MeshletMesh::from_mesh_cached(&source, 4, false, &file).unwrap();
    assert_eq!(encode(&actual), encode(&expected));
    assert_eq!(fs::read(&file).unwrap(), b"keep");
    fs::remove_file(file).unwrap();
    fs::remove_dir(directory).unwrap();
}
#[test]
fn derived_cache_recycling_counts_headers_preserves_foreign_and_recent_entries() {
    let directory = directory();
    let old = retention::tests::entry(&directory, b"old", 100, 1);
    let recent = retention::tests::entry(&directory, b"recent", 100, 2);
    let foreign = directory.join("foreign.mvg");
    let partial = directory.join("active.partial");
    fs::write(&foreign, b"foreign").unwrap();
    fs::write(&partial, b"partial").unwrap();
    let lease = retention::reserve(&directory, 100, 207).unwrap();
    assert!(!old.exists());
    assert!(recent.exists());
    assert_eq!(fs::read(&foreign).unwrap(), b"foreign");
    assert_eq!(fs::read(&partial).unwrap(), b"partial");
    assert!(retention::reserve(&directory, 100, 207).is_err());
    drop(lease);
    // Unknown .mvg files count against the cap but are never evicted.
    assert!(retention::reserve(&directory, 201, 207).is_err());
    assert!(foreign.exists());
    for entry in fs::read_dir(&directory).unwrap() {
        fs::remove_file(entry.unwrap().path()).unwrap();
    }
    fs::remove_dir(directory).unwrap();
}

#[test]
fn derived_cache_warm_use_refreshes_recency_and_lock_contention_keeps_output() {
    let source = source();
    let mesh = convert(&source, 4, false).unwrap();
    let directory = directory();
    let key = key(&source, 4, false, RECIPE);
    let path = directory.join(format!("{}.mvg", key.to_hex()));
    publish(&directory, &path, key.as_bytes(), &mesh).unwrap();
    let old = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1);
    fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(old)
        .unwrap();
    let hit = cached(&source, 4, false, &directory, || panic!("warm cache")).unwrap();
    assert_eq!(encode(&hit), encode(&mesh));
    assert!(fs::metadata(&path).unwrap().modified().unwrap() > old);
    fs::remove_file(&path).unwrap();
    let lease = retention::reserve(&directory, 0, MAX_DIRECTORY).unwrap();
    let fallback = cached(&source, 4, false, &directory, || Ok(mesh.clone())).unwrap();
    assert_eq!(encode(&fallback), encode(&mesh));
    assert!(!path.exists());
    drop(lease);
    fs::remove_file(directory.join(retention::LOCK)).unwrap();
    fs::remove_dir(directory).unwrap();
}
