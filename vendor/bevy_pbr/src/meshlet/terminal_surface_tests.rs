use super::*;

fn finest_triangles(mesh: &MeshletMesh) -> Vec<[[u32; 10]; 3]> {
    let mut triangles = Vec::new();
    for (m, cull) in mesh.meshlets.iter().zip(mesh.meshlet_cull_data.iter()) {
        if cull.aabb.error != 0.0 {
            continue;
        }
        let widths = [
            m.bits_per_vertex_position_channel_x,
            m.bits_per_vertex_position_channel_y,
            m.bits_per_vertex_position_channel_z,
        ];
        let minimum = [
            m.min_vertex_position_channel_x,
            m.min_vertex_position_channel_y,
            m.min_vertex_position_channel_z,
        ];
        let mut cursor = m.start_vertex_position_bit as usize;
        let mut vertices = Vec::new();
        for i in 0..=usize::from(m.vertex_count_minus_one) {
            let mut v = [0u32; 10];
            for channel in 0..3 {
                let mut value = 0u32;
                for bit in 0..usize::from(widths[channel]) {
                    value |= ((mesh.vertex_positions[cursor / 32] >> (cursor % 32)) & 1) << bit;
                    cursor += 1;
                }
                v[channel] = (minimum[channel] + value as f32).to_bits();
            }
            let index = m.start_vertex_attribute_id as usize + i;
            v[3] = mesh.vertex_normals[index];
            v[4..6].copy_from_slice(&mesh.vertex_uvs[index].to_array().map(f32::to_bits));
            let tangent = mesh.vertex_tangents[mesh.vertex_tangent_indices[index] as usize];
            v[6..].copy_from_slice(&tangent.to_array().map(f32::to_bits));
            vertices.push(v);
        }
        let first = m.start_index_id as usize;
        for ids in mesh.indices[first..first + usize::from(m.triangle_count) * 3].chunks_exact(3) {
            let mut tri = [
                vertices[ids[0] as usize],
                vertices[ids[1] as usize],
                vertices[ids[2] as usize],
            ];
            let start = (0..3).min_by_key(|&i| tri[i]).unwrap();
            tri.rotate_left(start);
            triangles.push(tri);
        }
    }
    triangles.sort_unstable();
    triangles
}

#[test]
fn terminal_surface_preserves_finest_triangles_attributes_and_cached_bytes() {
    assert_eq!(env!("MESHLET_CACHE_COMPILER_KNOWN"), "1");
    for varied in [false, true] {
        let mut source = source();
        if varied {
            if let Some(VertexAttributeValues::Float32x3(points)) =
                source.attribute_mut(Mesh::ATTRIBUTE_POSITION)
            {
                for p in points {
                    p[2] = (p[0] * 1.7).sin() * 0.25 + p[1] * 0.13;
                    p[0] -= 1700.0;
                    p[1] += 2200.0;
                }
            }
        }
        let started = std::time::Instant::now();
        let full = MeshletMesh::from_mesh_preserve_area(&source, 4).unwrap();
        let full_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = std::time::Instant::now();
        let terminal = MeshletMesh::from_mesh_terminal(&source, 4).unwrap();
        let terminal_ms = started.elapsed().as_secs_f64() * 1000.0;
        let finest = finest_triangles(&terminal);
        assert_eq!(finest.len(), 512);
        assert_eq!(finest, finest_triangles(&full));
        assert!(
            terminal
                .meshlet_cull_data
                .iter()
                .all(|c| c.aabb.error == 0.0)
        );
        assert!(terminal.meshlet_count() <= full.meshlet_count());
        assert!(terminal.storage_bytes() <= full.storage_bytes());
        println!(
            "terminal varied={varied} full_ms={full_ms:.3} terminal_ms={terminal_ms:.3} full_bytes={} terminal_bytes={}",
            full.storage_bytes(),
            terminal.storage_bytes()
        );
        let directory = directory();
        let cold = MeshletMesh::from_mesh_terminal_cached(&source, 4, &directory).unwrap();
        assert_eq!(encode(&cold), encode(&terminal));
        let terminal_key = preprocessed_key(&source, 4, true, b"terminal-surface-v1");
        assert_ne!(terminal_key, key(&source, 4, true, RECIPE));
        let warm = cached_key::<MeshToMeshletMeshConversionError>(terminal_key, &directory, || {
            panic!("terminal warm cache miss")
        })
        .unwrap();
        assert_eq!(encode(&warm), encode(&terminal));
        fs::remove_file(directory.join(format!("{}.mvg", terminal_key.to_hex()))).unwrap();
        fs::remove_file(directory.join(retention::LOCK)).unwrap();
        assert!(fs::read_dir(&directory).unwrap().next().is_none());
        fs::remove_dir(directory).unwrap();
    }
}
