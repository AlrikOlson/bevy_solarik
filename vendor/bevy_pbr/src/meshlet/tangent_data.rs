//! Optional authored `MikkTSpace` payload validation, before meshlet conversion.
use super::MeshToMeshletMeshConversionError;
use bevy_math::Vec4;
use bevy_mesh::{Mesh, VertexAttributeValues};

pub(super) fn source_tangents(
    mesh: &Mesh,
) -> Result<Option<&[[f32; 4]]>, MeshToMeshletMeshConversionError> {
    let Some(attribute) = mesh.attribute(Mesh::ATTRIBUTE_TANGENT) else {
        return Ok(None);
    };
    let VertexAttributeValues::Float32x4(values) = attribute else {
        return Err(MeshToMeshletMeshConversionError::InvalidVertexTangents);
    };
    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .map_or(0, VertexAttributeValues::len);
    if values.len() != positions
        || values
            .iter()
            .any(|t| !super::super::asset::valid_authored_tangent(Vec4::from_array(*t)))
    {
        return Err(MeshToMeshletMeshConversionError::InvalidVertexTangents);
    }
    Ok(Some(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::primitives::Cuboid;

    #[test]
    fn tangentless_mesh_retains_legacy_reconstruction() {
        assert!(
            source_tangents(&Mesh::from(Cuboid::default()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn original_vertex_tangents_and_handedness_are_preserved() {
        let mut mesh = Mesh::from(Cuboid::default());
        mesh.generate_tangents().unwrap();
        let values = source_tangents(&mesh).unwrap().unwrap();
        assert_eq!(values.len(), mesh.count_vertices());
        assert!(values.iter().all(|t| t[3].abs() == 1.0));
    }

    #[test]
    fn malformed_tangent_lengths_and_values_are_rejected() {
        let mut mesh = Mesh::from(Cuboid::default());
        for values in [
            vec![[1.0, 0.0, 0.0, 1.0]],
            vec![[f32::NAN, 0.0, 0.0, 1.0]; mesh.count_vertices()],
            vec![[2.0, 0.0, 0.0, 1.0]; mesh.count_vertices()],
            vec![[1.0, 0.0, 0.0, 0.0]; mesh.count_vertices()],
        ] {
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, values);
            assert!(source_tangents(&mesh).is_err());
        }
    }

    #[test]
    fn meshlet_remapping_keeps_tangent_paired_with_its_original_vertex() {
        bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::new);
        let mut mesh = Mesh::from(Cuboid::default());
        let uvs: Vec<[f32; 2]> = (0..mesh.count_vertices())
            .map(|i| [i as f32, 0.0])
            .collect();
        let tangents: Vec<[f32; 4]> = (0..mesh.count_vertices()).map(tagged_tangent).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents.clone());
        let prepared = super::super::MeshletMesh::from_mesh(&mesh, 4).unwrap();
        assert_eq!(
            prepared.vertex_tangent_indices.len(),
            prepared.vertex_uvs.len()
        );
        for (uv, index) in prepared
            .vertex_uvs
            .iter()
            .zip(prepared.vertex_tangent_indices.iter())
        {
            assert_eq!(
                prepared.vertex_tangents[*index as usize],
                Vec4::from_array(tangents[uv.x as usize])
            );
        }
    }

    fn tagged_tangent(index: usize) -> [f32; 4] {
        let angle = index as f32 * 0.1;
        [
            angle.cos(),
            angle.sin(),
            0.0,
            if index.is_multiple_of(2) { 1.0 } else { -1.0 },
        ]
    }
}
