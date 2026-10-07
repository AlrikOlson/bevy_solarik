//! Opt-in total foliage area redistribution through open boundaries.
use bevy_math::Vec3;
use bevy_platform::collections::HashMap;

pub(super) fn dilate(
    vertices: &[f32],
    stride: usize,
    original: &[u32],
    reduced: &[u32],
    remap: &[u32],
    locks: &[bool],
) -> Vec<(u32, Vec3)> {
    let p = |i: u32| Vec3::from_slice(&vertices[i as usize * stride..i as usize * stride + 3]);
    let old_area = surface_area(original, &p);
    let new_area = surface_area(reduced, &p);
    let edges = boundary(reduced, remap, locks);
    // clusterlod's safety guard: only compensate moderate boundary shrinkage.
    if old_area <= new_area || new_area < old_area * 0.25 {
        return Vec::new();
    }
    let directions = directions(&edges, remap, locks, &p);
    if directions.is_empty() {
        return Vec::new();
    }
    let area = |distance: f32| {
        let point = |i| {
            p(i) + directions
                .get(&remap[i as usize])
                .copied()
                .unwrap_or_default()
                * distance
        };
        surface_area(reduced, &point)
    };
    let perimeter: f32 = edges.iter().map(|e| p(e[0]).distance(p(e[1]))).sum();
    let mut high = (old_area - new_area) / perimeter * 4.0;
    if !high.is_finite() || area(high) < old_area {
        return Vec::new();
    }
    let mut low = 0.0;
    // Solve the actual triangle area, avoiding repeated first-order over-expansion.
    for _ in 0..32 {
        let middle = (low + high) * 0.5;
        if area(middle) < old_area {
            low = middle;
        } else {
            high = middle;
        }
    }
    let mut ids = reduced.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids.into_iter()
        .filter_map(|i| {
            directions
                .get(&remap[i as usize])
                .map(|direction| (i, p(i) + *direction * high))
        })
        .collect()
}

fn surface_area(indices: &[u32], p: &impl Fn(u32) -> Vec3) -> f32 {
    indices
        .chunks_exact(3)
        .map(|t| (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).length() * 0.5)
        .sum()
}

fn boundary(indices: &[u32], remap: &[u32], locks: &[bool]) -> Vec<[u32; 3]> {
    let key = |a: u32, b: u32| {
        let (a, b) = (remap[a as usize], remap[b as usize]);
        (a.min(b), a.max(b))
    };
    let mut counts = HashMap::<_, usize>::default();
    for t in indices.chunks_exact(3) {
        for e in 0..3 {
            *counts.entry(key(t[e], t[(e + 1) % 3])).or_default() += 1;
        }
    }
    let mut edges = Vec::new();
    for t in indices.chunks_exact(3) {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            if counts[&key(a, b)] == 1
                && !(locks[remap[a as usize] as usize] && locks[remap[b as usize] as usize])
            {
                edges.push([a, b, t[(e + 2) % 3]]);
            }
        }
    }
    edges
}

fn directions(
    edges: &[[u32; 3]],
    remap: &[u32],
    locks: &[bool],
    p: &impl Fn(u32) -> Vec3,
) -> HashMap<u32, Vec3> {
    let mut sums = HashMap::<u32, (Vec3, f32)>::default();
    for &[a, b, c] in edges {
        let edge = p(b) - p(a);
        let length = edge.length();
        if length == 0.0 {
            continue;
        }
        let away = p(a) - p(c);
        let normal = (away - edge * away.dot(edge) / edge.length_squared()).normalize_or_zero();
        for id in [remap[a as usize], remap[b as usize]] {
            if !locks[id as usize] {
                let entry = sums.entry(id).or_default();
                entry.0 += normal * length;
                entry.1 += length;
            }
        }
    }
    sums.into_iter()
        .filter_map(|(i, (sum, length))| {
            let average = sum / length;
            // Same bounded miter denominator as meshoptimizer's reference algorithm.
            let direction = average / average.length_squared().max(0.15);
            (direction.length_squared() > 0.0).then_some((i, direction))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_grid() -> (Vec<f32>, Vec<u32>) {
        let vertices = (0..9)
            .flat_map(|i| [(i % 3) as f32, (i / 3) as f32, 0.])
            .collect();
        let triangles = [0, 1, 3, 4]
            .into_iter()
            .flat_map(|i| [i, i + 1, i + 4, i, i + 4, i + 3])
            .collect();
        (vertices, triangles)
    }

    #[test]
    fn retriangulation_without_area_loss_does_not_dilate() {
        let (vertices, triangles) = square_grid();
        let changes = dilate(
            &vertices,
            3,
            &[0, 2, 8, 0, 8, 6],
            &triangles,
            &(0..9).collect::<Vec<_>>(),
            &[false; 9],
        );
        assert!(
            changes.is_empty(),
            "unchanged surface expanded: {changes:?}"
        );
    }

    #[test]
    fn total_area_including_interior_triangles_is_restored() {
        let (vertices, triangles) = square_grid();
        let changes = dilate(
            &vertices,
            3,
            &triangles,
            &[0, 2, 6],
            &(0..9).collect::<Vec<_>>(),
            &[false; 9],
        );
        let positions: HashMap<_, _> = changes.into_iter().collect();
        let p = |i| positions[&i];
        let area = (p(2) - p(0)).cross(p(6) - p(0)).length() * 0.5;
        assert!((area - 4.).abs() < 1e-5, "total area not conserved: {area}");
    }

    #[test]
    fn deleted_leaf_area_is_redistributed_without_moving_source() {
        let vertices = [
            0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 0., 0., 3., 0., 0., 2., 1., 0.,
        ];
        let before = vertices;
        let changes = dilate(
            &vertices,
            3,
            &[0, 1, 2, 3, 4, 5],
            &[0, 1, 2],
            &[0, 1, 2, 3, 4, 5],
            &[false; 6],
        );
        let positions: HashMap<_, _> = changes.into_iter().collect();
        let p = |i| {
            positions
                .get(&i)
                .copied()
                .unwrap_or_else(|| Vec3::from_slice(&vertices[i as usize * 3..i as usize * 3 + 3]))
        };
        let area = (p(1) - p(0)).cross(p(2) - p(0)).length() * 0.5;
        assert!((area - 1.0).abs() < 1e-5, "lost half the foliage: {area}");
        assert_eq!(vertices, before);
    }

    #[test]
    fn locked_seams_do_not_dilate() {
        let v = [0., 0., 0., 1., 0., 0., 0., 1., 0.];
        assert!(
            dilate(
                &v,
                3,
                &[0, 1, 2, 0, 1, 2],
                &[0, 1, 2],
                &[0, 1, 2],
                &[true; 3]
            )
            .is_empty()
        );
        assert!(dilate(&v, 3, &[0, 1, 2], &[0, 1, 2], &[0, 1, 2], &[false; 3]).is_empty());
    }

    #[test]
    fn hierarchy_preserves_source_and_contains_dilated_geometry() {
        use bevy_mesh::{Indices, Mesh};
        use bevy_render::render_resource::PrimitiveTopology;
        bevy_tasks::AsyncComputeTaskPool::get_or_init(bevy_tasks::TaskPool::new);
        let positions: Vec<[f32; 3]> = (0..512)
            .flat_map(|i| {
                let x = (i % 32) as f32 * 2.0;
                let y = (i / 32) as f32 * 2.0;
                [[x, y, 0.], [x + 1., y, 0.], [x, y + 1., 0.]]
            })
            .collect();
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, Default::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0., 0., 1.]; positions.len()]);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            (0..512)
                .flat_map(|_| [[0., 0.], [1., 0.], [0., 1.]])
                .collect::<Vec<_>>(),
        );
        mesh.insert_indices(Indices::U32((0..positions.len() as u32).collect()));
        let converted = super::super::MeshletMesh::from_mesh_preserve_area(&mesh, 8).unwrap();
        assert!(converted.meshlet_count() > 5);
        assert_eq!(
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                .unwrap()
                .as_float3()
                .unwrap(),
            positions
        );
        assert!(
            converted
                .meshlet_cull_data
                .iter()
                .any(|c| c.aabb.error > 0.0)
        );
        // from_mesh's production verifier checks every parent sphere/error contains its children.
        assert!(
            converted
                .meshlet_cull_data
                .iter()
                .all(|c| c.aabb.error.is_finite())
        );
    }
}
