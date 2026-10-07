//! Balanced spatial partitions for disconnected foliage.
use bevy_math::Vec3;
use meshopt::VertexDataAdapter;

pub(super) fn connected_partition(
    indices: &[u32],
    counts: &[u32],
    vertices: &VertexDataAdapter<'_>,
    capacity: usize,
) -> Vec<Vec<usize>> {
    if counts.is_empty() {
        return Vec::new();
    }
    if counts.len() <= capacity {
        return vec![(0..counts.len()).collect()];
    }
    // clusterlod.h (meshoptimizer c313abae): position-remapped indices preserve
    // connectivity, positions also group nearby disconnected leaf surfaces.
    let mut membership = vec![0; counts.len()];
    let count = meshopt::partition_clusters_with_positions(
        &mut membership,
        indices,
        counts,
        vertices,
        capacity,
    );
    let mut groups = vec![Vec::new(); count];
    for (id, group) in membership.into_iter().enumerate() {
        groups[group as usize].push(id);
    }
    groups
}

pub(super) fn partition(points: &[Vec3], capacity: usize) -> Vec<Vec<usize>> {
    assert!(capacity > 0);
    let mut ids: Vec<_> = (0..points.len()).collect();
    let mut result = Vec::new();
    split(points, &mut ids, capacity, &mut result);
    result
}

fn split(points: &[Vec3], ids: &mut [usize], capacity: usize, result: &mut Vec<Vec<usize>>) {
    if ids.is_empty() {
        return;
    }
    if ids.len() <= capacity {
        result.push(ids.to_vec());
        return;
    }
    let (mut low, mut high) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for &i in ids.iter() {
        low = low.min(points[i]);
        high = high.max(points[i]);
    }
    let extent = high - low;
    let axis = (0..3)
        .max_by(|&a, &b| extent[a].total_cmp(&extent[b]))
        .unwrap();
    ids.sort_unstable_by(|&a, &b| points[a][axis].total_cmp(&points[b][axis]).then(a.cmp(&b)));
    let middle = ids.len() / 2;
    let (left, right) = ids.split_at_mut(middle);
    split(points, left, capacity, result);
    split(points, right, capacity, result);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn connected_leaf_strips_stay_together_when_centroids_interleave() {
        // Two close parallel leaves, each split into eight connected pieces.
        // A centroid-only KD split cuts both leaves and locks their interiors.
        let positions: Vec<[f32; 3]> = (0..2)
            .flat_map(|leaf| {
                (0..9).flat_map(move |x| {
                    [
                        [x as f32, 0.0, leaf as f32 * 0.01],
                        [x as f32, 1.0, leaf as f32 * 0.01],
                    ]
                })
            })
            .collect();
        let indices: Vec<u32> = (0..8)
            .flat_map(|x| {
                (0..2).flat_map(move |leaf| {
                    let a = leaf * 18 + x * 2;
                    [a, a + 1, a + 2, a + 1, a + 3, a + 2]
                })
            })
            .collect();
        let vertices = VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0).unwrap();
        let groups = connected_partition(&indices, &[6; 16], &vertices, 8);
        assert_eq!(groups.len(), 2);
        assert!(
            groups
                .iter()
                .all(|g| g.len() == 8 && g.iter().all(|&i| i % 2 == g[0] % 2)),
            "connected leaf interiors were split: {groups:?}"
        );
        let mut membership: Vec<_> = groups.iter().flatten().copied().collect();
        membership.sort_unstable();
        assert_eq!(membership, (0..16).collect::<Vec<_>>());
        assert_eq!(
            groups,
            connected_partition(&indices, &[6; 16], &vertices, 8)
        );
    }
    #[test]
    fn interleaved_branches_partition_locally_with_exact_coverage() {
        let points: Vec<_> = (0..64)
            .map(|i| Vec3::new((i % 2) as f32 * 1000.0, (i / 2) as f32, 0.0))
            .collect();
        let groups = partition(&points, 8);
        assert!(groups.iter().all(|g| g.len() <= 8));
        for group in &groups {
            assert!(group.iter().all(|&i| points[i].x == points[group[0]].x));
        }
        let mut ids: Vec<_> = groups.into_iter().flatten().collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..64).collect::<Vec<_>>());
        assert_eq!(partition(&points, 8), partition(&points, 8));
    }
}
