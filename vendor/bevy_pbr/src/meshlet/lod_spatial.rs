//! Balanced spatial partitions for disconnected foliage.
use bevy_math::Vec3;

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
