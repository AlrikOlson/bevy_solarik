//! Convex support clipping in f64; keeps source footprint and terrain edges.
use bevy_math::DVec2;

pub(super) fn signed_area(points: &[DVec2]) -> f64 {
    let Some(first) = points.first() else {
        return 0.;
    };
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(a, b)| (*a - *first).perp_dot(*b - *first))
        .sum::<f64>()
        * 0.5
}
fn edge(points: &[DVec2], a: DVec2, b: DVec2, sign: f64) -> Vec<DVec2> {
    let mut result = Vec::new();
    let Some(last) = points.last() else {
        return result;
    };
    let mut previous = *last;
    let mut before = sign * (b - a).perp_dot(previous - a);
    for &current in points {
        let after = sign * (b - a).perp_dot(current - a);
        if (before >= 0.) != (after >= 0.) {
            result.push(previous + (current - previous) * (before / (before - after)));
        }
        if after >= 0. {
            result.push(current);
        }
        previous = current;
        before = after;
    }
    result
}
pub(super) fn intersect(subject: &[DVec2], clip: &[DVec2]) -> Vec<DVec2> {
    let sign = signed_area(clip).signum();
    let mut result = subject.to_vec();
    for (&a, &b) in clip
        .iter()
        .zip(clip.iter().cycle().skip(1))
        .take(clip.len())
    {
        result = edge(&result, a, b, sign);
        if result.is_empty() {
            break;
        }
    }
    result
}
pub(super) fn partition_valid(polygons: &[Vec<DVec2>], area: f64) -> bool {
    let total = polygons.iter().map(|p| signed_area(p).abs()).sum::<f64>();
    if (total / area - 1.).abs() > 1e-6 {
        return false;
    }
    for (i, a) in polygons.iter().enumerate() {
        for b in &polygons[..i] {
            if signed_area(&intersect(a, b)).abs() > area * 1e-8 {
                return false;
            }
        }
    }
    true
}
