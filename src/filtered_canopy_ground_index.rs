//! Bounded 2D broad phase; exact intersection and final partition checks remain separate.
use super::{
    CellGround, GroundTriangle, MAX_REFERENCES, PreparedRegion, Triangle, clip, footprint,
    prepare_triangles,
};
use bevy_math::DVec2;
use std::collections::BTreeMap;

struct Node {
    low: DVec2,
    high: DVec2,
    escape: usize,
    triangle: Option<usize>,
}
fn extent(triangle: &Triangle) -> (DVec2, DVec2) {
    triangle.lateral.iter().fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(low, high), &p| (low.min(p), high.max(p)),
    )
}
fn branch(indices: &mut [usize], triangles: &[Triangle], nodes: &mut Vec<Node>) {
    let (low, high) = indices.iter().map(|&i| extent(&triangles[i])).fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(low, high), (a, b)| (low.min(a), high.max(b)),
    );
    let start = nodes.len();
    nodes.push(Node {
        low,
        high,
        escape: 0,
        triangle: None,
    });
    if indices.len() == 1 {
        nodes[start].triangle = Some(indices[0]);
    } else {
        let axis = usize::from(high.y - low.y > high.x - low.x);
        indices.sort_unstable_by(|&a, &b| {
            let centre = |i: usize| triangles[i].lateral.iter().map(|p| p[axis]).sum::<f64>();
            centre(a)
                .total_cmp(&centre(b))
                .then(triangles[a].identity.cmp(&triangles[b].identity))
        });
        let middle = indices.len() / 2;
        let (left, right) = indices.split_at_mut(middle);
        branch(left, triangles, nodes);
        branch(right, triangles, nodes);
    }
    nodes[start].escape = nodes.len();
}
fn query(
    polygon: &[DVec2],
    nodes: &[Node],
    triangles: &[Triangle],
) -> Result<Vec<u32>, &'static str> {
    let (low, high) = polygon.iter().fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(low, high), &p| (low.min(p), high.max(p)),
    );
    let mut found = Vec::new();
    let mut at = 0;
    while at < nodes.len() {
        let node = &nodes[at];
        if node.high.cmplt(low).any() || node.low.cmpgt(high).any() {
            at = node.escape;
            continue;
        }
        at += 1;
        let Some(index) = node.triangle else {
            continue;
        };
        if clip::signed_area(&clip::intersect(polygon, &triangles[index].lateral)).abs() == 0. {
            continue;
        }
        if found.len() == MAX_REFERENCES {
            return Err("too many canopy ground intersections");
        }
        found.push(index as u32);
    }
    Ok(found)
}
pub(super) fn links(
    region: &PreparedRegion,
    triangles: &[GroundTriangle],
) -> Result<Vec<CellGround>, &'static str> {
    let prepared = prepare_triangles(triangles)?;
    let mut nodes = Vec::with_capacity(prepared.len() * 2);
    let mut indices: Vec<_> = (0..prepared.len()).collect();
    if !indices.is_empty() {
        branch(&mut indices, &prepared, &mut nodes);
    }
    region
        .cell_inputs()
        .map(|cell| {
            let polygon = footprint(&cell.cell)?.map(|p| p.as_dvec2());
            Ok(CellGround {
                identity: cell.identity,
                triangles: query(&polygon, &nodes, &prepared)?,
            })
        })
        .collect()
}
fn compact(triangles: &[GroundTriangle], links: &mut [CellGround]) -> Vec<GroundTriangle> {
    let used: BTreeMap<_, _> = links
        .iter()
        .flat_map(|c| &c.triangles)
        .map(|&i| (triangles[i as usize].identity, i))
        .collect();
    let mut remap = vec![0; triangles.len()];
    let selected = used
        .values()
        .enumerate()
        .map(|(new, &old)| {
            remap[old as usize] = new as u32;
            triangles[old as usize].clone()
        })
        .collect();
    for cell in links {
        for index in &mut cell.triangles {
            *index = remap[*index as usize];
        }
        cell.triangles.sort_unstable();
    }
    selected
}
pub(super) fn prepare(
    region: &PreparedRegion,
    triangles: &[GroundTriangle],
) -> Result<PreparedRegion, &'static str> {
    let mut links = links(region, triangles)?;
    let compact = compact(triangles, &mut links);
    region.with_ground(&compact, &links)
}
