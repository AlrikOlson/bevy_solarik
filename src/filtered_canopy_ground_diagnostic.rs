//! Bounded exact input for replaying the first rejected support cell.
use super::*;

#[derive(Clone, Debug)]
pub struct GroundFailure {
    pub reason: &'static str,
    pub cell: CellInput,
    pub source_extents: [[f32; 2]; 2],
    pub triangles: Vec<GroundTriangle>,
    pub footprint_area: f64,
    pub covered_area: f64,
    pub maximum_overlap: f64,
}
impl PreparedRegion {
    /// Does not accept, modify, or bypass the production support validation.
    pub fn first_ground_failure(
        &self,
        triangles: &[GroundTriangle],
    ) -> Result<Option<GroundFailure>, &'static str> {
        let links = index::links(self, triangles)?;
        let prepared = prepare_triangles(triangles)?;
        for (cell, link) in self.cell_inputs().zip(&links) {
            let error = if link.triangles.is_empty() {
                Some("invalid canopy ground reference count")
            } else {
                supported_bounds(&cell, self.source_extents, link, &prepared).err()
            };
            if let Some(reason) = error {
                return report(
                    reason,
                    cell,
                    self.source_extents,
                    link,
                    triangles,
                    &prepared,
                )
                .map(Some);
            }
        }
        Ok(None)
    }
}
fn maximum_overlap(polygons: &[Vec<DVec2>]) -> f64 {
    let mut maximum = 0_f64;
    for (i, a) in polygons.iter().enumerate() {
        for b in &polygons[..i] {
            maximum = maximum.max(clip::signed_area(&clip::intersect(a, b)).abs());
        }
    }
    maximum
}
fn report(
    reason: &'static str,
    cell: CellInput,
    source_extents: [[f32; 2]; 2],
    link: &CellGround,
    original: &[GroundTriangle],
    prepared: &[Triangle],
) -> Result<GroundFailure, &'static str> {
    let polygon = footprint(&cell.cell)?.map(bevy_math::Vec2::as_dvec2);
    let polygons: Vec<_> = link
        .triangles
        .iter()
        .map(|&i| clip::intersect(&polygon, &prepared[i as usize].lateral))
        .collect();
    Ok(GroundFailure {
        reason,
        cell,
        source_extents,
        triangles: link
            .triangles
            .iter()
            .map(|&i| original[i as usize].clone())
            .collect(),
        footprint_area: clip::signed_area(&polygon).abs(),
        covered_area: polygons.iter().map(|p| clip::signed_area(p).abs()).sum(),
        maximum_overlap: maximum_overlap(&polygons),
    })
}
