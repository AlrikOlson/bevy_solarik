//! Optional displayed terrain support, independent of immutable source optical frames.
use super::{
    Bounds, CELL_ROWS, CellInput, PreparedRegion, REGION_ALLOCATION_BYTES, build_node, footprint,
    frame, height_range,
};
use bevy_math::{DVec2, DVec3, DVec4, Vec3};
use std::collections::{BTreeMap, BTreeSet};
#[path = "filtered_canopy_ground_clip.rs"]
mod clip;
#[cfg(feature = "graphics_debug")]
#[path = "filtered_canopy_ground_diagnostic.rs"]
mod diagnostic;
#[path = "filtered_canopy_ground_index.rs"]
mod index;
#[cfg(feature = "graphics_debug")]
pub use diagnostic::GroundFailure;

const MAX_REFERENCES: usize = 32;
const TRIANGLE_ROWS: usize = 3;

#[derive(Clone, Debug)]
pub struct GroundTriangle {
    pub vertices: [Vec3; 3],
    pub identity: u64,
}
#[derive(Clone, Debug)]
pub struct CellGround {
    pub identity: u64,
    pub triangles: Vec<u32>,
}
struct Triangle {
    lateral: [DVec2; 3],
    plane: DVec4,
    anchor_height: f64,
    identity: u64,
}
impl PreparedRegion {
    /// Derive compact ground references from an indexed displayed surface.
    pub fn with_ground_surface(&self, triangles: &[GroundTriangle]) -> Result<Self, &'static str> {
        index::prepare(self, triangles)
    }
    /// Refit display bounds and append shared ground, preserving original source rows.
    /// Each nonempty cell requires a complete nonoverlapping support partition.
    pub fn with_ground(
        &self,
        triangles: &[GroundTriangle],
        links: &[CellGround],
    ) -> Result<Self, &'static str> {
        if links.len() != self.cells {
            return Err("missing canopy ground links");
        }
        let links = link_map(links)?;
        let triangles = prepare_triangles(triangles)?;
        let mut items = Vec::with_capacity(self.cells);
        for cell in self.cell_inputs() {
            let link = links
                .get(&cell.identity)
                .ok_or("missing canopy cell ground")?;
            let bounds = supported_bounds(&cell, self.source_extents, link, &triangles)?;
            items.push((cell, bounds));
        }
        pack(items, self.source_extents, &triangles, &links)
    }
}
fn link_map(links: &[CellGround]) -> Result<BTreeMap<u64, &CellGround>, &'static str> {
    let mut map = BTreeMap::new();
    for link in links {
        if map.insert(link.identity, link).is_some() {
            return Err("duplicate ground cell");
        }
        if link.triangles.is_empty() || link.triangles.len() > MAX_REFERENCES {
            return Err("invalid canopy ground reference count");
        }
        let unique: BTreeSet<_> = link.triangles.iter().collect();
        if unique.len() != link.triangles.len() {
            return Err("duplicate ground reference");
        }
    }
    Ok(map)
}
fn prepare_triangles(values: &[GroundTriangle]) -> Result<Vec<Triangle>, &'static str> {
    if values.len() > REGION_ALLOCATION_BYTES / (TRIANGLE_ROWS * 16) {
        return Err("too many canopy ground triangles");
    }
    let mut identities = BTreeSet::new();
    values
        .iter()
        .map(|value| {
            if !identities.insert(value.identity) {
                return Err("duplicate ground identity");
            }
            triangle(value)
        })
        .collect()
}
fn triangle(value: &GroundTriangle) -> Result<Triangle, &'static str> {
    if value.vertices.iter().any(|p| !p.is_finite()) {
        return Err("nonfinite ground");
    }
    let [a, b, c] = value.vertices.map(Vec3::as_dvec3);
    let mut normal = (b - a).cross(c - a).normalize_or_zero();
    if normal.y < 0. {
        normal = -normal;
    }
    if normal.y < 0.05 {
        return Err("ground is not a supported height surface");
    }
    Ok(Triangle {
        lateral: [a, b, c].map(|p| DVec2::new(p.x, p.z)),
        plane: normal.extend(-normal.dot(a)),
        anchor_height: a.y,
        identity: value.identity,
    })
}
fn supported_bounds(
    cell: &CellInput,
    extents: [[f32; 2]; 2],
    link: &CellGround,
    triangles: &[Triangle],
) -> Result<Bounds, &'static str> {
    let up = frame(&cell.cell)?.y_axis.as_dvec3();
    let polygon = footprint(&cell.cell)?.map(|p| p.as_dvec2());
    let heights = height_range(&cell.cell, extents)?.ok_or("empty linked canopy cell")?;
    let mut bounds = None;
    let mut polygons = Vec::with_capacity(link.triangles.len());
    for &index in &link.triangles {
        let triangle = triangles
            .get(index as usize)
            .ok_or("invalid ground reference")?;
        let clipped = clip::intersect(&polygon, &triangle.lateral);
        if clip::signed_area(&clipped).abs() == 0. {
            return Err("empty ground reference");
        }
        let next = piece_bounds(&clipped, up, heights, triangle.plane)?;
        bounds = Some(bounds.map_or(next, |b: Bounds| b.union(next)));
        polygons.push(clipped);
    }
    if !clip::partition_valid(&polygons, clip::signed_area(&polygon).abs()) {
        return Err("ground does not partition the source footprint");
    }
    bounds.ok_or("missing ground bounds")
}
fn piece_bounds(
    polygon: &[DVec2],
    up: DVec3,
    heights: [f32; 2],
    plane: DVec4,
) -> Result<Bounds, &'static str> {
    let projection = plane.truncate().dot(up);
    if projection < 0.05 {
        return Err("ground opposes the source optical frame");
    }
    let mut low = DVec3::splat(f64::INFINITY);
    let mut high = DVec3::splat(f64::NEG_INFINITY);
    for p in polygon {
        for height in heights {
            let y = (f64::from(height) * projection - plane.w - plane.x * p.x - plane.z * p.y)
                / plane.y;
            let point = DVec3::new(p.x, y, p.y);
            low = low.min(point);
            high = high.max(point);
        }
    }
    padded_bounds(low, high)
}
fn padded_bounds(low: DVec3, high: DVec3) -> Result<Bounds, &'static str> {
    let low = low.as_vec3();
    let high = high.as_vec3();
    let padding = low.abs().max(high.abs()) * (8. * f32::EPSILON) + Vec3::splat(0.001);
    let bounds = Bounds {
        low: low - padding,
        high: high + padding,
    };
    if !bounds.low.is_finite() || !bounds.high.is_finite() {
        return Err("canopy ground bounds overflow");
    }
    Ok(bounds)
}
fn triangle_rows(triangle: &Triangle) -> [[u32; 4]; TRIANGLE_ROWS] {
    let [a, b, c] = triangle.lateral.map(|p| p.as_vec2());
    [
        triangle
            .plane
            .truncate()
            .as_vec3()
            .extend(triangle.anchor_height as f32)
            .to_array()
            .map(f32::to_bits),
        [a.x, a.y, b.x, b.y].map(f32::to_bits),
        [
            c.x.to_bits(),
            c.y.to_bits(),
            triangle.identity as u32,
            (triangle.identity >> 32) as u32,
        ],
    ]
}
fn pack_cells(rows: &mut Vec<[u32; 4]>, cells: &[CellInput]) {
    for cell in cells {
        rows.extend(cell.cell.map(|r| r.map(f32::to_bits)));
        rows.push([cell.identity as u32, (cell.identity >> 32) as u32, 0, 0]);
    }
}
fn pack_links(
    rows: &mut Vec<[u32; 4]>,
    cells: &[CellInput],
    triangle_base: usize,
    links: &BTreeMap<u64, &CellGround>,
) {
    for (i, cell) in cells.iter().enumerate() {
        let link = links[&cell.identity];
        let address = rows[0][2] as usize + i * CELL_ROWS + 7;
        rows[address][2] = rows.len() as u32;
        rows[address][3] = link.triangles.len() as u32;
        for chunk in link.triangles.chunks(4) {
            let mut row = [0; 4];
            for (slot, &index) in row.iter_mut().zip(chunk) {
                *slot = (triangle_base + index as usize * TRIANGLE_ROWS) as u32;
            }
            rows.push(row);
        }
    }
}
fn pack(
    mut items: Vec<(CellInput, Bounds)>,
    source_extents: [[f32; 2]; 2],
    triangles: &[Triangle],
    links: &BTreeMap<u64, &CellGround>,
) -> Result<PreparedRegion, &'static str> {
    let bounds = items
        .iter()
        .map(|(_, b)| *b)
        .reduce(Bounds::union)
        .unwrap_or(Bounds {
            low: Vec3::ZERO,
            high: Vec3::ZERO,
        });
    let diameter = items
        .iter()
        .map(|(_, b)| (b.high - b.low).length())
        .fold(0., f32::max);
    let (mut nodes, mut cells) = (Vec::new(), Vec::new());
    if !items.is_empty() {
        build_node(&mut items, &mut nodes, &mut cells);
    }
    let mut rows = vec![[
        nodes.len() as u32 / 2,
        1,
        nodes.len() as u32 + 1,
        cells.len() as u32,
    ]];
    rows.extend(nodes);
    pack_cells(&mut rows, &cells);
    let triangle_base = rows.len();
    for triangle in triangles {
        rows.extend(triangle_rows(triangle));
    }
    pack_links(&mut rows, &cells, triangle_base, links);
    if rows.len() * 16 > REGION_ALLOCATION_BYTES {
        return Err("ground exceeds region slot");
    }
    if !diameter.is_finite() {
        return Err("ground diameter overflow");
    }
    Ok(PreparedRegion {
        rows,
        source_extents,
        minimum: bounds.low,
        maximum: bounds.high,
        cells: cells.len(),
        maximum_cell_diameter: diameter,
    })
}
