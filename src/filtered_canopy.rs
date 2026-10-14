//! Immutable finite canopy cells and a bounded stackless BVH.
//! This display model does not own or change authoritative plant identities.
use alloc::collections::BTreeSet;
use bevy_math::{Mat3, Vec2, Vec3};
#[path = "filtered_canopy_profile.rs"]
mod profile;
pub use profile::{CanopySources, SourceProfile};
#[path = "filtered_canopy_arena.rs"]
mod arena;
#[path = "filtered_canopy_asset.rs"]
mod asset;
pub use asset::{Canopy3d, CanopyRayMask, CanopyRegion};
#[path = "filtered_canopy_scene.rs"]
mod scene;
#[cfg(feature = "graphics_debug")]
pub use scene::diagnostic::{DiagnosticInstance, diagnostic_scene_rows};
pub(crate) use scene::{CanopyGpu, PrepareCanopy};
#[path = "filtered_canopy_plugin.rs"]
mod plugin;
#[path = "filtered_canopy_prepass.rs"]
mod prepass;
pub use plugin::CanopyPlugin;
pub const TRANSPORT_SHADER: &str = include_str!("filtered_canopy_transport.wesl");
pub const REGION_SHADER: &str = include_str!("filtered_canopy_region.wesl");
pub const MAX_CELLS: usize = 16_384;
/// Fixed slots make producer admission sufficient even after arbitrary retirement.
pub const REGION_ALLOCATION_BYTES: usize = 4 << 20;
pub const MAX_ARENA_BYTES: usize = 512 << 20;
pub const SOURCE_DIRECTORY: &str = env!("CARGO_MANIFEST_DIR");
pub const CELL_ROWS: usize = 8;
#[path = "filtered_canopy_ground.rs"]
mod ground;
#[cfg(feature = "graphics_debug")]
pub use ground::GroundFailure;
pub use ground::{CellGround, GroundTriangle};
pub type Cell = [[f32; 4]; 7];
#[derive(Clone, Debug)]
pub struct CellInput {
    pub cell: Cell,
    /// Stable logical identity supplied by the producer, never an allocation slot.
    pub identity: u64,
}
#[derive(Clone, Debug)]
pub struct PreparedRegion {
    rows: Vec<[u32; 4]>,
    source_extents: [[f32; 2]; 2],
    pub minimum: Vec3,
    pub maximum: Vec3,
    pub cells: usize,
    pub maximum_cell_diameter: f32,
}
#[derive(Clone, Copy)]
struct Bounds {
    low: Vec3,
    high: Vec3,
}
impl Bounds {
    fn union(self, other: Self) -> Self {
        Self {
            low: self.low.min(other.low),
            high: self.high.max(other.high),
        }
    }
}
impl PreparedRegion {
    /// The two source extents are the normalized physical height bounds of the
    /// SAME source profile used by the GPU. Cell heights scale these extents.
    pub fn build(
        cells: Vec<CellInput>,
        source_extents: [[f32; 2]; 2],
    ) -> Result<Self, &'static str> {
        validate_sources(source_extents)?;
        if cells.len() > MAX_CELLS {
            return Err("too many finite canopy cells");
        }
        let mut identities = BTreeSet::new();
        let mut items = Vec::with_capacity(cells.len());
        for cell in cells {
            if !identities.insert(cell.identity) {
                return Err("duplicate canopy cell identity");
            }
            if let Some(bounds) = cell_bounds(&cell.cell, source_extents)? {
                items.push((cell, bounds));
            }
        }
        let mut nodes = Vec::new();
        let mut packed_cells = Vec::new();
        if !items.is_empty() {
            build_node(&mut items, &mut nodes, &mut packed_cells);
        }
        let bounds = items
            .iter()
            .map(|(_, b)| *b)
            .reduce(Bounds::union)
            .unwrap_or(Bounds {
                low: Vec3::ZERO,
                high: Vec3::ZERO,
            });
        let count = packed_cells.len();
        let maximum_cell_diameter = items
            .iter()
            .map(|(_, b)| (b.high - b.low).length())
            .fold(0., f32::max);
        if !maximum_cell_diameter.is_finite() {
            return Err("canopy cell diameter overflow");
        }
        let mut rows = vec![[
            nodes.len() as u32 / 2,
            1,
            nodes.len() as u32 + 1,
            count as u32,
        ]];
        rows.extend(nodes);
        for input in packed_cells {
            rows.extend(input.cell.map(|r| r.map(f32::to_bits)));
            rows.push([input.identity as u32, (input.identity >> 32) as u32, 0, 0]);
        }
        Ok(Self {
            rows,
            source_extents,
            minimum: bounds.low,
            maximum: bounds.high,
            cells: count,
            maximum_cell_diameter,
        })
    }
    pub fn rows(&self) -> &[[u32; 4]] {
        &self.rows
    }
    pub fn bytes(&self) -> usize {
        self.rows.len() * 16
    }
    pub fn cell_inputs(&self) -> impl Iterator<Item = CellInput> + '_ {
        let start = self.rows[0][2] as usize;
        self.rows[start..start + self.cells * CELL_ROWS]
            .as_chunks::<CELL_ROWS>()
            .0
            .iter()
            .map(|rows| CellInput {
                cell: core::array::from_fn(|i| rows[i].map(f32::from_bits)),
                identity: u64::from(rows[7][0]) | (u64::from(rows[7][1]) << 32),
            })
    }
}
fn validate_sources(extents: [[f32; 2]; 2]) -> Result<(), &'static str> {
    if extents
        .iter()
        .all(|v| v[0].is_finite() && v[1].is_finite() && v[0] < v[1])
    {
        Ok(())
    } else {
        Err("invalid canopy source extents")
    }
}
fn frame(cell: &Cell) -> Result<Mat3, &'static str> {
    if cell.iter().flatten().any(|x| !x.is_finite()) {
        return Err("nonfinite canopy cell");
    }
    let axes = [0, 1, 2].map(|i| Vec3::from_slice(&cell[i][..3]));
    let frame = Mat3::from_cols(axes[0], axes[1], axes[2]);
    let error = frame.transpose() * frame - Mat3::IDENTITY;
    if error.to_cols_array().iter().any(|v| v.abs() > 1e-4)
        || (frame.determinant() - 1.).abs() > 1e-4
        || axes[1].y < 0.05
    {
        return Err("invalid canopy source frame");
    }
    if cell[0][3] != 0. || cell[2][3] != 0. {
        return Err("nonzero canopy frame padding");
    }
    Ok(frame)
}
fn footprint(cell: &Cell) -> Result<[Vec2; 4], &'static str> {
    let p = [
        Vec2::new(cell[3][0], cell[3][1]),
        Vec2::new(cell[3][2], cell[3][3]),
        Vec2::new(cell[4][0], cell[4][1]),
        Vec2::new(cell[4][2], cell[4][3]),
    ];
    let turns = core::array::from_fn::<_, 4, _>(|i| {
        (p[(i + 1) % 4] - p[i]).perp_dot(p[(i + 2) % 4] - p[(i + 1) % 4])
    });
    if turns.iter().all(|v| v.is_finite())
        && (turns.iter().all(|v| *v > 0.) || turns.iter().all(|v| *v < 0.))
    {
        Ok(p)
    } else {
        Err("canopy support must be a convex nondegenerate quadrilateral")
    }
}
fn height_range(cell: &Cell, extents: [[f32; 2]; 2]) -> Result<Option<[f32; 2]>, &'static str> {
    let mut range = [f32::INFINITY, f32::NEG_INFINITY];
    for (p, extent) in cell[5..].iter().zip(extents) {
        if p[0] < 0. || p[1] < 0. || p[2] != 0. || p[3] != 0. {
            return Err("invalid canopy population moments");
        }
        if p[1] == 0. {
            continue;
        }
        if p[0] == 0. {
            return Err("nonempty canopy has zero mean height");
        }
        range[0] = range[0].min(p[0] * extent[0]);
        range[1] = range[1].max(p[0] * extent[1]);
    }
    if range[0] == f32::INFINITY {
        return Ok(None);
    }
    if !range.iter().all(|x| x.is_finite()) {
        return Err("canopy height overflow");
    }
    Ok(Some(range))
}
fn cell_bounds(cell: &Cell, extents: [[f32; 2]; 2]) -> Result<Option<Bounds>, &'static str> {
    let up = frame(cell)?.y_axis;
    let points = footprint(cell)?;
    let Some(range) = height_range(cell, extents)? else {
        return Ok(None);
    };
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for p in points {
        for h in range {
            let y = (h - cell[1][3] - up.x * p.x - up.z * p.y) / up.y;
            let point = Vec3::new(p.x, y, p.y);
            low = low.min(point);
            high = high.max(point);
        }
    }
    if !low.is_finite() || !high.is_finite() {
        return Err("canopy bounds overflow");
    }
    let padding = low.abs().max(high.abs()) * (8. * f32::EPSILON) + Vec3::splat(0.001);
    let bounds = Bounds {
        low: low - padding,
        high: high + padding,
    };
    if !bounds.low.is_finite() || !bounds.high.is_finite() {
        return Err("canopy padded bounds overflow");
    }
    Ok(Some(bounds))
}
fn build_node(
    items: &mut [(CellInput, Bounds)],
    nodes: &mut Vec<[u32; 4]>,
    cells: &mut Vec<CellInput>,
) {
    let bounds = items
        .iter()
        .map(|(_, b)| *b)
        .reduce(Bounds::union)
        .expect("nonempty BVH node");
    let start = nodes.len();
    nodes.extend([[0; 4]; 2]);
    let mut leaf = u32::MAX;
    if items.len() == 1 {
        leaf = cells.len() as u32;
        cells.push(items[0].0.clone());
    } else {
        let extent = bounds.high - bounds.low;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        items.sort_unstable_by(|(a, ba), (b, bb)| {
            (ba.low[axis] + ba.high[axis])
                .total_cmp(&(bb.low[axis] + bb.high[axis]))
                .then_with(|| a.identity.cmp(&b.identity))
        });
        let middle = items.len() / 2;
        let (left, right) = items.split_at_mut(middle);
        build_node(left, nodes, cells);
        build_node(right, nodes, cells);
    }
    nodes[start] = [
        bounds.low.x.to_bits(),
        bounds.low.y.to_bits(),
        bounds.low.z.to_bits(),
        nodes.len() as u32 / 2,
    ];
    nodes[start + 1] = [
        bounds.high.x.to_bits(),
        bounds.high.y.to_bits(),
        bounds.high.z.to_bits(),
        leaf,
    ];
}
#[cfg(test)]
#[path = "filtered_canopy_tests.rs"]
mod tests;
