//! Immutable sparse prototype data shared by coarse raster/ray consumers.
//! Packing is lossless; this module does not infer a scattering or opacity model.
use crate::coarse_cache::{self, DIRECTIONS, Directional};
use bevy_render::{
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};
use bytemuck::{Pod, Zeroable};
#[path = "coarse_scene_pack.rs"]
mod packing;
pub use packing::PackedDirectional;
pub const SHADER: &str = include_str!("coarse_scene.wgsl");
pub const WALK_SHADER: &str = include_str!("coarse_walk.wgsl");
pub const MISSING: u32 = u32::MAX;
pub const MAX_GPU_BYTES: usize = 128 << 20;

/// Shader address frame. Local source-height units, half-open integer cells.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Grid {
    pub origin_resolution: [i32; 4],
    pub dimensions: [u32; 4],
    pub metadata: [u32; 4],
}
pub struct PackedSource {
    pub identity: [u8; 32],
    pub grid: Grid,
    pub lookup: Vec<u32>,
    pub measurements: Vec<PackedDirectional>,
}
pub struct GpuSource {
    pub grid: Buffer,
    pub lookup: Buffer,
    pub measurements: Buffer,
    pub bytes: u64,
}
pub(crate) fn load_shader(app: &mut bevy_app::App) {
    bevy_shader::load_shader_library!(app, "coarse_scene.wgsl");
    bevy_shader::load_shader_library!(app, "coarse_walk.wgsl");
    bevy_shader::load_shader_library!(app, "coarse_appearance.wgsl");
}
impl Grid {
    pub fn index(&self, point: [f32; 3]) -> Option<usize> {
        let relative: [f32; 3] = core::array::from_fn(|a| {
            (point[a] * self.origin_resolution[3] as f32).floor() - self.origin_resolution[a] as f32
        });
        if (0..3).any(|a| {
            !relative[a].is_finite() || relative[a] < 0. || relative[a] >= self.dimensions[a] as f32
        }) {
            return None;
        }
        Some(
            (relative[0] as usize * self.dimensions[1] as usize + relative[1] as usize)
                * self.dimensions[2] as usize
                + relative[2] as usize,
        )
    }
}
impl PackedSource {
    pub fn decode(bytes: &[u8], identity: [u8; 32]) -> Option<Self> {
        let source = coarse_cache::decode(bytes, identity)?;
        let low: [i32; 3] =
            core::array::from_fn(|a| source.cells.iter().map(|c| c.grid[a]).min().unwrap());
        let high: [i32; 3] =
            core::array::from_fn(|a| source.cells.iter().map(|c| c.grid[a]).max().unwrap());
        // Keep all grid arithmetic exactly representable in f32.
        if low.iter().chain(&high).any(|v| v.unsigned_abs() > 32768) {
            return None;
        }
        let dimensions: [u32; 3] = core::array::from_fn(|a| (high[a] - low[a] + 1) as u32);
        let count = dimensions
            .into_iter()
            .try_fold(1usize, |a, b| a.checked_mul(b as usize))?;
        let rows = source.cells.len().checked_mul(DIRECTIONS)?;
        let size = count
            .checked_mul(4)?
            .checked_add(rows.checked_mul(size_of::<PackedDirectional>())?)?
            .checked_add(size_of::<Grid>())?;
        if size > MAX_GPU_BYTES {
            return None;
        }
        let grid = Grid {
            origin_resolution: [low[0], low[1], low[2], source.resolution as i32],
            dimensions: [dimensions[0], dimensions[1], dimensions[2], 0],
            metadata: [
                source.cells.len() as u32,
                source.samples_side,
                source.prototype,
                1,
            ],
        };
        let mut lookup = vec![MISSING; count];
        let mut measurements = Vec::with_capacity(rows);
        for (i, cell) in source.cells.iter().enumerate() {
            let index = grid.index(cell.centre_half[..3].try_into().ok()?)?;
            if lookup[index] != MISSING {
                return None;
            }
            lookup[index] = i as u32;
            for bin in &cell.directions {
                measurements.push(PackedDirectional::new(bin, source.samples_side)?);
            }
        }
        Some(Self {
            identity,
            grid,
            lookup,
            measurements,
        })
    }
    pub fn sample(&self, point: [f32; 3], direction: usize) -> Option<Directional> {
        if direction >= DIRECTIONS {
            return None;
        }
        let cell = *self.lookup.get(self.grid.index(point)?)?;
        if cell == MISSING {
            return None;
        }
        self.measurements
            .get(cell as usize * DIRECTIONS + direction)
            .map(PackedDirectional::unpack)
    }
    pub fn bytes(&self) -> usize {
        size_of::<Grid>()
            + self.lookup.len() * 4
            + self.measurements.len() * size_of::<PackedDirectional>()
    }
    /// The caller owns submission/fence lifetime, via these immutable handles.
    pub fn upload(&self, device: &RenderDevice) -> GpuSource {
        let buffer = |label, contents: &[u8]| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: BufferUsages::STORAGE,
            })
        };
        GpuSource {
            grid: buffer("coarse source frame", bytemuck::bytes_of(&self.grid)),
            lookup: buffer(
                "coarse source indirection",
                bytemuck::cast_slice(&self.lookup),
            ),
            measurements: buffer(
                "coarse source observations",
                bytemuck::cast_slice(&self.measurements),
            ),
            bytes: self.bytes() as u64,
        }
    }
}
#[cfg(test)]
#[path = "coarse_scene_tests.rs"]
mod tests;
