//! Sparse source-derived directional measurements for coarse geometry.
//! Measurements are finite footprint samples, not proof of empty space or a
//! fitted scattering law. Cells with zero sampled hits remain represented.
#[path = "coarse_cache_codec.rs"]
mod codec;
pub use codec::{decode, encode};
pub const DIRECTIONS: usize = 14;
pub const MATERIALS: usize = 8;
pub const MAX_CACHE_BYTES: usize = 128 << 20;

#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Directional {
    /// Rays intersecting cell, covered rays, accepted surface events, overflow.
    pub counts: [u32; 4],
    /// Minimum first, maximum last, sum first, sum last: normalized cell depth.
    pub depth: [f32; 4],
    /// Visible first-hit geometric-normal second moments xx, yy, zz; reserved.
    pub diagonal: [f32; 4],
    /// Visible first-hit geometric-normal second moments xy, xz, yz; reserved.
    pub cross: [f32; 4],
    /// Sum of visible first-hit geometric normals; reserved.
    pub normal: [f32; 4],
    /// First-hit counts for each original material/part, without merging.
    pub materials: [u32; MATERIALS],
}
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Cell {
    /// Source-height-normalized centre xyz and cubic half extent.
    pub centre_half: [f32; 4],
    /// Integer xyz cell address and cells per source height.
    pub grid: [i32; 4],
    pub directions: [Directional; DIRECTIONS],
}
pub struct SourceCache {
    pub identity: [u8; 32],
    pub prototype: u32,
    pub resolution: u32,
    pub samples_side: u32,
    pub cells: Vec<Cell>,
}

pub fn directions() -> [[f32; 3]; DIRECTIONS] {
    use bevy_math::Vec3;
    let basis = [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::ONE,
        Vec3::new(1., 1., -1.),
        Vec3::new(1., -1., 1.),
        Vec3::new(-1., 1., 1.),
    ];
    core::array::from_fn(|i| {
        (basis[i / 2].normalize() * if i % 2 == 0 { 1. } else { -1. }).to_array()
    })
}

impl Directional {
    pub fn validate(&self, samples_side: u32) -> bool {
        let [support, covered, events, overflow] = self.counts;
        if !(1..=16).contains(&samples_side)
            || overflow != 0
            || support > samples_side * samples_side
            || covered > support
            || events < covered
            || self.materials.iter().map(|&n| u64::from(n)).sum::<u64>() != u64::from(covered)
        {
            return false;
        }
        if self
            .depth
            .iter()
            .chain(&self.diagonal)
            .chain(&self.cross)
            .chain(&self.normal)
            .any(|v| !v.is_finite())
        {
            return false;
        }
        let tolerance = 1e-4 * (covered as f32).max(1.);
        let [xx, yy, zz, _] = self.diagonal;
        let [xy, xz, yz, _] = self.cross;
        let trace = xx + yy + zz;
        let scale = (covered as f32).max(1.);
        let [a, b, c, d, e, f] = [xx, yy, zz, xy, xz, yz].map(|v| v / scale);
        let determinant = a * b * c + 2. * d * e * f - a * f * f - b * e * e - c * d * d;
        let [nx, ny, nz, _] = self.normal.map(|v| v / scale);
        let [ca, cb, cc, cd, ce, cf] = [
            a - nx * nx,
            b - ny * ny,
            c - nz * nz,
            d - nx * ny,
            e - nx * nz,
            f - ny * nz,
        ];
        let covariance_det =
            ca * cb * cc + 2. * cd * ce * cf - ca * cf * cf - cb * ce * ce - cc * cd * cd;
        if (trace - covered as f32).abs() > tolerance
            || xx < 0.
            || yy < 0.
            || zz < 0.
            || determinant < -1e-5
            || covariance_det < -1e-5
            || ca < -1e-4
            || cb < -1e-4
            || cc < -1e-4
            || cd * cd > ca * cb + 1e-4
            || ce * ce > ca * cc + 1e-4
            || cf * cf > cb * cc + 1e-4
            || self.diagonal[3] != 0.
            || self.cross[3] != 0.
            || self.normal[3] != 0.
            || self.normal[..3].iter().map(|v| v * v).sum::<f32>() > scale * scale + tolerance
            || xy * xy > xx * yy + tolerance
            || xz * xz > xx * zz + tolerance
            || yz * yz > yy * zz + tolerance
        {
            return false;
        }
        if covered == 0 {
            return events == 0
                && self.depth == [0.; 4]
                && self.normal == [0.; 4]
                && self.diagonal == [0.; 4]
                && self.cross == [0.; 4];
        }
        self.depth[0] >= -1e-4
            && self.depth[1] <= 1.0001
            && self.depth[0] <= self.depth[1] + 1e-4
            && self.depth[2] <= self.depth[3] + tolerance
            && self.depth[2] >= -tolerance
            && self.depth[3] <= covered as f32 + tolerance
    }
}
#[cfg(test)]
#[path = "coarse_cache_tests.rs"]
mod tests;
