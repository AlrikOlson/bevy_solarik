//! Per-material visible-first appearance observations, separate from geometry.
//! These measurements are not an area NDF or an accepted scattering closure.
use crate::coarse_cache::{DIRECTIONS, Directional, SourceCache};
#[path = "coarse_appearance_codec.rs"]
mod codec;
pub use codec::{decode, encode};
pub const SHADER: &str = include_str!("coarse_appearance.wgsl");
pub const MAX_BYTES: usize = 128 << 20;
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Material {
    pub base_color: [f32; 4],
    /// Perceptual roughness, metallic, reflectance, diffuse transmission.
    pub surface: [f32; 4],
    /// IOR, normal Y flip (0/1), reserved zeros.
    pub optical: [f32; 4],
}
impl Material {
    pub fn valid(&self) -> bool {
        self.base_color
            .iter()
            .chain(&self.surface)
            .all(|v| v.is_finite() && (0. ..=1.).contains(v))
            && self.optical[0].is_finite()
            && (1. ..=3.).contains(&self.optical[0])
            && matches!(self.optical[1], 0. | 1.)
            && self.optical[2..] == [0.; 2]
    }
}
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Moment {
    /// Linear source RGBA sums after material factor.
    pub color: [f32; 4],
    /// Visible first-hit shading normal first and second moments.
    pub normal: [f32; 4],
    pub diagonal: [f32; 4],
    pub cross: [f32; 4],
    /// Sum roughness^4, metallic, reflectance, diffuse transmission.
    pub surface: [f32; 4],
    /// Hit count; overflow/invalid and reserved zeros.
    pub counts: [u32; 4],
}
impl Moment {
    pub fn valid(&self, side: u32) -> bool {
        let count = self.counts[0];
        if count == 0 && (self.color != [0.; 4] || self.surface != [0.; 4]) {
            return false;
        }
        let mut materials = [0; 8];
        materials[0] = count;
        let normal = Directional {
            counts: [count, count, count, self.counts[1]],
            normal: self.normal,
            diagonal: self.diagonal,
            cross: self.cross,
            materials,
            ..Default::default()
        };
        self.counts[2..] == [0; 2]
            && normal.validate(side)
            && self
                .color
                .iter()
                .chain(&self.surface)
                .all(|v| v.is_finite() && *v >= 0. && *v <= count as f32 + 1e-4)
    }
}
pub struct AppearanceCache {
    pub identity: [u8; 32],
    pub materials: Vec<Material>,
    /// Cell-major, then direction, then original part/material.
    pub moments: Vec<Moment>,
}
impl AppearanceCache {
    pub fn valid(&self, source: &SourceCache) -> bool {
        let parts = self.materials.len();
        if parts == 0
            || parts > 8
            || self.materials.iter().any(|v| !v.valid())
            || self.moments.len()
                != source
                    .cells
                    .len()
                    .saturating_mul(DIRECTIONS)
                    .saturating_mul(parts)
        {
            return false;
        }
        for (cell, rows) in source
            .cells
            .iter()
            .zip(self.moments.chunks_exact(DIRECTIONS * parts))
        {
            for (direction, moments) in cell.directions.iter().zip(rows.chunks_exact(parts)) {
                if direction.materials[parts..].iter().any(|&v| v != 0) {
                    return false;
                }
                for (moment, &count) in moments.iter().zip(&direction.materials) {
                    if moment.counts[0] != count || !moment.valid(source.samples_side) {
                        return false;
                    }
                }
            }
        }
        true
    }
}
#[cfg(test)]
#[path = "coarse_appearance_tests.rs"]
mod tests;
