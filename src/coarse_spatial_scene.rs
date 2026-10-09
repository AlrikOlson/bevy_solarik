//! Lossless resident layout for immutable spatial hit maps.
use bevy_render::{
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};
pub const SHADER: &str = include_str!("coarse_spatial_scene.wgsl");
pub const MISSING: u32 = u32::MAX;
pub struct PackedSpatial {
    pub rows: Vec<[u32; 4]>,
    pub hits: Vec<u32>,
}
pub struct GpuSpatial {
    pub rows: Buffer,
    pub hits: Buffer,
    pub bytes: u64,
}
impl PackedSpatial {
    pub fn pack(words: &[u32]) -> Option<Self> {
        if words.is_empty() || !words.len().is_multiple_of(64) || words.len() > 32 * 1024 * 1024 {
            return None;
        }
        let mut rows = Vec::with_capacity(words.len() / 64);
        let mut hits = Vec::new();
        for chunk in words.chunks_exact(64) {
            let mut row = [0, 0, hits.len() as u32, 0];
            for (i, &word) in chunk.iter().enumerate() {
                if word != 0 {
                    row[i / 32] |= 1 << (i % 32);
                    hits.push(word);
                }
            }
            rows.push(row);
        }
        // Bindable sentinel allocation only. No occupancy bit points at it.
        if hits.is_empty() {
            hits.push(0);
        }
        let packed = Self { rows, hits };
        packed.valid().then_some(packed)
    }
    pub fn valid(&self) -> bool {
        let mut offset = 0usize;
        for row in &self.rows {
            if row[3] != 0 || row[2] as usize != offset {
                return false;
            }
            offset += row[0].count_ones() as usize + row[1].count_ones() as usize;
        }
        !self.rows.is_empty()
            && self.bytes() <= 128 << 20
            && if offset == 0 {
                self.hits == [0]
            } else {
                offset == self.hits.len() && self.hits.iter().all(|&w| w != 0)
            }
    }
    pub fn sample(&self, row: usize, texel: u32) -> Option<u32> {
        if texel >= 64 {
            return None;
        }
        let row = self.rows.get(row)?;
        let word = row[(texel / 32) as usize];
        let bit = 1u32 << (texel % 32);
        if word & bit == 0 {
            return Some(0);
        }
        let rank =
            (word & (bit - 1)).count_ones() + if texel >= 32 { row[0].count_ones() } else { 0 };
        self.hits.get((row[2] + rank) as usize).copied()
    }
    pub fn bytes(&self) -> usize {
        self.rows.len() * 16 + self.hits.len() * 4
    }
    /// Immutable handles; the publisher owns submission/fence retirement.
    pub fn upload(&self, device: &RenderDevice) -> Option<GpuSpatial> {
        if !self.valid() {
            return None;
        }
        let buffer = |label, bytes: &[u8]| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: BufferUsages::STORAGE,
            })
        };
        Some(GpuSpatial {
            rows: buffer(
                "spatial occupancy and offsets",
                bytemuck::cast_slice(&self.rows),
            ),
            hits: buffer("spatial source hits", bytemuck::cast_slice(&self.hits)),
            bytes: self.bytes() as u64,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_mask_boundaries_and_words_round_trip() {
        let mut words = vec![0u32; 64 * 3];
        for i in [0, 1, 30, 31, 32, 33, 62, 63, 64, 127] {
            words[i] = u32::MAX - i as u32;
        }
        for (i, w) in words[128..].iter_mut().enumerate() {
            *w = i as u32 + 1;
        }
        let packed = PackedSpatial::pack(&words).unwrap();
        assert_eq!(packed.rows[0][0], 0xc0000003);
        assert_eq!(packed.rows[0][1], 0xc0000003);
        for (i, &word) in words.iter().enumerate() {
            assert_eq!(packed.sample(i / 64, (i % 64) as u32), Some(word));
        }
        assert_eq!(packed.sample(0, 64), None);
        assert_eq!(packed.sample(3, 0), None);
    }
    #[test]
    fn empty_rows_bind_a_sentinel_without_occupancy() {
        let mut packed = PackedSpatial::pack(&[0; 128]).unwrap();
        assert_eq!(packed.bytes(), 36);
        assert_eq!(packed.sample(1, 63), Some(0));
        packed.rows[1][2] = 1;
        assert!(!packed.valid());
        assert!(PackedSpatial::pack(&[]).is_none());
        assert!(PackedSpatial::pack(&[1; 63]).is_none());
    }
}
