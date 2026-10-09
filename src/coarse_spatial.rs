//! Spatial source observations for an inactive coarse visibility experiment.
use crate::coarse_cache::{self, SourceCache};
pub const PACK_SHADER: &str = include_str!("coarse_spatial_pack.wgsl");
pub const SHADER: &str = include_str!("coarse_spatial.wgsl");
pub const WORDS_PER_CELL: usize = 6 * 64;
pub const MAX_BYTES: usize = 128 << 20;
const HEADER: usize = 120;
pub struct SpatialCache {
    pub identity: [u8; 32],
    pub parts: u32,
    pub two_sided: u32,
    /// Cell, signed axis, v, u. Zero is an unobserved sample, not proven empty.
    pub samples: Vec<u32>,
}
impl SpatialCache {
    pub fn valid(&self, source: &SourceCache) -> bool {
        if source.samples_side != 8
            || !(1..=4).contains(&self.parts)
            || self.two_sided >> self.parts != 0
            || self.samples.len() != source.cells.len().saturating_mul(WORDS_PER_CELL)
        {
            return false;
        }
        for (cell, samples) in source
            .cells
            .iter()
            .zip(self.samples.chunks_exact(WORDS_PER_CELL))
        {
            for (direction, row) in cell.directions[..6].iter().zip(samples.chunks_exact(64)) {
                let mut counts = [0u32; 8];
                for &word in row {
                    if word == 0 {
                        continue;
                    }
                    let part = (word >> 12) & 3;
                    if word & 4095 == 0 || part >= self.parts {
                        return false;
                    }
                    counts[part as usize] += 1;
                }
                if counts.iter().sum::<u32>() > direction.counts[1] {
                    return false;
                }
            }
        }
        true
    }
    pub fn encode(&self, source: &SourceCache) -> Option<Vec<u8>> {
        if !self.valid(source) {
            return None;
        }
        let size = HEADER.checked_add(self.samples.len().checked_mul(4)?)?;
        if size > MAX_BYTES {
            return None;
        }
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(b"SFSP0002");
        bytes.extend_from_slice(&self.identity);
        bytes.extend_from_slice(blake3::hash(&coarse_cache::encode(source)?).as_bytes());
        bytes.extend_from_slice(&[0; 32]);
        for word in [
            source.prototype,
            source.resolution,
            self.parts | (self.two_sided << 8),
            self.samples.len() as u32,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(bytemuck::cast_slice(&self.samples));
        let hash = blake3::hash(&bytes[104..]);
        bytes[72..104].copy_from_slice(hash.as_bytes());
        Some(bytes)
    }
    pub fn decode(bytes: &[u8], identity: [u8; 32], source: &SourceCache) -> Option<Self> {
        if !cfg!(target_endian = "little")
            || bytes.len() < HEADER
            || bytes.len() > MAX_BYTES
            || bytes.get(..8)? != b"SFSP0002"
            || bytes.get(8..40)? != identity
            || blake3::hash(&bytes[104..]).as_bytes() != bytes.get(72..104)?
            || blake3::hash(&coarse_cache::encode(source)?).as_bytes() != bytes.get(40..72)?
        {
            return None;
        }
        let word = |at| Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
        let meta = word(112)?;
        let count = word(116)? as usize;
        if word(104)? != source.prototype
            || word(108)? != source.resolution
            || meta & 0xffff0000 != 0
            || HEADER.checked_add(count.checked_mul(4)?)? != bytes.len()
        {
            return None;
        }
        let cache = Self {
            identity,
            parts: meta & 255,
            two_sided: (meta >> 8) & 255,
            samples: bytes[HEADER..]
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
        };
        cache.valid(source).then_some(cache)
    }
    pub fn unknown_cells(&self) -> Vec<u32> {
        self.samples
            .chunks_exact(WORDS_PER_CELL)
            .map(|c| u32::from(c.iter().all(|&w| w == 0)))
            .collect()
    }
}
