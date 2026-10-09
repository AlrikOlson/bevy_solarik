use super::{AppearanceCache, MAX_BYTES, Material, Moment};
use crate::coarse_cache::{self, SourceCache};
const HEADER: usize = 120;
pub fn encode(cache: &AppearanceCache, source: &SourceCache) -> Option<Vec<u8>> {
    if !cache.valid(source) {
        return None;
    }
    let size = HEADER
        .checked_add(cache.materials.len().checked_mul(size_of::<Material>())?)?
        .checked_add(cache.moments.len().checked_mul(size_of::<Moment>())?)?;
    if size > MAX_BYTES {
        return None;
    }
    let source_bytes = coarse_cache::encode(source)?;
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(b"SFCA0001");
    bytes.extend_from_slice(&cache.identity);
    bytes.extend_from_slice(blake3::hash(&source_bytes).as_bytes());
    bytes.extend_from_slice(&[0; 32]);
    for word in [
        source.prototype,
        source.resolution,
        cache.materials.len() as u32,
        cache.moments.len() as u32,
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(bytemuck::cast_slice(&cache.materials));
    bytes.extend_from_slice(bytemuck::cast_slice(&cache.moments));
    let checksum = blake3::hash(&bytes[104..]);
    bytes[72..104].copy_from_slice(checksum.as_bytes());
    Some(bytes)
}
pub fn decode(bytes: &[u8], identity: [u8; 32], source: &SourceCache) -> Option<AppearanceCache> {
    if !cfg!(target_endian = "little")
        || bytes.len() < HEADER
        || bytes.len() > MAX_BYTES
        || bytes.get(..8)? != b"SFCA0001"
        || bytes.get(8..40)? != identity
        || blake3::hash(&bytes[104..]).as_bytes() != bytes.get(72..104)?
        || blake3::hash(&coarse_cache::encode(source)?).as_bytes() != bytes.get(40..72)?
    {
        return None;
    }
    let word = |at| Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
    let parts = word(112)? as usize;
    let rows = word(116)? as usize;
    if word(104)? != source.prototype
        || word(108)? != source.resolution
        || !(1..=8).contains(&parts)
    {
        return None;
    }
    let split = HEADER.checked_add(parts.checked_mul(size_of::<Material>())?)?;
    if split.checked_add(rows.checked_mul(size_of::<Moment>())?)? != bytes.len() {
        return None;
    }
    let cache = AppearanceCache {
        identity,
        materials: bytes
            .get(HEADER..split)?
            .chunks_exact(size_of::<Material>())
            .map(bytemuck::pod_read_unaligned)
            .collect(),
        moments: bytes
            .get(split..)?
            .chunks_exact(size_of::<Moment>())
            .map(bytemuck::pod_read_unaligned)
            .collect(),
    };
    cache.valid(source).then_some(cache)
}
