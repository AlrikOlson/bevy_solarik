use super::{Cell, MAX_CACHE_BYTES, SourceCache};
const MAGIC: &[u8; 8] = b"SFCG0001";
const HEADER: usize = 88;

pub fn encode(cache: &SourceCache) -> Option<Vec<u8>> {
    let payload = bytemuck::cast_slice(&cache.cells);
    if !valid(cache) || payload.len() > MAX_CACHE_BYTES - HEADER {
        return None;
    }
    let mut bytes = Vec::with_capacity(HEADER + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&cache.identity);
    bytes.extend_from_slice(&[0; 32]);
    for word in [
        cache.prototype,
        cache.resolution,
        cache.samples_side,
        cache.cells.len() as u32,
    ] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    let checksum = blake3::hash(&bytes[72..]);
    bytes[40..72].copy_from_slice(checksum.as_bytes());
    Some(bytes)
}

pub fn decode(bytes: &[u8], identity: [u8; 32]) -> Option<SourceCache> {
    if !cfg!(target_endian = "little")
        || bytes.len() < HEADER
        || bytes.len() > MAX_CACHE_BYTES
        || bytes.get(..8)? != MAGIC
        || bytes.get(8..40)? != identity
    {
        return None;
    }
    let payload = bytes.get(HEADER..)?;
    if blake3::hash(bytes.get(72..)?).as_bytes() != bytes.get(40..72)? {
        return None;
    }
    let word = |offset| {
        Some(u32::from_le_bytes(
            bytes.get(offset..offset + 4)?.try_into().ok()?,
        ))
    };
    let count = word(84)? as usize;
    if count.checked_mul(size_of::<Cell>())? != payload.len() {
        return None;
    }
    let cells = payload
        .chunks_exact(size_of::<Cell>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let cache = SourceCache {
        identity,
        prototype: word(72)?,
        resolution: word(76)?,
        samples_side: word(80)?,
        cells,
    };
    valid(&cache).then_some(cache)
}

fn valid(cache: &SourceCache) -> bool {
    if !cfg!(target_endian = "little")
        || cache.prototype == 0
        || cache.cells.is_empty()
        || !super::valid_resolution(cache.resolution)
        || !matches!(cache.samples_side, 1 | 2 | 4 | 8 | 16)
    {
        return false;
    }
    let step = 1. / cache.resolution as f32;
    let mut previous = None;
    for cell in &cache.cells {
        let key = [cell.grid[0], cell.grid[1], cell.grid[2]];
        if previous.is_some_and(|old| old >= key)
            || cell.grid[3] != cache.resolution as i32
            || cell.centre_half.iter().any(|v| !v.is_finite())
            || cell.centre_half[3] != step * 0.5
            || (0..3).any(|i| cell.centre_half[i] != (cell.grid[i] as f32 + 0.5) * step)
            || cell
                .directions
                .iter()
                .any(|d| !d.validate(cache.samples_side))
        {
            return false;
        }
        previous = Some(key);
    }
    true
}
