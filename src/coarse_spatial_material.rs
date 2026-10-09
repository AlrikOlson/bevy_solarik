//! Source material samples indexed by dense spatial hit rank.
use crate::{coarse_appearance::Material, coarse_cache::SourceCache, coarse_spatial::SpatialCache};
use bevy_render::{
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
};
const HEADER: usize = 112;
const MAX_BYTES: usize = 128 << 20;
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Surface {
    pub color: [f32; 4],
    pub normal: [f32; 4],
    /// Roughness fourth power, metallic, reflectance, diffuse transmission.
    pub surface: [f32; 4],
}
impl Surface {
    pub fn valid(&self) -> bool {
        self.color
            .iter()
            .chain(&self.surface)
            .all(|v| v.is_finite() && (0. ..=1.).contains(v))
            && self.normal.iter().all(|v| v.is_finite())
            && self.normal[3] == 0.
            && (self.normal[..3].iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 0.001
    }
}
pub struct MaterialCache {
    pub identity: [u8; 32],
    pub materials: Vec<Material>,
    /// One surface for each nonzero spatial word, in exactly the same order.
    pub surfaces: Vec<Surface>,
}
impl MaterialCache {
    pub fn valid(&self, spatial: &SpatialCache, source: &SourceCache) -> bool {
        spatial.valid(source)
            && self.materials.len() == spatial.parts as usize
            && self.materials.iter().all(Material::valid)
            && self.surfaces.len() == spatial.samples.iter().filter(|&&w| w != 0).count()
            && self.surfaces.iter().all(Surface::valid)
    }
    pub fn encode(&self, spatial: &SpatialCache, source: &SourceCache) -> Option<Vec<u8>> {
        if !self.valid(spatial, source) || !cfg!(target_endian = "little") {
            return None;
        }
        let size =
            HEADER.checked_add((self.materials.len() + self.surfaces.len()).checked_mul(48)?)?;
        if size > MAX_BYTES {
            return None;
        }
        let mut bytes = Vec::with_capacity(size);
        bytes.extend_from_slice(b"SFSH0001");
        bytes.extend_from_slice(&self.identity);
        bytes.extend_from_slice(blake3::hash(&spatial.encode(source)?).as_bytes());
        bytes.extend_from_slice(&[0; 32]);
        bytes.extend_from_slice(&(self.materials.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.surfaces.len() as u32).to_le_bytes());
        bytes.extend_from_slice(bytemuck::cast_slice(&self.materials));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.surfaces));
        let hash = blake3::hash(&bytes[104..]);
        bytes[72..104].copy_from_slice(hash.as_bytes());
        Some(bytes)
    }
    pub fn decode(
        bytes: &[u8],
        identity: [u8; 32],
        spatial: &SpatialCache,
        source: &SourceCache,
    ) -> Option<Self> {
        if !cfg!(target_endian = "little")
            || bytes.len() < HEADER
            || bytes.len() > MAX_BYTES
            || bytes.get(..8)? != b"SFSH0001"
            || bytes.get(8..40)? != identity
            || blake3::hash(&bytes[104..]).as_bytes() != bytes.get(72..104)?
            || blake3::hash(&spatial.encode(source)?).as_bytes() != bytes.get(40..72)?
        {
            return None;
        }
        let word = |at| Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
        let parts = word(104)? as usize;
        let count = word(108)? as usize;
        if parts != spatial.parts as usize
            || HEADER.checked_add(parts.checked_add(count)?.checked_mul(48)?)? != bytes.len()
        {
            return None;
        }
        let end = HEADER + parts * 48;
        let cache = Self {
            identity,
            materials: bytes[HEADER..end]
                .chunks_exact(48)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
            surfaces: bytes[end..]
                .chunks_exact(48)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
        };
        cache.valid(spatial, source).then_some(cache)
    }
    /// Immutable source buffers; lifetime and submission fences belong to the publisher.
    pub fn upload(
        &self,
        device: &RenderDevice,
        spatial: &SpatialCache,
        source: &SourceCache,
    ) -> Option<[Buffer; 2]> {
        if !self.valid(spatial, source) {
            return None;
        }
        let dummy = [Surface::default()];
        let surfaces = if self.surfaces.is_empty() {
            &dummy[..]
        } else {
            &self.surfaces
        };
        Some(
            [
                bytemuck::cast_slice(&self.materials),
                bytemuck::cast_slice(surfaces),
            ]
            .map(|bytes| {
                device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("immutable spatial source material"),
                    contents: bytes,
                    usage: BufferUsages::STORAGE,
                })
            }),
        )
    }
}
