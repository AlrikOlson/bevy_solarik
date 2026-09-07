//! Metal-compatible storage of canonical mesh streams.
//!
//! Geometry is copied byte-for-byte from Bevy's allocator. Per-mesh word offsets
//! describe the real vertex stride; colour, UV1 and any additional source
//! attributes remain in the pool. Two bounded pages share one word address space,
//! so a scene can exceed one storage binding without adding Metal buffer slots.
use super::blas::BlasManager;
use bevy_asset::AssetId;
use bevy_mesh::{Indices, Mesh};
use bevy_platform::collections::HashMap;
use bevy_render::{
    mesh::allocator::MeshAllocator,
    render_resource::{
        Buffer, BufferDescriptor, BufferId, BufferUsages, CommandEncoder, PrimitiveTopology,
    },
    renderer::RenderDevice,
};

/// An absent optional stream. Mirrored by the WGSL vertex accessor.
pub(super) const NO_STREAM: u32 = u32::MAX;

/// Word offsets in the allocator's actual interleaved vertex layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct VertexLayout {
    pub stride: u32,
    pub normal: u32,
    pub uv0: u32,
    pub uv1: u32,
    pub tangent: u32,
    pub colour: u32,
}

impl VertexLayout {
    pub fn of(mesh: &Mesh) -> Option<Self> {
        if !mesh.enable_raytracing
            || mesh.primitive_topology() != PrimitiveTopology::TriangleList
            || !matches!(mesh.indices(), Some(Indices::U32(_)))
        {
            return None;
        }
        let mut layout = Self {
            stride: 0,
            normal: NO_STREAM,
            uv0: NO_STREAM,
            uv1: NO_STREAM,
            tangent: NO_STREAM,
            colour: NO_STREAM,
        };
        let mut position = false;
        // Mesh::attributes() uses the same ascending id order as
        // Mesh::create_packed_vertex_buffer_data(). Attribute formats define their size.
        for (attribute, _) in mesh.attributes() {
            let at = layout.stride;
            if attribute.id == Mesh::ATTRIBUTE_POSITION.id {
                position = at == 0 && attribute.format == Mesh::ATTRIBUTE_POSITION.format;
            } else if attribute.id == Mesh::ATTRIBUTE_NORMAL.id {
                layout.normal = at;
            } else if attribute.id == Mesh::ATTRIBUTE_UV_0.id {
                layout.uv0 = at;
            } else if attribute.id == Mesh::ATTRIBUTE_UV_1.id {
                layout.uv1 = at;
            } else if attribute.id == Mesh::ATTRIBUTE_TANGENT.id {
                layout.tangent = at;
            } else if attribute.id == Mesh::ATTRIBUTE_COLOR.id {
                layout.colour = at;
            }
            let bytes = attribute.format.size();
            if bytes % 4 != 0 {
                return None;
            }
            layout.stride += (bytes / 4) as u32;
        }
        (position
            && layout.normal != NO_STREAM
            && layout.uv0 != NO_STREAM
            && layout.tangent != NO_STREAM)
            .then_some(layout)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PoolKey {
    mesh: AssetId<Mesh>,
    vertex: BufferId,
    index: BufferId,
    v0: u32,
    v1: u32,
    i0: u32,
    i1: u32,
    layout: VertexLayout,
    revision: u64,
}

#[derive(Default)]
pub(super) struct GeometryPool {
    pages: Option<[Buffer; 2]>,
    key: Vec<PoolKey>,
    pub offsets: HashMap<AssetId<Mesh>, (u32, u32)>,
}

pub(super) struct PoolUpload {
    pub pages: [Buffer; 2],
    copies: Vec<(AssetId<Mesh>, u64, u64, u64, u64, u64, u64)>,
}

impl PoolUpload {
    pub fn encode(&self, allocator: &MeshAllocator, encoder: &mut CommandEncoder) {
        for &(mesh, v_src, v_len, v_dst, i_src, i_len, i_dst) in &self.copies {
            if let (Some(vs), Some(is)) = (
                allocator.mesh_vertex_slice(&mesh),
                allocator.mesh_index_slice(&mesh),
            ) {
                for (source, mut src, len, dst) in [
                    (vs.buffer, v_src, v_len, v_dst),
                    (is.buffer, i_src, i_len, i_dst),
                ] {
                    for (page, at, bytes) in copy_spans(self.pages[0].size(), dst, len) {
                        if bytes > 0 {
                            encoder.copy_buffer_to_buffer(
                                source,
                                src,
                                &self.pages[page],
                                at,
                                bytes,
                            );
                            src += bytes;
                        }
                    }
                }
            }
        }
    }
}

impl GeometryPool {
    pub fn meshes(&self) -> usize {
        self.key.len()
    }
    pub fn bytes(&self) -> u64 {
        self.pages
            .as_ref()
            .map_or(0, |pages| pages.iter().map(|buffer| buffer.size()).sum())
    }

    pub fn prepare(
        &mut self,
        meshes: impl Iterator<Item = AssetId<Mesh>>,
        allocator: &MeshAllocator,
        blas: &BlasManager,
        device: &RenderDevice,
    ) -> Option<PoolUpload> {
        let mut keys = Vec::new();
        for mesh in meshes {
            let (Some(_), Some(layout), Some(vs), Some(is)) = (
                blas.get(&mesh),
                blas.vertex_layout(&mesh),
                allocator.mesh_vertex_slice(&mesh),
                allocator.mesh_index_slice(&mesh),
            ) else {
                continue;
            };
            if vs.range.is_empty() || is.range.is_empty() {
                continue;
            }
            keys.push(PoolKey {
                mesh,
                vertex: vs.buffer.id(),
                index: is.buffer.id(),
                v0: vs.range.start,
                v1: vs.range.end,
                i0: is.range.start,
                i1: is.range.end,
                layout,
                revision: blas.mesh_revision(&mesh),
            });
        }
        keys.sort_unstable_by_key(|key| key.mesh);
        keys.dedup_by_key(|key| key.mesh);
        if keys.is_empty() {
            return None;
        }
        let mut copies = Vec::new();
        if keys != self.key {
            // A deformation changes content, not allocation. Keep both pools
            // and copy only changed source spans; streaming changes repack once.
            let reuse = keys.len() == self.key.len()
                && keys.iter().zip(&self.key).all(|(a, b)| {
                    a.mesh == b.mesh
                        && a.layout == b.layout
                        && a.v1 - a.v0 == b.v1 - b.v0
                        && a.i1 - a.i0 == b.i1 - b.i0
                });
            let mut cursor = 0u64;
            let mut offsets = HashMap::default();
            for (k, key) in keys.iter().enumerate() {
                let stride = u64::from(key.layout.stride) * 4;
                let v_len = u64::from(key.v1 - key.v0) * stride;
                let i_len = u64::from(key.i1 - key.i0) * 4;
                let Ok(vertex_word) = u32::try_from(cursor / 4) else {
                    return None;
                };
                let index_at = cursor.checked_add(v_len)?;
                let Ok(index_word) = u32::try_from(index_at / 4) else {
                    return None;
                };
                offsets.insert(key.mesh, (vertex_word, index_word));
                if !reuse || *key != self.key[k] {
                    copies.push((
                        key.mesh,
                        u64::from(key.v0) * stride,
                        v_len,
                        cursor,
                        u64::from(key.i0) * 4,
                        i_len,
                        index_at,
                    ));
                }
                cursor = index_at.checked_add(i_len)?;
            }
            let limits = device.limits();
            let limit = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            let Some(sizes) = page_sizes(cursor, limit) else {
                tracing::error!(
                    "Solarik geometry exceeds two-page capacity: bytes={cursor}, page_limit={limit}"
                );
                return None;
            };
            let buffer = |label, size| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            if !reuse {
                self.pages = Some([
                    buffer("solarik_geometry_page_0", sizes[0]),
                    buffer("solarik_geometry_page_1", sizes[1]),
                ]);
            }
            self.offsets = offsets;
            self.key = keys;
        }
        Some(PoolUpload {
            pages: self.pages.as_ref()?.clone(),
            copies,
        })
    }
}

/// Keep each binding within both the device's byte limit and the shader's
/// 32-bit buffer-size representation. A one-page scene binds a dummy word.
fn page_sizes(bytes: u64, limit: u64) -> Option<[u64; 2]> {
    let limit = limit.min(u64::from(u32::MAX)) & !3;
    if bytes == 0 || !bytes.is_multiple_of(4) || limit < 4 || bytes > limit * 2 {
        return None;
    }
    Some([bytes.min(limit), bytes.saturating_sub(limit).max(4)])
}

/// (page, destination offset, byte count), in source order. A stream can
/// straddle a page even inside a vertex or triangle; no padding changes it.
fn copy_spans(page_bytes: u64, dst: u64, len: u64) -> [(usize, u64, u64); 2] {
    let first = page_bytes.saturating_sub(dst).min(len);
    [
        (0, dst.min(page_bytes), first),
        (1, dst.saturating_sub(page_bytes), len - first),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_tree_scene_fits_two_metal_bindings() {
        let bytes = 4_546_257_208 + 688_896_000;
        let limit = 4_294_967_292;
        let sizes = page_sizes(bytes, limit).expect("full source cohort");
        assert_eq!(sizes, [limit, 940_185_916]);
        assert_eq!(sizes.iter().sum::<u64>(), bytes);
        assert!(page_sizes(limit * 2 + 4, limit).is_none());
        assert_eq!(page_sizes(228, limit), Some([228, 4]));
        assert_eq!(page_sizes(8, 7), Some([4, 4]));
        assert!(page_sizes(3, limit).is_none());
        assert!(page_sizes(8, 3).is_none());
    }

    #[test]
    fn page_upload_preserves_bytes_and_partial_updates() {
        let canonical: Vec<u8> = (0..228).map(|n| (n * 71) as u8).collect();
        for boundary in [4usize, 28, 116, 220, 228] {
            let mut pages = [vec![0; boundary], vec![0; (228 - boundary).max(4)]];
            let mut upload = |dst: usize, data: &[u8]| {
                let mut source = 0;
                for (page, at, len) in copy_spans(boundary as u64, dst as u64, data.len() as u64) {
                    let (at, len) = (at as usize, len as usize);
                    pages[page][at..at + len].copy_from_slice(&data[source..source + len]);
                    source += len;
                }
                assert_eq!(source, data.len());
            };
            // Vertex and index allocator spans, then a deformation crossing
            // the boundary. No other mesh bytes may change on that update.
            upload(0, &canonical[..216]);
            upload(216, &canonical[216..]);
            let begin = boundary.saturating_sub(4);
            let end = (boundary + 4).min(228);
            upload(begin, &vec![255; end - begin]);
            let mut expected = canonical.clone();
            expected[begin..end].fill(255);
            let actual: Vec<u8> = pages.into_iter().flatten().take(228).collect();
            assert_eq!(actual, expected, "boundary {boundary}");
        }
    }
    fn source() -> Mesh {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, Default::default());
        mesh.enable_raytracing = true;
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[1.0, 2.0, 3.0]; 3]);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 3]);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.1, 0.2]; 3]);
        mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 3]);
        mesh.insert_indices(Indices::U32(vec![0, 1, 2]));
        mesh
    }
    #[test]
    fn offsets_follow_source_bytes_with_and_without_extra_streams() {
        let mut mesh = source();
        let base = VertexLayout::of(&mesh).expect("base PNUVT layout");
        assert_eq!(
            (base.stride, base.normal, base.uv0, base.tangent),
            (12, 3, 6, 8)
        );
        assert_eq!(base.colour, NO_STREAM);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.25, 0.75]; 3]);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.2, 0.4, 0.6, 1.0]; 3]);
        let layout = VertexLayout::of(&mesh).expect("UV1 and colour");
        let data = mesh.create_packed_vertex_buffer_data();
        let word = |at: u32| {
            f32::from_le_bytes(
                data[at as usize * 4..at as usize * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        };
        assert_eq!(layout.stride, 18);
        assert_eq!(word(layout.uv1), 0.25);
        assert_eq!(word(layout.colour + 2), 0.6);
        assert_eq!(word(layout.tangent + 3), 1.0);
    }
    #[test]
    fn unsupported_geometry_is_explicitly_rejected() {
        let mut mesh = source();
        mesh.enable_raytracing = false;
        assert!(VertexLayout::of(&mesh).is_none());
        mesh.enable_raytracing = true;
        mesh.remove_attribute(Mesh::ATTRIBUTE_POSITION);
        assert!(VertexLayout::of(&mesh).is_none());
    }
}
