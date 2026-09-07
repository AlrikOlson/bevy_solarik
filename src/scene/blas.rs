use super::geometry::VertexLayout;
use super::light_sampling::{triangle_edges, world_area};
use alloc::collections::VecDeque;
use bevy_asset::AssetId;
use bevy_ecs::{
    change_detection::Mut,
    resource::Resource,
    system::{Res, ResMut},
    world::World,
};
use bevy_math::{Mat4, Vec3};
use bevy_mesh::Mesh;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    mesh::{
        RenderMesh,
        allocator::{MeshAllocator, MeshBufferSlice},
    },
    render_asset::ExtractedAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};

/// After compacting this many vertices worth of meshes per frame, no further BLAS will be compacted.
/// Lower this number to distribute the work across more frames.
const MAX_COMPACTION_VERTICES_PER_FRAME: u32 = 400_000;

#[derive(Resource, Default)]
pub struct BlasManager {
    blas: HashMap<AssetId<Mesh>, Blas>,
    layouts: HashMap<AssetId<Mesh>, VertexLayout>,
    rejected: HashMap<AssetId<Mesh>, String>,
    revisions: HashMap<AssetId<Mesh>, u64>,
    next_revision: u64,
    triangle_edges: HashMap<AssetId<Mesh>, Vec<[Vec3; 2]>>,
    compaction_queue: VecDeque<(AssetId<Mesh>, u32, bool)>,
    opacity: OpacityBook,
    /// Meshes whose BLAS was dropped because its opacity no longer matched
    /// the materials on it; rebuilt by `prepare_raytracing_blas` next frame.
    rebuild_queue: Vec<AssetId<Mesh>>,
}

impl BlasManager {
    fn queue_compaction(&mut self, mesh: AssetId<Mesh>, vertices: u32) {
        // Tickets address a mesh slot. Replacing its BLAS invalidates every
        // earlier ticket, including one already awaiting async compaction.
        self.compaction_queue
            .retain(|(queued, _, _)| *queued != mesh);
        self.compaction_queue.push_back((mesh, vertices, false));
    }

    pub(super) fn vertex_layout(&self, mesh: &AssetId<Mesh>) -> Option<VertexLayout> {
        self.layouts.get(mesh).copied()
    }
    pub(super) fn mesh_revision(&self, mesh: &AssetId<Mesh>) -> u64 {
        self.revisions.get(mesh).copied().unwrap_or(0)
    }
    pub fn mesh_world_area(&self, mesh: &AssetId<Mesh>, transform: Mat4) -> f64 {
        self.triangle_edges
            .get(mesh)
            .map_or(0.0, |edges| world_area(edges, transform))
    }

    pub(super) fn rejection(&self, mesh: &AssetId<Mesh>) -> Option<&str> {
        self.rejected.get(mesh).map(String::as_str)
    }

    pub fn get(&self, mesh: &AssetId<Mesh>) -> Option<&Blas> {
        self.blas.get(mesh)
    }

    /// Tell the manager which meshes carry an alpha-masked or blended
    /// material this frame. Their BLAS is built without the OPAQUE flag so
    /// the shader gets to alpha-test every candidate hit; a BLAS built with
    /// the wrong flag is dropped here and rebuilt next frame.
    pub fn set_non_opaque_meshes(&mut self, meshes: HashSet<AssetId<Mesh>>) {
        for mesh in self.opacity.require(meshes) {
            if self.blas.remove(&mesh).is_some() {
                self.rebuild_queue.push(mesh);
            }
        }
    }
}

/// Which meshes need alpha-tested (non-opaque) acceleration structures, and
/// which opacity each built BLAS actually has. Pure bookkeeping, so the
/// rebuild decision can be tested without a GPU.
#[derive(Default)]
struct OpacityBook {
    required_non_opaque: HashSet<AssetId<Mesh>>,
    built_non_opaque: HashMap<AssetId<Mesh>, bool>,
}

impl OpacityBook {
    /// Replace the requirement; returns the meshes whose existing BLAS was
    /// built with the other opacity and must be rebuilt.
    fn require(&mut self, meshes: HashSet<AssetId<Mesh>>) -> Vec<AssetId<Mesh>> {
        self.required_non_opaque = meshes;
        let stale: Vec<_> = self
            .built_non_opaque
            .iter()
            .filter(|(mesh, built)| **built != self.required_non_opaque.contains(*mesh))
            .map(|(mesh, _)| *mesh)
            .collect();
        for mesh in &stale {
            self.built_non_opaque.remove(mesh);
        }
        stale
    }

    fn is_non_opaque(&self, mesh: &AssetId<Mesh>) -> bool {
        self.required_non_opaque.contains(mesh)
    }

    fn record_built(&mut self, mesh: AssetId<Mesh>, non_opaque: bool) {
        self.built_non_opaque.insert(mesh, non_opaque);
    }

    fn forget(&mut self, mesh: &AssetId<Mesh>) {
        self.built_non_opaque.remove(mesh);
    }
}

pub fn prepare_raytracing_blas(
    mut blas_manager: ResMut<BlasManager>,
    extracted_meshes: Res<ExtractedAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // Delete BLAS for deleted or modified meshes
    for asset_id in extracted_meshes
        .removed
        .iter()
        .chain(extracted_meshes.modified.iter())
    {
        blas_manager.blas.remove(asset_id);
        blas_manager.layouts.remove(asset_id);
        blas_manager.rejected.remove(asset_id);
        blas_manager.revisions.remove(asset_id);
        blas_manager.triangle_edges.remove(asset_id);
        blas_manager.opacity.forget(asset_id);
    }

    for (asset_id, mesh) in &extracted_meshes.extracted {
        if let Some(layout) = VertexLayout::of(mesh) {
            blas_manager.layouts.insert(*asset_id, layout);
            blas_manager.next_revision += 1;
            let revision = blas_manager.next_revision;
            blas_manager.revisions.insert(*asset_id, revision);
            blas_manager
                .triangle_edges
                .insert(*asset_id, triangle_edges(mesh));
        } else {
            let attributes: Vec<_> = mesh.attributes().map(|(a, _)| a.name).collect();
            blas_manager.rejected.insert(
                *asset_id,
                format!(
                    "enabled={} topology={:?} vertices={} indices={:?} attributes={attributes:?}",
                    mesh.enable_raytracing,
                    mesh.primitive_topology(),
                    mesh.count_vertices(),
                    mesh.indices()
                        .map(|i| (matches!(i, bevy_mesh::Indices::U32(_)), i.len()))
                ),
            );
        }
    }

    // Meshes whose opacity changed keep their allocation; only the BLAS is
    // rebuilt (a mesh that was freed meanwhile is simply skipped).
    let rebuilds: Vec<AssetId<Mesh>> = blas_manager
        .rebuild_queue
        .drain(..)
        .filter(|asset_id| {
            !extracted_meshes.modified.contains(asset_id)
                && !extracted_meshes.removed.contains(asset_id)
                && mesh_allocator.mesh_vertex_slice(asset_id).is_some()
                && mesh_allocator.mesh_index_slice(asset_id).is_some()
        })
        .collect();

    if extracted_meshes.extracted.is_empty() && rebuilds.is_empty() {
        return;
    }

    // Create new BLAS for added or changed meshes
    let blas_resources = extracted_meshes
        .extracted
        .iter()
        .filter(|(_, mesh)| is_mesh_raytracing_compatible(mesh))
        .map(|(asset_id, _)| *asset_id)
        .chain(rebuilds)
        .map(|asset_id| {
            let vertex_slice = mesh_allocator.mesh_vertex_slice(&asset_id).unwrap();
            let index_slice = mesh_allocator.mesh_index_slice(&asset_id).unwrap();

            let non_opaque = blas_manager.opacity.is_non_opaque(&asset_id);
            let (blas, blas_size) = allocate_blas(
                &vertex_slice,
                &index_slice,
                &asset_id,
                non_opaque,
                &render_device,
            );

            blas_manager.blas.insert(asset_id, blas);
            blas_manager.opacity.record_built(asset_id, non_opaque);
            blas_manager.queue_compaction(asset_id, blas_size.vertex_count);

            (asset_id, vertex_slice, index_slice, blas_size)
        })
        .collect::<Vec<_>>();

    // Build geometry into each BLAS
    let build_entries = blas_resources
        .iter()
        .map(|(asset_id, vertex_slice, index_slice, blas_size)| {
            let geometry = BlasTriangleGeometry {
                size: blas_size,
                vertex_buffer: vertex_slice.buffer,
                first_vertex: vertex_slice.range.start,
                vertex_stride: u64::from(blas_manager.layouts[asset_id].stride) * 4,
                index_buffer: Some(index_slice.buffer),
                first_index: Some(index_slice.range.start),
                transform_buffer: None,
                transform_buffer_offset: None,
            };
            BlasBuildEntry {
                blas: &blas_manager.blas[asset_id],
                geometry: BlasGeometries::TriangleGeometries(vec![geometry]),
            }
        })
        .collect::<Vec<_>>();

    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_blas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    render_queue.submit([command_encoder.finish()]);
}

/// Compaction must not interleave with meshlet `Queue::submit` on Metal:
/// wgpu's snatch/command-index lock order otherwise deadlocks at startup.
pub fn compact_raytracing_blas(world: &mut World) {
    world.resource_scope(|world, mut manager: Mut<BlasManager>| {
        compact_queue(&mut manager, world.resource::<RenderQueue>());
    });
}

fn compact_queue(blas_manager: &mut BlasManager, render_queue: &RenderQueue) {
    let queue_size = blas_manager.compaction_queue.len();
    let mut meshes_processed = 0;
    let mut vertices_compacted = 0;

    while !blas_manager.compaction_queue.is_empty()
        && vertices_compacted < MAX_COMPACTION_VERTICES_PER_FRAME
        && meshes_processed < queue_size
    {
        meshes_processed += 1;

        let (mesh, vertex_count, compaction_started) =
            blas_manager.compaction_queue.pop_front().unwrap();

        let Some(blas) = blas_manager.get(&mesh) else {
            continue;
        };

        if !compaction_started {
            blas.prepare_compaction_async(|_| {});
        }

        if blas.ready_for_compaction() {
            let compacted_blas = render_queue.compact_blas(blas);
            blas_manager.blas.insert(mesh, compacted_blas);

            vertices_compacted += vertex_count;
            continue;
        }

        // BLAS not ready for compaction, put back in queue
        blas_manager
            .compaction_queue
            .push_back((mesh, vertex_count, true));
    }
}

fn allocate_blas(
    vertex_slice: &MeshBufferSlice,
    index_slice: &MeshBufferSlice,
    asset_id: &AssetId<Mesh>,
    non_opaque: bool,
    render_device: &RenderDevice,
) -> (Blas, BlasTriangleGeometrySizeDescriptor) {
    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: vertex_slice.range.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_slice.range.len() as u32),
        // Without OPAQUE every hit on this geometry is a candidate the
        // shader's trace_ray loop alpha-tests before confirming.
        flags: if non_opaque {
            AccelerationStructureGeometryFlags::empty()
        } else {
            AccelerationStructureGeometryFlags::OPAQUE
        },
    };

    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some(&asset_id.to_string()),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::ALLOW_COMPACTION,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    (blas, blas_size)
}

fn is_mesh_raytracing_compatible(mesh: &Mesh) -> bool {
    VertexLayout::of(mesh).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;
    use bevy_mesh::Indices;

    fn mesh(n: u128) -> AssetId<Mesh> {
        AssetId::from(Uuid::from_u128(n))
    }

    #[test]
    fn replacement_cancels_pending_compaction_of_the_old_blas() {
        let mut manager = BlasManager::default();
        manager.compaction_queue.push_back((mesh(1), 100, true));
        manager.compaction_queue.push_back((mesh(2), 200, false));
        // The same mesh receives a new BLAS after an opacity or geometry change.
        manager.queue_compaction(mesh(1), 300);
        assert_eq!(
            manager.compaction_queue.len(),
            2,
            "one ticket per live BLAS"
        );
        assert_eq!(manager.compaction_queue[0], (mesh(2), 200, false));
        assert_eq!(manager.compaction_queue[1], (mesh(1), 300, false));
    }

    #[test]
    fn canonical_colour_uv1_and_skin_streams_remain_raytraceable() {
        let mut source = Mesh::new(PrimitiveTopology::TriangleList, Default::default());
        source.enable_raytracing = true;
        source.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, 0.0]; 3]);
        source.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 3]);
        source.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; 3]);
        source.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.25, 0.75]; 3]);
        source.insert_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0, 0.0, 0.0, 1.0]; 3]);
        source.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.2, 0.4, 0.6, 1.0]; 3]);
        source.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, vec![[1.0, 0.0, 0.0, 0.0]; 3]);
        source.insert_indices(Indices::U32(vec![0, 1, 2]));
        let before = source.create_packed_vertex_buffer_data();
        assert!(
            is_mesh_raytracing_compatible(&source),
            "extra canonical streams must not reject a mesh"
        );
        assert_eq!(
            source.create_packed_vertex_buffer_data(),
            before,
            "inspection cannot strip or rewrite source data"
        );
    }

    #[test]
    fn a_new_requirement_rebuilds_only_the_blas_built_the_other_way() {
        let mut book = OpacityBook::default();
        book.record_built(mesh(1), false);
        book.record_built(mesh(2), false);
        book.record_built(mesh(3), true);

        // Mesh 1 now needs alpha testing, mesh 3 no longer does, mesh 2 is unchanged.
        let stale: HashSet<_> = book
            .require(HashSet::from_iter([mesh(1)]))
            .into_iter()
            .collect();
        assert_eq!(stale, HashSet::from_iter([mesh(1), mesh(3)]));
        assert!(book.is_non_opaque(&mesh(1)));
        assert!(!book.is_non_opaque(&mesh(2)));
        assert!(!book.is_non_opaque(&mesh(3)));

        // Once rebuilt the right way, the same requirement asks for nothing.
        book.record_built(mesh(1), true);
        book.record_built(mesh(3), false);
        assert!(book.require(HashSet::from_iter([mesh(1)])).is_empty());
    }

    #[test]
    fn a_mesh_never_built_is_not_a_rebuild() {
        let mut book = OpacityBook::default();
        assert!(book.require(HashSet::from_iter([mesh(9)])).is_empty());
        assert!(book.is_non_opaque(&mesh(9)));
        book.record_built(mesh(9), true);
        book.forget(&mesh(9));
        assert!(
            book.require(HashSet::new()).is_empty(),
            "forgotten meshes are not rebuilt"
        );
    }
}
