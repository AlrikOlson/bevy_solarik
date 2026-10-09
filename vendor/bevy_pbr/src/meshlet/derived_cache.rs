//! Exact, bounded derived data. Cache failures always fall back to the compiler.
use super::{
    MeshToMeshletMeshConversionError, MeshletMesh,
    asset::{MESHLET_MESH_ASSET_VERSION, MeshletAabb},
};
use alloc::sync::Arc;
use bevy_mesh::{Indices, Mesh};
use bytemuck::Pod;
use core::sync::atomic::{AtomicU64, Ordering};
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"MVGDDC01";
const HEADER: usize = 80;
const MAX_ENTRY: usize = 512 << 20;
const MAX_DIRECTORY: u64 = 2 << 30;
const RECIPE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/meshlet-compiler-recipe.bin"));
static NEXT_WRITE: AtomicU64 = AtomicU64::new(0);

impl MeshletMesh {
    /// Compile or load byte-identical derived geometry. The key includes every
    /// mesh input, conversion policy, source recipe and verified build toolchain.
    /// Corrupt, obsolete, unavailable or full caches fall back without affecting
    /// geometry, tolerances, ray meshes or the caller's source asset.
    pub fn from_mesh_cached(
        mesh: &Mesh,
        precision: u8,
        preserve_area: bool,
        directory: &Path,
    ) -> Result<Self, MeshToMeshletMeshConversionError> {
        cached(mesh, precision, preserve_area, directory, || {
            convert(mesh, precision, preserve_area)
        })
    }

    /// Load an exact cooked result before deterministic caller preprocessing.
    ///
    /// The caller recipe must identify all preprocessing code, dependencies and
    /// non-mesh inputs. The source mesh, precision, area policy and renderer
    /// compiler recipe are also keyed. The closure runs only on a miss; its
    /// errors propagate unchanged and are never persisted.
    pub fn from_mesh_cached_preprocessed<E>(
        mesh: Mesh,
        precision: u8,
        preserve_area: bool,
        directory: &Path,
        preprocessing_recipe: &[u8],
        compile: impl FnOnce(Mesh) -> Result<Self, E>,
    ) -> Result<Self, E> {
        let key = preprocessed_key(&mesh, precision, preserve_area, preprocessing_recipe);
        let mut compiled = false;
        let result = cached_key(key, directory, || {
            compiled = true;
            bevy_render::diagnostic::profile_value(
                "geometry.preprocessed_cache_miss",
                1.0,
                "count",
            );
            compile(mesh)
        });
        if !compiled && result.is_ok() {
            bevy_render::diagnostic::profile_value("geometry.preprocessed_cache_hit", 1.0, "count");
        }
        result
    }
}

fn preprocessed_key(
    mesh: &Mesh,
    precision: u8,
    preserve_area: bool,
    recipe: &[u8],
) -> blake3::Hash {
    let mut hash = blake3::Hasher::new();
    hash_field(&mut hash, b"meshlet-preprocessing-v1");
    hash_field(
        &mut hash,
        key(mesh, precision, preserve_area, RECIPE).as_bytes(),
    );
    hash_field(&mut hash, recipe);
    hash.finalize()
}

fn cached(
    mesh: &Mesh,
    precision: u8,
    preserve_area: bool,
    directory: &Path,
    compile: impl FnOnce() -> Result<MeshletMesh, MeshToMeshletMeshConversionError>,
) -> Result<MeshletMesh, MeshToMeshletMeshConversionError> {
    cached_key(
        key(mesh, precision, preserve_area, RECIPE),
        directory,
        compile,
    )
}

fn cached_key<E>(
    key: blake3::Hash,
    directory: &Path,
    compile: impl FnOnce() -> Result<MeshletMesh, E>,
) -> Result<MeshletMesh, E> {
    if env!("MESHLET_CACHE_COMPILER_KNOWN") != "1" {
        bevy_render::diagnostic::profile_value("geometry.cache_disabled", 1.0, "count");
        return compile();
    }
    let path = directory.join(format!("{}.mvg", key.to_hex()));
    {
        let _profile = bevy_render::diagnostic::profile_scope("geometry.cache_read");
        if let Ok(cached) = load(&path, key.as_bytes()) {
            bevy_render::diagnostic::profile_value("geometry.cache_hit", 1.0, "count");
            retention::touch(&path);
            return Ok(cached);
        }
    }
    bevy_render::diagnostic::profile_value("geometry.cache_miss", 1.0, "count");
    let geometry = {
        let _profile = bevy_render::diagnostic::profile_scope("geometry.compile");
        compile()?
    };
    let _profile = bevy_render::diagnostic::profile_scope("geometry.cache_write");
    if let Err(error) = publish(directory, &path, key.as_bytes(), &geometry) {
        tracing::warn!("Meshlet derived cache unavailable: {error}; compiled geometry retained");
    }
    Ok(geometry)
}

fn convert(
    mesh: &Mesh,
    precision: u8,
    preserve_area: bool,
) -> Result<MeshletMesh, MeshToMeshletMeshConversionError> {
    if preserve_area {
        MeshletMesh::from_mesh_preserve_area(mesh, precision)
    } else {
        MeshletMesh::from_mesh(mesh, precision)
    }
}

fn hash_field(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn key(mesh: &Mesh, precision: u8, preserve_area: bool, recipe: &[u8]) -> blake3::Hash {
    let mut hash = blake3::Hasher::new();
    hash_field(&mut hash, MAGIC);
    hash_field(&mut hash, &MESHLET_MESH_ASSET_VERSION.to_le_bytes());
    hash_field(&mut hash, recipe);
    hash_field(
        &mut hash,
        &[
            precision,
            u8::from(preserve_area),
            u8::from(cfg!(target_endian = "little")),
        ],
    );
    hash_field(
        &mut hash,
        format!("{:?}", mesh.primitive_topology()).as_bytes(),
    );
    for (attribute, values) in mesh.attributes() {
        hash_field(
            &mut hash,
            format!(
                "{:?}:{:?}:{}",
                attribute.id, attribute.format, attribute.name
            )
            .as_bytes(),
        );
        hash_field(&mut hash, values.get_bytes());
    }
    match mesh.indices() {
        Some(Indices::U16(indices)) => {
            hash_field(&mut hash, b"u16");
            hash_field(&mut hash, bytemuck::cast_slice(indices));
        }
        Some(Indices::U32(indices)) => {
            hash_field(&mut hash, b"u32");
            hash_field(&mut hash, bytemuck::cast_slice(indices));
        }
        None => hash_field(&mut hash, b"unindexed"),
    }
    hash.finalize()
}

fn invalid() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid meshlet derived entry")
}

fn encode(mesh: &MeshletMesh) -> Vec<u8> {
    let mut payload = Vec::with_capacity(mesh.storage_bytes() + 100);
    payload.extend_from_slice(bytemuck::bytes_of(&mesh.aabb));
    payload.extend_from_slice(&mesh.bvh_depth.to_le_bytes());
    macro_rules! field {
        ($name:ident) => {{
            let bytes = bytemuck::cast_slice(&mesh.$name);
            payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            payload.extend_from_slice(bytes);
        }};
    }
    field!(vertex_positions);
    field!(vertex_normals);
    field!(vertex_uvs);
    field!(vertex_tangents);
    field!(vertex_tangent_indices);
    field!(indices);
    field!(bvh);
    field!(meshlets);
    field!(meshlet_cull_data);
    payload
}

fn take<'a>(bytes: &mut &'a [u8], count: usize) -> io::Result<&'a [u8]> {
    if count > bytes.len() {
        return Err(invalid());
    }
    let (head, tail) = bytes.split_at(count);
    *bytes = tail;
    Ok(head)
}

fn array<T: Pod>(bytes: &mut &[u8]) -> io::Result<Arc<[T]>> {
    let length = u64::from_le_bytes(take(bytes, 8)?.try_into().map_err(|_| invalid())?);
    let length = usize::try_from(length).map_err(|_| invalid())?;
    if !length.is_multiple_of(size_of::<T>()) {
        return Err(invalid());
    }
    let data = take(bytes, length)?;
    let mut values: Arc<[T]> = core::iter::repeat_with(T::zeroed)
        .take(length / size_of::<T>())
        .collect();
    bytemuck::cast_slice_mut(Arc::get_mut(&mut values).ok_or_else(invalid)?).copy_from_slice(data);
    Ok(values)
}

fn decode(mut bytes: &[u8]) -> io::Result<MeshletMesh> {
    let aabb: MeshletAabb =
        bytemuck::pod_read_unaligned(take(&mut bytes, size_of::<MeshletAabb>())?);
    let bvh_depth = u32::from_le_bytes(take(&mut bytes, 4)?.try_into().map_err(|_| invalid())?);
    let mesh = MeshletMesh {
        aabb,
        bvh_depth,
        vertex_positions: array(&mut bytes)?,
        vertex_normals: array(&mut bytes)?,
        vertex_uvs: array(&mut bytes)?,
        vertex_tangents: array(&mut bytes)?,
        vertex_tangent_indices: array(&mut bytes)?,
        indices: array(&mut bytes)?,
        bvh: array(&mut bytes)?,
        meshlets: array(&mut bytes)?,
        meshlet_cull_data: array(&mut bytes)?,
    };
    if !bytes.is_empty()
        || mesh.storage_bytes() > MAX_ENTRY
        || !(1..=64).contains(&mesh.bvh_depth)
        || mesh.bvh.is_empty()
        || mesh.meshlets.is_empty()
        || !mesh.aabb.center.is_finite()
        || !mesh.aabb.half_extent.is_finite()
        || mesh.aabb.half_extent.min_element() < 0.0
        || mesh.meshlets.len() != mesh.meshlet_cull_data.len()
        || mesh.vertex_normals.len() != mesh.vertex_uvs.len()
        || mesh.vertex_normals.len() != mesh.vertex_tangent_indices.len()
        || mesh
            .vertex_tangent_indices
            .iter()
            .any(|&i| i as usize >= mesh.vertex_tangents.len())
    {
        return Err(invalid());
    }
    for m in mesh.meshlets.iter() {
        let vertices = u64::from(m.vertex_count_minus_one) + 1;
        let bits = u64::from(m.bits_per_vertex_position_channel_x)
            + u64::from(m.bits_per_vertex_position_channel_y)
            + u64::from(m.bits_per_vertex_position_channel_z);
        if u64::from(m.start_vertex_position_bit) + vertices * bits
            > mesh.vertex_positions.len() as u64 * 32
            || u64::from(m.start_vertex_attribute_id) + vertices > mesh.vertex_normals.len() as u64
            || u64::from(m.start_index_id) + u64::from(m.triangle_count) * 3
                > mesh.indices.len() as u64
        {
            return Err(invalid());
        }
    }
    Ok(mesh)
}

fn load(path: &Path, key: &[u8; 32]) -> io::Result<MeshletMesh> {
    let file = fs::File::open(path)?;
    let length = usize::try_from(file.metadata()?.len()).map_err(|_| invalid())?;
    if !(HEADER..=MAX_ENTRY).contains(&length) {
        return Err(invalid());
    }
    let mut bytes = Vec::with_capacity(length);
    file.take((MAX_ENTRY + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() != length || &bytes[..8] != MAGIC || &bytes[8..40] != key {
        return Err(invalid());
    }
    let size = u64::from_le_bytes(bytes[40..48].try_into().map_err(|_| invalid())?);
    if size != (length - HEADER) as u64
        || &bytes[48..80] != blake3::hash(&bytes[HEADER..]).as_bytes()
    {
        return Err(invalid());
    }
    decode(&bytes[HEADER..])
}

fn publish(directory: &Path, path: &Path, key: &[u8; 32], mesh: &MeshletMesh) -> io::Result<()> {
    fs::create_dir_all(directory)?;
    let payload = encode(mesh);
    if payload.len() > MAX_ENTRY - HEADER {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            "derived cache entry limit",
        ));
    }
    let _lease = retention::reserve(directory, (payload.len() + HEADER) as u64, MAX_DIRECTORY)?;
    // A competing compiler may have published this exact key while we worked.
    if load(path, key).is_ok() {
        retention::touch(path);
        return Ok(());
    }
    let temp = directory.join(format!(
        "{}.{}.{}.partial",
        blake3::Hash::from_bytes(*key).to_hex(),
        std::process::id(),
        NEXT_WRITE.fetch_add(1, Ordering::Relaxed)
    ));
    // Only clean up a temporary file that this invocation actually created.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        let mut file = file;
        file.write_all(MAGIC)?;
        file.write_all(key)?;
        file.write_all(&(payload.len() as u64).to_le_bytes())?;
        file.write_all(blake3::hash(&payload).as_bytes())?;
        file.write_all(&payload)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[path = "derived_cache_retention.rs"]
mod retention;

#[cfg(test)]
#[path = "derived_cache_tests.rs"]
mod tests;
