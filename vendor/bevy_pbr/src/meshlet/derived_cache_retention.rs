//! Disposable derived storage, bounded without evicting unrelated files.
use super::{HEADER, MAGIC, MAX_ENTRY, invalid};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read},
    path::Path,
    time::SystemTime,
};

pub(super) const LOCK: &str = ".meshlet-writer.lock";

fn full() -> io::Error {
    io::Error::new(io::ErrorKind::StorageFull, "derived cache byte limit")
}

/// A hit is already validated. Updating age is best-effort and changes no payload.
pub(super) fn touch(path: &Path) {
    if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_file()) {
        let _ = OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|file| file.set_modified(SystemTime::now()));
    }
}

fn owned(path: &Path, length: u64) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    let Ok(key) = blake3::Hash::from_hex(stem) else {
        return false;
    };
    if !(HEADER as u64..=MAX_ENTRY as u64).contains(&length) {
        return false;
    }
    let mut header = [0u8; HEADER];
    if File::open(path)
        .and_then(|mut f| f.read_exact(&mut header))
        .is_err()
    {
        return false;
    }
    &header[..8] == MAGIC
        && &header[8..40] == key.as_bytes()
        && u64::from_le_bytes(header[40..48].try_into().expect("fixed header range"))
            == length - HEADER as u64
}

/// Hold the returned lease until rename finishes. Readers need no lease.
/// Directory symlinks are resolved once; only its direct regular owned files
/// become candidates. Unknown entries count against capacity but are retained.
pub(super) fn reserve(directory: &Path, incoming: u64, capacity: u64) -> io::Result<File> {
    if incoming > capacity {
        return Err(full());
    }
    let root = directory.canonicalize()?;
    let lock_path = root.join(LOCK);
    match fs::symlink_metadata(&lock_path) {
        Ok(metadata) if !metadata.file_type().is_file() => return Err(invalid()),
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    // Never delete this inode: every cooperating writer locks the same file.
    let lease = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    lease.try_lock().map_err(io::Error::from)?;
    let mut size = 0u64;
    let mut candidates = Vec::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "mvg") || !entry.file_type()?.is_file() {
            continue;
        }
        let metadata = entry.metadata()?;
        size = size.saturating_add(metadata.len());
        if owned(&path, metadata.len()) {
            candidates.push((metadata.modified()?, path, metadata.len()));
        }
    }
    candidates.sort_unstable();
    for (modified, path, length) in candidates {
        if size.saturating_add(incoming) <= capacity {
            break;
        }
        // A concurrent read may refresh a candidate after the directory scan.
        let current = fs::symlink_metadata(&path)?;
        if !current.file_type().is_file()
            || current.modified()? != modified
            || current.len() != length
            || path.canonicalize()?.parent() != Some(root.as_path())
            || !owned(&path, length)
        {
            continue;
        }
        fs::remove_file(&path)?;
        size = size.saturating_sub(length);
        bevy_render::diagnostic::profile_value("geometry.cache_eviction", 1.0, "count");
    }
    if size.saturating_add(incoming) > capacity {
        return Err(full());
    }
    Ok(lease)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    pub(crate) fn entry(root: &Path, label: &[u8], length: usize, age: u64) -> std::path::PathBuf {
        let key = blake3::hash(label);
        let path = root.join(format!("{}.mvg", key.to_hex()));
        let mut bytes = vec![0; length];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..40].copy_from_slice(key.as_bytes());
        bytes[40..48].copy_from_slice(&((length - HEADER) as u64).to_le_bytes());
        fs::write(&path, bytes).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(age))
            .unwrap();
        path
    }
}
