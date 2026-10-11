#![expect(
    clippy::print_stdout,
    reason = "Explicit diagnostics emit descriptor comparison timings."
)]
use super::{DirtySlots, changed_dense_slots};
use bevy_platform::collections::HashSet;
use core::hint::black_box;
use std::time::Instant;

fn indexed_changes(old: &[u32], next: &[u32], dirty: &[u32], capacity: usize) -> Vec<(u32, u32)> {
    let dirty = DirtySlots::new(capacity, dirty);
    changed_dense_slots(old, next.iter().copied(), |slot| dirty.contains(slot))
}

fn hashed_changes(old: &[u32], next: &[u32], dirty: &[u32]) -> Vec<(u32, u32)> {
    let set: HashSet<u32> = dirty.iter().copied().collect();
    changed_dense_slots(old, next.iter().copied(), |slot| set.contains(&slot))
}

#[test]
fn indexed_membership_preserves_dense_descriptor_changes() {
    let old = [0, 2, 63, 64, 65, 128];
    for (next, dirty) in [
        (old.to_vec(), vec![]),
        (old.to_vec(), vec![0, 63, 64, 128]),
        (vec![0, 128, 63, 64, 65], vec![64]),
        (vec![0, 2, 63, 64, 65, 128, 129], vec![129]),
        (old.to_vec(), old.to_vec()),
        (vec![], vec![]),
    ] {
        assert_eq!(
            indexed_changes(&old, &next, &dirty, 130),
            hashed_changes(&old, &next, &dirty)
        );
    }
    assert_eq!(indexed_changes(&[], &[], &[], 0), Vec::new());
}

#[test]
#[ignore = "full authored forest CPU diagnostic; run explicitly in release"]
fn forest_scale_dense_descriptor_comparison() {
    const COUNT: u32 = 3_145_728;
    let old: Vec<_> = (0..COUNT).collect();
    let dirty: Vec<_> = (0..4096).map(|i| i * 768).collect();
    for repetition in 0..3 {
        let start = Instant::now();
        let reference = hashed_changes(black_box(&old), black_box(&old), black_box(&dirty));
        let hashed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let indexed = indexed_changes(
            black_box(&old),
            black_box(&old),
            black_box(&dirty),
            COUNT as usize,
        );
        let indexed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(indexed, reference);
        assert_eq!(indexed.len(), dirty.len());
        println!(
            "TLAS stable repetition={repetition} count={COUNT} dirty={} hash_ms={hashed_ms:.3} indexed_ms={indexed_ms:.3}",
            dirty.len()
        );
    }
    let mut next = old.clone();
    next.swap_remove(63);
    next.push(COUNT);
    assert_eq!(
        indexed_changes(&old, &next, &dirty, COUNT as usize + 1),
        hashed_changes(&old, &next, &dirty)
    );
    assert_eq!(
        indexed_changes(&old, &old, &old, COUNT as usize),
        hashed_changes(&old, &old, &old)
    );
    assert!(indexed_changes(&old, &[], &[], COUNT as usize).is_empty());
}
