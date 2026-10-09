//! Capacity and copy scheduling for persistent scene storage.
use core::ops::Range;

const PAGE: usize = 64 * 1024;
const MAX_RANGES: usize = 128;

pub(super) fn allocation_size(required: u64, limit: u64) -> u64 {
    let required = required.max(4);
    assert!(
        required <= limit,
        "scene storage exceeds the device buffer limit"
    );
    required
        .checked_next_power_of_two()
        .unwrap_or(limit)
        .max(PAGE as u64)
        .min(limit)
}

/// Keep small edits exact. Bound fragmented work using already published bytes
/// for the gaps, so unmarked CPU changes never become visible accidentally.
pub(super) fn upload_ranges(ranges: Vec<Range<usize>>, size: usize) -> Vec<Range<usize>> {
    if ranges.len() <= MAX_RANGES {
        return ranges;
    }
    let mut pages: Vec<Range<usize>> = Vec::new();
    for range in ranges {
        let start = range.start / PAGE * PAGE;
        let end = range.end.div_ceil(PAGE).saturating_mul(PAGE).min(size);
        if let Some(last) = pages.last_mut().filter(|last| last.end >= start) {
            last.end = last.end.max(end);
        } else {
            pages.push(start..end);
            if pages.len() > MAX_RANGES {
                return core::iter::once(0..size).collect();
            }
        }
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_is_amortized_and_clamped_to_the_device_limit() {
        assert_eq!(allocation_size(4, 1024 * 1024), PAGE as u64);
        assert_eq!(
            allocation_size(PAGE as u64 + 4, 1024 * 1024),
            2 * PAGE as u64
        );
        assert_eq!(
            allocation_size(2 * PAGE as u64 + 4, 3 * PAGE as u64),
            3 * PAGE as u64
        );
        assert_eq!(allocation_size(4, 16), 16);
        assert_eq!(allocation_size(u64::MAX - 3, u64::MAX), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "scene storage exceeds the device buffer limit")]
    fn impossible_capacity_is_rejected_before_allocation() {
        allocation_size(1028, 1024);
    }

    #[test]
    fn sparse_copies_stay_exact_and_fragmented_pages_cover_every_edit() {
        let sparse = vec![4..8, 4 * PAGE..4 * PAGE + 4];
        assert_eq!(upload_ranges(sparse.clone(), 5 * PAGE), sparse);
        let edits: Vec<_> = (0..4096).map(|i| i * 32..i * 32 + 4).collect();
        let size = 2 * PAGE - 16;
        let merged = upload_ranges(edits.clone(), size);
        assert_eq!(merged, vec![0..size]);
        assert!(edits.iter().all(|edit| {
            merged
                .iter()
                .any(|range| range.start <= edit.start && range.end >= edit.end)
        }));
    }

    #[test]
    fn excessive_page_fragmentation_uses_one_bounded_full_copy() {
        let edits = (0..129).map(|i| i * PAGE * 2..i * PAGE * 2 + 4).collect();
        let size = PAGE * 258;
        assert_eq!(upload_ranges(edits, size), vec![0..size]);
    }
}
