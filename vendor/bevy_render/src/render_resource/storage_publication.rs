//! Capacity scheduling for persistent scene storage.

const PAGE: usize = 64 * 1024;

/// Bound CPU copy-command overhead without turning sparse edits into dense uploads.
/// Each bridged gap is at most 64 bytes and three times its following dirty run:
/// total copied bytes are therefore at most four times the exact dirty bytes.
pub(super) fn coalesce(ranges: Vec<core::ops::Range<usize>>) -> Vec<core::ops::Range<usize>> {
    if ranges.len() <= 128 {
        return ranges;
    }
    let mut packed: Vec<core::ops::Range<usize>> = Vec::new();
    for range in ranges {
        if let Some(last) = packed.last_mut() {
            let gap = range.start - last.end;
            if gap <= 64 && gap <= range.len().saturating_mul(3) {
                last.end = range.end;
                continue;
            }
        }
        packed.push(range);
    }
    packed
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalescing_limits_extra_bytes_and_preserves_distant_sparse_edits() {
        let small: Vec<_> = (0..128).map(|i| i * 16..i * 16 + 4).collect();
        assert_eq!(coalesce(small.clone()), small);
        let sparse: Vec<_> = (0..1024).map(|i| i * 122_880..i * 122_880 + 160).collect();
        assert_eq!(coalesce(sparse.clone()), sparse);
        for gap in [4, 8, 12, 16, 64, 68] {
            let ranges: Vec<_> = (0..1024)
                .map(|i| i * (4 + gap)..i * (4 + gap) + 4)
                .collect();
            let packed = coalesce(ranges.clone());
            assert!(packed.iter().map(|r| r.len()).sum::<usize>() <= 4 * 4096);
            for original in &ranges {
                assert!(
                    packed
                        .iter()
                        .any(|r| r.start <= original.start && r.end >= original.end)
                );
            }
            if gap <= 12 {
                assert_eq!(packed.len(), 1);
            } else {
                assert_eq!(packed, ranges);
            }
        }
    }

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
}
