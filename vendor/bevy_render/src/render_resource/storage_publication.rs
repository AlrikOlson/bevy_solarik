//! Capacity scheduling for persistent scene storage.

const PAGE: usize = 64 * 1024;

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
