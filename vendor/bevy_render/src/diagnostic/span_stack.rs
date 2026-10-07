//! Stable removal matters when independent threads nest profiling spans.

pub(crate) fn remove_last_matching<T>(
    spans: &mut Vec<T>,
    predicate: impl FnMut(&T) -> bool,
) -> Option<T> {
    let index = spans.iter().rposition(predicate)?;
    Some(spans.remove(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_another_thread_preserves_parent_child_order() {
        // A closes while B has an outer and inner span. swap_remove would
        // move B's inner before its outer and close the wrong GPU query.
        let mut spans = vec![(0, "a"), (1, "b-outer"), (1, "b-inner")];
        assert_eq!(
            remove_last_matching(&mut spans, |s| s.0 == 0),
            Some((0, "a"))
        );
        assert_eq!(
            remove_last_matching(&mut spans, |s| s.0 == 1),
            Some((1, "b-inner"))
        );
        assert_eq!(
            remove_last_matching(&mut spans, |s| s.0 == 1),
            Some((1, "b-outer"))
        );
        assert!(spans.is_empty());
    }

    #[test]
    fn absent_thread_leaves_other_open_spans_unchanged() {
        let mut spans = vec![(1, "outer"), (1, "inner")];
        assert_eq!(remove_last_matching(&mut spans, |s| s.0 == 2), None);
        assert_eq!(spans, [(1, "outer"), (1, "inner")]);
    }
}
