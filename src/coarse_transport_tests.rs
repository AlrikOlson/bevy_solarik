use super::*;
#[test]
fn rates_fit_measured_transmission_over_actual_chords() {
    for direction in directions() {
        let lengths = chords(direction, 8);
        assert!(!lengths.is_empty());
        for hit in 1..lengths.len() {
            let target = hit as f64 / lengths.len() as f64;
            let rate = fit(target, &lengths).unwrap();
            assert!((coverage(f64::from(rate), &lengths) - target).abs() < 1e-7);
        }
    }
    assert!(fit(0., &[1.]).is_none());
    assert!(fit(1., &[1.]).is_none());
    assert!(fit(0.5, &[f64::NAN]).is_none());
}
#[test]
fn unit_axis_fit_has_analytic_optical_depth_and_reciprocity() {
    for direction in directions().into_iter().take(6) {
        let lengths = chords(direction, 8);
        assert_eq!(lengths, vec![1.; 64]);
        assert!((f64::from(fit(0.5, &lengths).unwrap()) - 2.0_f64.ln()).abs() < 3e-8);
    }
    for pair in directions().chunks_exact(2) {
        let mut a = chords(pair[0], 8);
        let mut b = chords(pair[1], 8);
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-14));
    }
}
