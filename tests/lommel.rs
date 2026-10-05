//! Particulate scattering reciprocity and hemispherical energy contracts.
use bevy_solarik::lommel::lommel_seeliger;

#[test]
fn reciprocal_flat_full_phase_and_conservative_energy() {
    for mu in [0.001, 0.1, 0.5, 1.0] {
        let rho = 0.2;
        assert!((lommel_seeliger(rho, mu, mu) * mu * core::f64::consts::PI - rho).abs() < 1e-12);
        for other in [0.01, 0.3, 0.9] {
            assert_eq!(
                lommel_seeliger(rho, mu, other),
                lommel_seeliger(rho, other, mu)
            );
        }
        let steps = 100_000;
        let albedo: f64 = (0..steps)
            .map(|i| {
                let outgoing = (f64::from(i) + 0.5) / f64::from(steps);
                2.0 * core::f64::consts::PI * lommel_seeliger(0.25, mu, outgoing) * outgoing
                    / f64::from(steps)
            })
            .sum();
        assert!(albedo <= 1.0);
        let exact = 1.0 - mu * (1.0 + 1.0 / mu).ln();
        assert!((albedo - exact).abs() < 1e-7);
    }
    assert_eq!(lommel_seeliger(0.2, 0.0, 1.0), 0.0);
    assert_eq!(lommel_seeliger(0.2, 1.0, -0.1), 0.0);
}
