//! Optional photometric camera metering and conservative forward-scatter optics.
mod background;
mod cinematic;
pub use cinematic::CinematicLens;
mod gpu;
use alloc::sync::Arc;
pub use background::CameraOpticsSystems;
use bevy_ecs::component::Component;
use bevy_render::extract_component::ExtractComponent;
pub use gpu::CameraOpticsPlugin;
use std::sync::Mutex;

/// Actual GPU state: adapted/target EV100, metered cd/m², sample count,
/// automatic flag, manual EV100, input centre HDR luminance, pre-exposure,
/// followed by the centre RGBA after optical scattering and exposure.
#[derive(Component, Clone, Default, ExtractComponent)]
pub struct OpticsReadback(pub Arc<Mutex<Option<[f32; 12]>>>);

/// A generated camera, not an empirical model of a particular lens or eye.
#[derive(Component, Clone, ExtractComponent)]
pub struct CameraOptics {
    pub automatic: bool,
    /// Positive compensation makes the image brighter.
    pub compensation: f32,
    pub min_ev: f32,
    pub max_ev: f32,
    /// Bounds on adaptation in stops per second, in each direction.
    pub brighten_speed: f32,
    pub darken_speed: f32,
    /// Standard deviation of a small-angle forward scattering lobe, radians.
    pub scatter_sigma: f32,
    /// Fraction of transmitted light redistributed, without a brightness threshold.
    pub scatter_fraction: f32,
    /// Explicit discontinuity token; a new token initializes at the camera's EV.
    pub reset: u32,
}
impl Default for CameraOptics {
    fn default() -> Self {
        Self {
            automatic: true,
            compensation: 0.0,
            min_ev: -6.0,
            max_ev: 20.0,
            brighten_speed: 1.0,
            darken_speed: 4.0,
            scatter_sigma: 0.0017453293,
            scatter_fraction: 0.01,
            reset: 0,
        }
    }
}
impl CameraOptics {
    pub fn validate(&self) -> Result<(), &'static str> {
        for (v, lo, hi) in [
            (self.compensation, -8.0, 8.0),
            (self.min_ev, -20.0, 30.0),
            (self.max_ev, -20.0, 30.0),
            (self.brighten_speed, 0.01, 20.0),
            (self.darken_speed, 0.01, 20.0),
            (self.scatter_sigma, 0.0, 0.01),
            (self.scatter_fraction, 0.0, 0.1),
        ] {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err("invalid camera optics");
            }
        }
        if self.min_ev >= self.max_ev {
            return Err("unordered exposure bounds");
        }
        Ok(())
    }
}

/// Reflected-light meter with K=12.5 and ISO100 (Sekonic calibration).
pub fn meter_ev(luminance: f64) -> Option<f64> {
    (luminance.is_finite() && luminance > 0.0).then(|| (8.0 * luminance).log2())
}
/// Exact first-order relaxation with a hard rate bound; tau=1 second.
pub fn adapt_ev(current: f64, target: f64, dt: f64, brighten: f64, darken: f64) -> f64 {
    let dt = dt.clamp(0.0, 0.1);
    let delta = target - current;
    let rate = if delta > 0.0 { darken } else { brighten };
    current + (delta * (-(-dt).exp_m1())).clamp(-rate * dt, rate * dt)
}
/// Normalized discrete Gaussian; radius ceil(4 sigma), at most128 pixels.
pub fn scatter_kernel(sigma_pixels: f64) -> Result<Vec<f64>, &'static str> {
    if !sigma_pixels.is_finite() || !(0.0..=32.0).contains(&sigma_pixels) {
        return Err("scatter footprint outside0..32px");
    }
    if sigma_pixels < 0.1 {
        return Ok(vec![1.0]);
    }
    let r = (4.0 * sigma_pixels).ceil() as i32;
    let mut k: Vec<_> = (-r..=r)
        .map(|i| (-0.5 * (f64::from(i) / sigma_pixels).powi(2)).exp())
        .collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|w| *w /= sum);
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibrated_meter_and_invalid_luminance() {
        assert_eq!(meter_ev(4096.0), Some(15.0));
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(meter_ev(v), None);
        }
        assert!((meter_ev(8192.0).unwrap() - meter_ev(4096.0).unwrap() - 1.0).abs() < 1e-12);
    }
    #[test]
    fn bounded_adaptation_cannot_overshoot_and_converges() {
        let mut ev = 15.0;
        for _ in 0..1000 {
            let next = adapt_ev(ev, 3.0, 0.02, 1.0, 4.0);
            assert!(next >= 3.0 && next <= ev && ev - next <= 0.0200001);
            ev = next;
        }
        assert!((ev - 3.0).abs() < 0.001);
        assert_eq!(adapt_ev(3.0, 15.0, 0.1, 1.0, 4.0), 3.4);
    }
    #[test]
    fn scattering_conserves_uniform_fields_and_impulse_energy_at_edges() {
        let n = 64;
        for sigma in [0.0, 0.3, 2.2, 12.0] {
            let k = scatter_kernel(sigma).unwrap();
            let r = k.len() / 2;
            for source in [0, 1, 31, 63] {
                let mut sum = 0.0;
                for x in 0..n {
                    let mut constant = 0.0;
                    for (j, w) in k.iter().enumerate() {
                        let p = (x + j as i32 - r as i32).rem_euclid(2 * n);
                        let p = if p >= n { 2 * n - 1 - p } else { p };
                        constant += w;
                        if p == source {
                            sum += w;
                        }
                    }
                    assert!((constant - 1.0).abs() < 1e-12);
                }
                assert!((sum - 1.0).abs() < 1e-12);
            }
        }
        assert!(scatter_kernel(33.0).is_err());
        let lens = CameraOptics {
            scatter_fraction: f32::NAN,
            ..Default::default()
        };
        assert!(lens.validate().is_err());
    }
}
