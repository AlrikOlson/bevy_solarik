use bevy_math::{Vec3, Vec4};

/// Optional physical dry medium. All public lengths are metres, coefficients m^-1.
/// Gas is hydrostatic with a linear temperature lapse and isothermal cap.
/// Aerosol is exponential or a Gaussian layer. These are transport inputs,
/// not a chemical/cloud microphysics simulation. See docs/atmosphere.md.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalAtmosphere {
    /// Altitude of the pressure reference above the spherical geometry floor, metres.
    pub reference_height: f32,
    pub molecular_extinction: Vec3,
    pub gas_scale_height: f32,
    pub temperature: f32,
    pub lapse: f32,
    pub cap_temperature: f32,
    pub aerosol_extinction: Vec3,
    pub aerosol_scale: f32,
    pub aerosol_centre: f32,
    pub aerosol_albedo: Vec3,
    pub aerosol_asymmetry: Vec3,
}
impl PhysicalAtmosphere {
    pub fn validate(&self) -> Result<(), &'static str> {
        for v in [self.molecular_extinction, self.aerosol_extinction] {
            if !v.is_finite() || v.min_element() < 0.0 || v.max_element() > 0.1 {
                return Err("physical extinction outside0..0.1/m");
            }
        }
        for (v, lo, hi) in [
            (self.gas_scale_height, 100.0, 100_000.0),
            (self.aerosol_scale, 100.0, 100_000.0),
            (self.temperature, 50.0, 2000.0),
            (self.cap_temperature, 50.0, self.temperature),
            (self.reference_height, 0.0, 80_000.0),
            (self.lapse, 0.0, 0.02),
            (self.aerosol_centre, 0.0, 150_000.0),
        ] {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err("invalid physical density profile");
            }
        }
        for (v, hi) in [(self.aerosol_albedo, 1.0), (self.aerosol_asymmetry, 0.95)] {
            if !v.is_finite() || v.min_element() < 0.0 || v.max_element() > hi {
                return Err("invalid particle scattering");
            }
        }
        Ok(())
    }
    /// GPU profile fields in kilometres and inverse kilometres.
    pub fn gpu_fields(self) -> [Vec4; 5] {
        [
            (self.molecular_extinction * 1000.0).extend(self.gas_scale_height * 0.001),
            Vec4::new(
                self.temperature,
                self.lapse * 1000.0,
                self.cap_temperature,
                self.reference_height * 0.001,
            ),
            (self.aerosol_extinction * 1000.0).extend(self.aerosol_scale * 0.001),
            self.aerosol_albedo.extend(self.aerosol_centre * 0.001),
            self.aerosol_asymmetry.extend(0.0),
        ]
    }
    pub fn extinction(self, height: f64) -> bevy_math::DVec3 {
        let h = height - f64::from(self.reference_height);
        let scale = f64::from(self.gas_scale_height);
        let t0 = f64::from(self.temperature);
        let lapse = f64::from(self.lapse);
        let tmin = f64::from(self.cap_temperature);
        let density = if lapse == 0.0 {
            (-h / scale).exp()
        } else {
            let t = (t0 - lapse * h).max(tmin);
            let cap = (t0 - tmin) / lapse;
            (t / t0).powf(t0 / (lapse * scale) - 1.0)
                * (-(h - cap).max(0.0) / (scale * tmin / t0)).exp()
        };
        let a = if self.aerosol_centre == 0.0 {
            (-h / f64::from(self.aerosol_scale)).exp()
        } else {
            (-0.5 * ((h - f64::from(self.aerosol_centre)) / f64::from(self.aerosol_scale)).powi(2))
                .exp()
        };
        self.molecular_extinction.as_dvec3() * density + self.aerosol_extinction.as_dvec3() * a
    }
}
