//! Experimental directional homogeneous transport fitted to finite source rays.
//! Passive transport is not a claim of source correlation or scattering fidelity.
use crate::coarse_cache::{DIRECTIONS, SourceCache, directions};
use bevy_math::DVec3;
pub const SHADER: &str = include_str!("coarse_transport.wgsl");
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Kernel {
    /// Optical rate per cell width; never an intersection count.
    pub rates: [f32; DIRECTIONS],
    /// Interior coverage measurements only. Zero/saturated bins require fallback.
    pub measured: u32,
    pub reserved: u32,
}

/// Reconstruct the cooker footprint in f64, in units of one cell width.
pub fn chords(direction: [f32; 3], side: u32) -> Vec<f64> {
    let d = DVec3::from_array(direction.map(f64::from));
    let a = d.abs();
    let axis = if a.z < a.x.min(a.y) {
        DVec3::Z
    } else if a.y < a.x {
        DVec3::Y
    } else {
        DVec3::X
    };
    let right = d.cross(axis).normalize();
    let up = right.cross(d);
    let extent = [
        right.abs().element_sum() * 0.5,
        up.abs().element_sum() * 0.5,
    ];
    let mut lengths = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let u = (f64::from(x) + 0.5) / f64::from(side) * 2. - 1.;
            let v = (f64::from(y) + 0.5) / f64::from(side) * 2. - 1.;
            let origin = right * u * extent[0] + up * v * extent[1] - d * 1.5;
            let mut low = 0.0_f64;
            let mut high = 1e10_f64;
            for axis in 0..3 {
                if d[axis] == 0. {
                    if origin[axis].abs() > 0.5 {
                        high = -1.;
                    }
                } else {
                    let p = (-0.5 - origin[axis]) / d[axis];
                    let q = (0.5 - origin[axis]) / d[axis];
                    low = low.max(p.min(q));
                    high = high.min(p.max(q));
                }
            }
            if high > low {
                lengths.push(high - low);
            }
        }
    }
    lengths
}
pub fn coverage(rate: f64, chords: &[f64]) -> f64 {
    chords
        .iter()
        .map(|&length| -(-rate * length).exp_m1())
        .sum::<f64>()
        / chords.len() as f64
}
pub fn fit(target: f64, chords: &[f64]) -> Option<f32> {
    if !(0. ..1.).contains(&target)
        || target == 0.
        || chords.is_empty()
        || chords.iter().any(|&v| !v.is_finite() || v <= 0.)
    {
        return None;
    }
    let mut low = 0.;
    let mut high = 1.;
    while coverage(high, chords) < target {
        high *= 2.;
        if high > 1e6 {
            return None;
        }
    }
    for _ in 0..48 {
        let middle = (low + high) * 0.5;
        if coverage(middle, chords) < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    Some(((low + high) * 0.5) as f32)
}
pub fn prepare(source: &SourceCache) -> Option<Vec<Kernel>> {
    if !(1..=16).contains(&source.samples_side) {
        return None;
    }
    let lengths = directions().map(|d| chords(d, source.samples_side));
    // Only 14 * (N+1) fits, independent of the number of source cells.
    let tables: [Vec<f32>; DIRECTIONS] = core::array::from_fn(|i| {
        let count = lengths[i].len();
        (0..=count)
            .map(|hit| {
                // Endpoint estimates are diagnostic only and MUST retain unresolved.
                let target = if hit == 0 || hit == count {
                    (hit as f64 + 0.5) / (count as f64 + 1.)
                } else {
                    hit as f64 / count as f64
                };
                fit(target, &lengths[i]).unwrap_or(f32::NAN)
            })
            .collect()
    });
    source
        .cells
        .iter()
        .map(|cell| {
            let mut kernel = Kernel {
                rates: [0.; DIRECTIONS],
                measured: 0,
                reserved: 0,
            };
            for i in 0..DIRECTIONS {
                let row = &cell.directions[i];
                let [support, hit, _, overflow] = row.counts;
                if !row.validate(source.samples_side)
                    || overflow != 0
                    || support as usize != lengths[i].len()
                {
                    return None;
                }
                kernel.rates[i] = *tables[i].get(hit as usize)?;
                if !kernel.rates[i].is_finite() {
                    return None;
                }
                if hit > 0 && hit < support {
                    kernel.measured |= 1 << i;
                }
            }
            Some(kernel)
        })
        .collect()
}
#[cfg(test)]
#[path = "coarse_transport_tests.rs"]
mod tests;
