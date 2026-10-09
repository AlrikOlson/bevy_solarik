//! Deterministic, source-sampled regional surfaces for distant geometry.
//! This is a geometric approximation, not a homogeneous extinction model.
//! Columns retain the two exterior events. Interior transport and oblique-ray
//! equivalence require application controls before selecting this representation.
use crate::coarse_spatial_material::Surface;
use alloc::collections::BTreeMap;
use bevy_math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Facet {
    pub corners: [Vec3; 4],
    pub surface: Surface,
    pub material: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct Patch {
    pub corners: [Vec3; 4],
    pub surface: Surface,
    pub material: u32,
}
#[derive(Clone, Copy)]
struct Hit {
    depth: f32,
    surface: Surface,
    material: u32,
    positive: bool,
}
struct Column {
    low: Hit,
    high: Hit,
}
/// Coordinates must be relative to a stable, metre-scale regional frame.
pub struct Builder {
    step: f32,
    limit: usize,
    columns: BTreeMap<(u8, i32, i32), Column>,
    failed: bool,
    pub tested_samples: usize,
}
impl Builder {
    pub fn new(step: f32, limit: usize) -> Result<Self, &'static str> {
        if !step.is_finite() || step <= 0. || limit == 0 || limit > 1_000_000 {
            return Err("invalid regional sampling budget");
        }
        Ok(Self {
            step,
            limit,
            columns: BTreeMap::new(),
            failed: false,
            tested_samples: 0,
        })
    }
    pub fn insert(&mut self, facet: Facet) -> Result<(), &'static str> {
        let result = self.insert_inner(facet);
        self.failed |= result.is_err();
        result
    }
    fn insert_inner(&mut self, facet: Facet) -> Result<(), &'static str> {
        if self.failed {
            return Err("regional builder already failed");
        }
        if !facet.surface.valid() || facet.corners.iter().any(|p| !p.is_finite()) {
            return Err("invalid regional source facet");
        }
        self.triangle(
            [facet.corners[0], facet.corners[1], facet.corners[2]],
            facet,
        )?;
        self.triangle(
            [facet.corners[0], facet.corners[2], facet.corners[3]],
            facet,
        )
    }
    fn triangle(&mut self, p: [Vec3; 3], facet: Facet) -> Result<(), &'static str> {
        let normal = (p[1] - p[0]).cross(p[2] - p[0]);
        let absolute = normal.abs();
        if absolute.max_element() <= f32::MIN_POSITIVE {
            return Ok(());
        }
        let axis = if absolute.x >= absolute.y && absolute.x >= absolute.z {
            0
        } else if absolute.y >= absolute.z {
            1
        } else {
            2
        };
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let low = p.into_iter().fold(Vec3::splat(f32::INFINITY), Vec3::min) / self.step;
        let high = p
            .into_iter()
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max)
            / self.step;
        if low.min_element() < -8_000_000. || high.max_element() > 8_000_000. {
            return Err("regional lattice exceeds float precision");
        }
        let lower = [(low[u] - 0.5).ceil() as i32, (low[v] - 0.5).ceil() as i32];
        let upper = [
            (high[u] - 0.5).floor() as i32,
            (high[v] - 0.5).floor() as i32,
        ];
        let samples = i64::from((upper[0] - lower[0] + 1).max(0))
            * i64::from((upper[1] - lower[1] + 1).max(0));
        if samples > self.limit as i64 * 4 {
            return Err("regional triangle exceeds sampling budget");
        }
        for y in lower[1]..=upper[1] {
            for x in lower[0]..=upper[0] {
                self.tested_samples += 1;
                let a = (x as f32 + 0.5) * self.step;
                let b = (y as f32 + 0.5) * self.step;
                let Some(weights) = barycentric(p, u, v, a, b) else {
                    continue;
                };
                let depth =
                    weights[0] * p[0][axis] + weights[1] * p[1][axis] + weights[2] * p[2][axis];
                let hit = Hit {
                    depth,
                    surface: facet.surface,
                    material: facet.material,
                    positive: normal[axis] > 0.,
                };
                let key = (axis as u8, x, y);
                if let Some(column) = self.columns.get_mut(&key) {
                    if hit.depth < column.low.depth {
                        column.low = hit;
                    }
                    if hit.depth > column.high.depth {
                        column.high = hit;
                    }
                } else {
                    if self.columns.len() >= self.limit {
                        return Err("regional column budget exceeded");
                    }
                    self.columns.insert(
                        key,
                        Column {
                            low: hit,
                            high: hit,
                        },
                    );
                }
            }
        }
        Ok(())
    }
    pub fn finish(self) -> Result<Vec<Patch>, &'static str> {
        if self.failed {
            return Err("cannot publish a partially failed regional cook");
        }
        let mut patches = Vec::with_capacity(self.columns.len() * 2);
        for ((axis, x, y), column) in self.columns {
            patches.push(patch(axis as usize, x, y, self.step, column.low));
            if column.high.depth.to_bits() != column.low.depth.to_bits() {
                patches.push(patch(axis as usize, x, y, self.step, column.high));
            }
        }
        Ok(patches)
    }
}
fn barycentric(p: [Vec3; 3], u: usize, v: usize, x: f32, y: f32) -> Option<[f32; 3]> {
    let edge =
        |a: Vec3, b: Vec3, x: f32, y: f32| (b[u] - a[u]) * (y - a[v]) - (b[v] - a[v]) * (x - a[u]);
    let area = edge(p[0], p[1], p[2][u], p[2][v]);
    let a = edge(p[1], p[2], x, y) / area;
    let b = edge(p[2], p[0], x, y) / area;
    let c = 1. - a - b;
    (a >= -1e-5 && b >= -1e-5 && c >= -1e-5).then_some([a, b, c])
}
fn patch(axis: usize, x: i32, y: i32, step: f32, hit: Hit) -> Patch {
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut corners = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)].map(|(a, b)| {
        let mut p = Vec3::ZERO;
        p[axis] = hit.depth;
        p[u] = (x as f32 + a) * step;
        p[v] = (y as f32 + b) * step;
        p
    });
    if !hit.positive {
        corners.swap(1, 3);
    }
    Patch {
        corners,
        surface: hit.surface,
        material: hit.material,
    }
}
#[cfg(test)]
#[path = "coarse_region_tests.rs"]
mod tests;
