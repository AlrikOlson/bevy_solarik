//! Validated immutable source-response profiles.
use alloc::sync::Arc;
use bevy_ecs::resource::Resource;
use bevy_render::extract_resource::ExtractResource;
#[derive(Debug)]
pub struct SourceProfile {
    rows: Arc<[[u32; 4]]>,
    pub identity: [u8; 32],
    pub extents: [[f32; 2]; 2],
}
impl SourceProfile {
    pub fn new(rows: Vec<[u32; 4]>) -> Result<Self, &'static str> {
        if rows.len() < 531 || rows.len() * 16 > 64 << 20 || rows[0] != [2, 33, 8, 3] {
            return Err("invalid finite canopy source header");
        }
        let extents = [
            [f32::from_bits(rows[1][0]), f32::from_bits(rows[1][1])],
            [f32::from_bits(rows[2][0]), f32::from_bits(rows[2][1])],
        ];
        super::validate_sources(extents)?;
        if rows[1][2..] == rows[2][2..] {
            return Err("duplicate canopy source identity");
        }
        let mut next = 531;
        for row in &rows[3..531] {
            next = validate_row(&rows, *row, next)?;
        }
        if next != rows.len() {
            return Err("unreferenced canopy source data");
        }
        Ok(Self {
            identity: *blake3::hash(bytemuck::cast_slice(&rows)).as_bytes(),
            rows: rows.into(),
            extents,
        })
    }
    pub fn rows(&self) -> &[[u32; 4]] {
        &self.rows
    }
    pub fn bytes(&self) -> usize {
        self.rows.len() * 16
    }
}
fn validate_row(data: &[[u32; 4]], row: [u32; 4], next: usize) -> Result<usize, &'static str> {
    let area = f32::from_bits(row[2]);
    let end = next
        .checked_add(row[1] as usize * 3)
        .ok_or("canopy source overflow")?;
    if row[0] as usize != next
        || end > data.len()
        || row[3] != 0
        || !area.is_finite()
        || area < 0.
        || (row[1] == 0) != (area == 0.)
    {
        return Err("invalid canopy source row");
    }
    let mut cdf = 0.;
    for mark in data[next..end].as_chunks::<3>().0 {
        let n = mark[0].map(f32::from_bits);
        let color = mark[1][..3].iter().map(|x| f32::from_bits(*x));
        let surface = mark[2].map(f32::from_bits);
        let unit = n[..3].iter().map(|v| v * v).sum::<f32>();
        if n.iter().any(|x| !x.is_finite())
            || (unit - 1.).abs() > 0.001
            || n[3] < cdf
            || n[3] > 1.
            || color
                .chain(surface)
                .any(|x| !x.is_finite() || !(0. ..=1.).contains(&x))
        {
            return Err("invalid canopy joint source mark");
        }
        cdf = n[3];
    }
    if row[1] > 0 && cdf != 1. {
        return Err("incomplete canopy source distribution");
    }
    Ok(end)
}
/// One immutable source set for all canopy regions in this scene.
/// Changing it invalidates region readiness until matching regions prepare.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(bevy_render::RenderApp)]
pub struct CanopySources(pub Option<Arc<SourceProfile>>);

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> Vec<[u32; 4]> {
        let mut rows = vec![[0; 4]; 531];
        rows[0] = [2, 33, 8, 3];
        rows[1] = [0, 1f32.to_bits(), 1, 0];
        rows[2] = [0, 1f32.to_bits(), 2, 0];
        for i in 3..531 {
            rows[i] = [rows.len() as u32, 1, 1f32.to_bits(), 0];
            rows.extend([
                [0, 1f32.to_bits(), 0, 1f32.to_bits()],
                [0.2f32.to_bits(), 0.4f32.to_bits(), 0.1f32.to_bits(), 7],
                [0.1f32.to_bits(), 0, 0.5f32.to_bits(), 0.3f32.to_bits()],
            ]);
        }
        rows
    }
    #[test]
    fn immutable_profile_validates_joint_marks_and_region_contract() {
        let rows = valid();
        let profile = SourceProfile::new(rows.clone()).unwrap();
        assert_eq!(profile.rows(), rows);
        assert_eq!(profile.bytes(), rows.len() * 16);
        for (r, c, v) in [
            (0, 0, 3),
            (1, 1, 0),
            (3, 0, 530),
            (3, 2, f32::NAN.to_bits()),
            (531, 1, 2f32.to_bits()),
            (531, 3, 0.5f32.to_bits()),
            (533, 3, 2f32.to_bits()),
        ] {
            let mut bad = rows.clone();
            bad[r][c] = v;
            assert!(SourceProfile::new(bad).is_err());
        }
        let mut duplicate = rows.clone();
        duplicate[2] = duplicate[1];
        assert!(SourceProfile::new(duplicate).is_err());
        let mut modified = rows;
        modified[533][0] = 0.2f32.to_bits();
        assert_ne!(
            SourceProfile::new(modified).unwrap().identity,
            profile.identity
        );
        let region = Arc::new(super::super::PreparedRegion::build(vec![], [[0., 1.]; 2]).unwrap());
        assert!(super::super::CanopyRegion::new(region, &profile).is_ok());
        let bad = Arc::new(super::super::PreparedRegion::build(vec![], [[0., 2.]; 2]).unwrap());
        assert!(super::super::CanopyRegion::new(bad, &profile).is_err());
    }
}
