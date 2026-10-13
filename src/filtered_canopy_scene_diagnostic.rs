//! Offline query fixtures use the production scene bounds and hierarchy packer.
use super::*;

pub struct DiagnosticInstance {
    pub identity: u64,
    pub transform: Affine3A,
    pub base: u32,
    pub minimum: Vec3,
    pub maximum: Vec3,
}

fn item(value: &DiagnosticInstance) -> Result<Item, &'static str> {
    if !value.minimum.cmple(value.maximum).all() {
        return Err("inverted diagnostic canopy bounds");
    }
    let bounds = transformed_bounds(value.minimum, value.maximum, value.transform)
        .ok_or("nonrigid or nonfinite diagnostic canopy transform")?;
    let instance = Instance {
        id: Default::default(),
        transform: value.transform,
        previous: value.transform,
        identity: value.identity,
        rays: true,
    };
    Ok((instance, value.base, bounds))
}

/// Only packs metadata; callers own immutable rows and validate allocation limits.
pub fn diagnostic_scene_rows(
    values: &[DiagnosticInstance],
    frame: u32,
    mode: u32,
) -> Result<Vec<[u32; 4]>, &'static str> {
    if values.len() > 4096 || mode > 2 {
        return Err("diagnostic scene limit or mode");
    }
    let identities: alloc::collections::BTreeSet<_> = values.iter().map(|v| v.identity).collect();
    if identities.len() != values.len() {
        return Err("duplicate diagnostic scene identity");
    }
    let mut items: Vec<_> = values.iter().map(item).collect::<Result<_, _>>()?;
    let mut rows = pack(&mut items, frame);
    rows[1][1] = mode;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn instance() -> DiagnosticInstance {
        DiagnosticInstance {
            identity: 1,
            transform: Affine3A::IDENTITY,
            base: 0,
            minimum: -Vec3::ONE,
            maximum: Vec3::ONE,
        }
    }
    #[test]
    fn rejects_duplicate_owners_and_nonrigid_transforms() {
        assert!(diagnostic_scene_rows(&[instance(), instance()], 1, 0).is_err());
        let mut scaled = instance();
        scaled.transform = Affine3A::from_scale(Vec3::splat(2.));
        assert!(diagnostic_scene_rows(&[scaled], 1, 0).is_err());
    }
}
