use super::*;
fn facet(x: f32, z: f32, material: u32) -> Facet {
    Facet {
        corners: [
            Vec3::new(x, 0., z),
            Vec3::new(x + 1., 0., z),
            Vec3::new(x + 1., 1., z),
            Vec3::new(x, 1., z),
        ],
        surface: Surface {
            color: [0.2, 0.4, 0.1, 1.],
            normal: [0., 0., 1., 0.],
            surface: [0.5, 0., 0.5, 0.],
        },
        material,
    }
}
#[test]
fn empty_columns_stay_empty_and_interior_events_collapse() {
    let mut builder = Builder::new(0.5, 100).unwrap();
    for f in [
        facet(0., 1., 1),
        facet(0., 2., 2),
        facet(0., 3., 3),
        facet(2., 2., 4),
    ] {
        builder.insert(f).unwrap();
    }
    let patches = builder.finish().unwrap();
    assert_eq!(patches.len(), 12);
    assert!(
        patches
            .iter()
            .all(|p| p.corners[0].x < 1. || p.corners[0].x >= 2.)
    );
    assert!(!patches.iter().any(|p| p.material == 2));
    assert_eq!(patches.iter().filter(|p| p.material == 1).count(), 4);
    assert_eq!(patches.iter().filter(|p| p.material == 3).count(), 4);
}
#[test]
fn oblique_source_depth_and_winding_survive_sampling() {
    let mut source = facet(0., 0., 9);
    for p in &mut source.corners {
        p.z = 0.3 * p.x + 0.2 * p.y;
    }
    for reversed in [false, true] {
        let mut input = source;
        if reversed {
            input.corners.swap(1, 3);
        }
        let mut b = Builder::new(0.25, 100).unwrap();
        b.insert(input).unwrap();
        let patches = b.finish().unwrap();
        assert_eq!(patches.len(), 16);
        for p in patches {
            let centre = (p.corners[0] + p.corners[2]) * 0.5;
            assert!((centre.z - 0.3 * centre.x - 0.2 * centre.y).abs() < 1e-6);
            let n = (p.corners[1] - p.corners[0]).cross(p.corners[2] - p.corners[0]);
            assert_eq!(n.z < 0., reversed);
        }
    }
}
#[test]
fn failed_cooks_cannot_publish_partial_coverage() {
    assert!(Builder::new(f32::NAN, 10).is_err());
    let mut b = Builder::new(0.5, 2).unwrap();
    assert!(b.insert(facet(0., 1., 0)).is_err());
    assert!(b.finish().is_err());
}
#[test]
fn rebased_lattice_is_deterministic_for_source_order_and_duplicate_triangles() {
    let sources = [facet(0., 1., 1), facet(0., 3., 3), facet(2., 2., 4)];
    let cook = |sources: Vec<Facet>| {
        let mut b = Builder::new(0.5, 100).unwrap();
        for f in sources {
            b.insert(f).unwrap();
        }
        b.finish()
            .unwrap()
            .into_iter()
            .map(|p| {
                (
                    p.corners.map(|v| v.to_array().map(f32::to_bits)),
                    p.material,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        cook(sources.to_vec()),
        cook(sources.into_iter().rev().collect())
    );
}
