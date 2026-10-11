use super::*;
fn cell(x: f32, y: f32, z: f32, identity: u64) -> CellInput {
    CellInput {
        identity,
        cell: [
            [1., 0., 0., 0.],
            [0., 1., 0., -y],
            [0., 0., 1., 0.],
            [x - 0.5, z - 0.5, x + 0.5, z - 0.5],
            [x + 0.5, z + 0.5, x - 0.5, z + 0.5],
            [1., 2., 0., 0.],
            [0., 0., 0., 0.],
        ],
    }
}
#[test]
fn stable_identity_and_preorder_escape_preserve_every_cell() {
    let inputs: Vec<_> = (0..128)
        .map(|i| cell((i % 16) as f32 * 2., 0., (i / 16) as f32 * 2., i))
        .collect();
    let region = PreparedRegion::build(inputs.clone(), [[0., 1.]; 2]).unwrap();
    let mut reversed = inputs;
    reversed.reverse();
    let other = PreparedRegion::build(reversed, [[0., 1.]; 2]).unwrap();
    assert_eq!(region.rows(), other.rows());
    assert_eq!(region.bytes(), 16 + (128 * 2 - 1) * 32 + 128 * 128);
    let rows = region.rows();
    let header = rows[0];
    let mut leaves = BTreeSet::new();
    for i in 0..header[0] as usize {
        let low = rows[1 + 2 * i];
        let high = rows[2 + 2 * i];
        assert!(low[3] as usize > i && low[3] <= header[0]);
        if high[3] == u32::MAX {
            continue;
        }
        assert_eq!(low[3], i as u32 + 1);
        let address = header[2] as usize + high[3] as usize * CELL_ROWS;
        assert!(leaves.insert(rows[address + 7][0]));
        let c: Cell = core::array::from_fn(|r| rows[address + r].map(f32::from_bits));
        for p in footprint(&c).unwrap() {
            for y in [0., 1.] {
                let point = Vec3::new(p.x, y, p.y);
                let lower = Vec3::from_array([low[0], low[1], low[2]].map(f32::from_bits));
                let upper = Vec3::from_array([high[0], high[1], high[2]].map(f32::from_bits));
                assert!(point.cmpge(lower).all() && point.cmple(upper).all());
            }
        }
    }
    assert_eq!(leaves.len(), 128);
}
#[test]
fn bounds_follow_tilted_height_planes_without_lateral_gaps() {
    let mut c = cell(10., 2., 20., 7);
    let rotation = Mat3::from_rotation_z(0.8);
    c.cell[0] = rotation.x_axis.extend(0.).to_array();
    c.cell[1] = rotation
        .y_axis
        .extend(-rotation.y_axis.dot(Vec3::new(10., 2., 20.)))
        .to_array();
    c.cell[2] = rotation.z_axis.extend(0.).to_array();
    c.cell[5] = [15., 0.5, 0., 0.];
    let b = cell_bounds(&c.cell, [[0., 1.]; 2]).unwrap().unwrap();
    for x in 0..17 {
        for z in 0..17 {
            for h in [0., 15.] {
                let px = 9.5 + x as f32 / 16.;
                let pz = 19.5 + z as f32 / 16.;
                let y = (h - c.cell[1][3] - rotation.y_axis.x * px) / rotation.y_axis.y;
                let p = Vec3::new(px, y, pz);
                assert!(p.cmpge(b.low).all() && p.cmple(b.high).all());
            }
        }
    }
}
#[test]
fn invalid_sources_cells_and_duplicate_identity_fail_closed() {
    let valid = cell(0., 0., 0., 3);
    assert!(PreparedRegion::build(vec![valid.clone(); 2], [[0., 1.]; 2]).is_err());
    assert!(PreparedRegion::build(vec![valid.clone()], [[1., 1.]; 2]).is_err());
    assert!(PreparedRegion::build(vec![valid.clone(); MAX_CELLS + 1], [[0., 1.]; 2]).is_err());
    for (row, col, value) in [
        (0, 0, 2.),
        (1, 1, 0.),
        (3, 0, f32::NAN),
        (5, 0, 0.),
        (5, 1, -1.),
        (6, 2, 1.),
    ] {
        let mut bad = valid.clone();
        bad.cell[row][col] = value;
        assert!(PreparedRegion::build(vec![bad], [[0., 1.]; 2]).is_err());
    }
    let empty = PreparedRegion::build(vec![], [[0., 1.]; 2]).unwrap();
    assert_eq!(empty.rows(), &[[0, 1, 1, 0]]);
    let mut zero = valid;
    zero.cell[5][1] = 0.;
    assert_eq!(
        PreparedRegion::build(vec![zero], [[0., 1.]; 2])
            .unwrap()
            .cells,
        0
    );
}
