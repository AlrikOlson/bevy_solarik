use super::*;
use std::io::Cursor;

#[test]
fn legacy_payload_does_not_consume_new_tangent_field() {
    let mut reader = Cursor::new([7u8; 16]);
    assert_eq!(
        &*read_vertex_tangents(&mut reader, 3, 2).unwrap(),
        &[Vec4::ZERO; 2]
    );
    assert_eq!(reader.position(), 0);
}

#[test]
fn version_four_preserves_authored_handedness() {
    let values = [Vec4::new(1.0, 0.0, 0.0, -1.0), Vec4::ZERO];
    let mut bytes = Vec::new();
    write_slice(&values, &mut bytes).unwrap();
    assert_eq!(
        &*read_vertex_tangents(&mut Cursor::new(bytes), 4, 2).unwrap(),
        &values
    );
}

#[test]
fn mismatched_count_and_nonfinite_tangents_fail_before_upload() {
    assert!(read_vertex_tangents(&mut Cursor::new(u64::MAX.to_le_bytes()), 4, 2).is_err());
    let mut bytes = Vec::new();
    write_slice(&[Vec4::splat(f32::NAN)], &mut bytes).unwrap();
    assert!(read_vertex_tangents(&mut Cursor::new(bytes), 4, 1).is_err());
}
