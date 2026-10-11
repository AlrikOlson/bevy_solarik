#![expect(
    clippy::print_stdout,
    reason = "Explicit diagnostics emit publication timing receipts."
)]
use super::super::*;
use core::hint::black_box;
use std::time::Instant;

#[derive(Default)]
struct Records {
    transforms: Vec<Mat4>,
    previous: Vec<Mat4>,
    geometry: Vec<GpuInstanceGeometryIds>,
    materials: Vec<u32>,
}
impl Records {
    fn new(count: usize) -> Self {
        Self {
            transforms: vec![Mat4::IDENTITY; count],
            previous: vec![Mat4::IDENTITY; count],
            geometry: vec![GpuInstanceGeometryIds::default(); count],
            materials: vec![0; count],
        }
    }
    fn same_as(&self, reference: &Self) {
        assert_eq!(self.transforms, reference.transforms);
        assert_eq!(self.previous, reference.previous);
        assert!(self.geometry == reference.geometry);
        assert_eq!(self.materials, reference.materials);
    }
}

fn reference(
    records: &mut Records,
    inputs: &[RayInstanceInput],
    slots: &bevy_render::scene_slots::SceneSlots,
    sources: &[(GpuInstanceGeometryIds, u32)],
) -> (Vec<u32>, Vec<u32>) {
    let mut dirty = (Vec::new(), Vec::new());
    for (index, input) in inputs.iter().enumerate() {
        let slot = slots.get_part(input.entity).unwrap();
        let (geometry, material) = &sources[input.entity.part as usize];
        records.transforms[index] = Mat4::from(input.transform);
        records.previous[index] = Mat4::from(input.previous);
        if records.geometry[index] != *geometry {
            dirty.0.push(slot.index);
        }
        if records.materials[index] != *material {
            dirty.1.push(slot.index);
        }
        records.geometry[index] = geometry.clone();
        records.materials[index] = *material;
    }
    dirty
}

fn retained(
    records: &mut Records,
    inputs: &[RayInstanceInput],
    slots: &bevy_render::scene_slots::SceneSlots,
    sources: &[(GpuInstanceGeometryIds, u32)],
    dirty: &[u32],
    initialize: bool,
) -> (Vec<u32>, Vec<u32>) {
    let transforms = super::super::super::tlas::DirtySlots::new(inputs.len(), dirty);
    let mut changed = (Vec::new(), Vec::new());
    for (index, input) in inputs.iter().enumerate() {
        let slot = slots.get_part_at(index as u32, input.entity).unwrap();
        let (geometry, material) = &sources[input.entity.part as usize];
        let row = super::write(
            index,
            input,
            (geometry, *material),
            (
                &mut records.transforms,
                &mut records.previous,
                &mut records.geometry,
                &mut records.materials,
            ),
            initialize || transforms.contains(slot.index),
        );
        if row.geometry {
            changed.0.push(slot.index);
        }
        if row.material {
            changed.1.push(slot.index);
        }
    }
    changed
}

fn fixture(count: usize) -> (bevy_render::scene_slots::SceneSlots, Vec<RayInstanceInput>) {
    let mut slots = bevy_render::scene_slots::SceneSlots::default();
    slots.begin();
    let inputs = (0..count)
        .map(|index| {
            let entity = SceneInstance {
                entity: Entity::from_raw_u32(index as u32 / 3 + 1).unwrap(),
                part: index as u32 % 3,
            };
            assert_eq!(slots.touch_part(entity).index as usize, index);
            RayInstanceInput {
                entity,
                mesh: Handle::<Mesh>::default().id(),
                material: Handle::<StandardMaterial>::default().id(),
                transform: Affine3A::from_scale_rotation_translation(
                    Vec3::new(1.0, 2.0, 3.0),
                    bevy_math::Quat::from_rotation_y(0.25),
                    Vec3::new(index as f32, 17.0, -31.0),
                ),
                previous: Affine3A::IDENTITY,
            }
        })
        .collect();
    (slots, inputs)
}

#[test]
fn full_forest_rows_match_dense_reference_through_remaps_motion_and_settling() {
    const COUNT: usize = 262_144 * 3;
    let (slots, mut inputs) = fixture(COUNT);
    let mut expected = Records::new(COUNT);
    let mut actual = Records::new(COUNT);
    let mut sources: Vec<_> = (0..3u32)
        .map(|part| {
            (
                GpuInstanceGeometryIds {
                    vertex_buffer_id: part + 1,
                    index_buffer_id: 7,
                    vertex_buffer_offset: part * 10,
                    index_buffer_offset: part * 30,
                    triangle_count: 10,
                    light_probability: 0.0,
                },
                part + 1,
            )
        })
        .collect();
    for phase in 0..5 {
        let dirty = alter(phase, &mut inputs, &mut sources);
        let start = Instant::now();
        let reference_dirty = reference(&mut expected, black_box(&inputs), &slots, &sources);
        let reference_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let actual_dirty = retained(
            &mut actual,
            black_box(&inputs),
            &slots,
            &sources,
            &dirty,
            phase == 0,
        );
        let retained_ms = start.elapsed().as_secs_f64() * 1000.0;
        actual.same_as(&expected);
        assert_eq!(actual_dirty, reference_dirty);
        println!(
            "full publication rows={COUNT} phase={phase} input_changes={} reference_ms={reference_ms:.3} retained_ms={retained_ms:.3}",
            dirty.len()
        );
    }
}

fn alter(
    phase: usize,
    inputs: &mut [RayInstanceInput],
    sources: &mut [(GpuInstanceGeometryIds, u32)],
) -> Vec<u32> {
    if phase == 1 {
        sources[0].1 = 3;
        sources[2].1 = 1;
        sources[1].0.vertex_buffer_id = 9;
    }
    let changed: Vec<_> = if matches!(phase, 2 | 3) {
        (0..inputs.len())
            .step_by(1024)
            .map(|index| index as u32)
            .collect()
    } else {
        Vec::new()
    };
    for &index in &changed {
        let input = &mut inputs[index as usize];
        input.previous = input.transform;
        if phase == 2 {
            input.transform.translation += bevy_math::Vec3A::new(2.0, 3.0, -7.0);
        }
    }
    changed
}

#[test]
fn binding_only_republication_preserves_matrix_bytes_until_source_is_ready() {
    let (_, mut inputs) = fixture(1);
    let mut rows = Records::new(1);
    let geometry = GpuInstanceGeometryIds {
        triangle_count: 9,
        ..Default::default()
    };
    let write = |rows: &mut Records, input: &RayInstanceInput, changed| {
        super::write(
            0,
            input,
            (&geometry, 3),
            (
                &mut rows.transforms,
                &mut rows.previous,
                &mut rows.geometry,
                &mut rows.materials,
            ),
            changed,
        )
    };
    assert_eq!(
        write(&mut rows, &inputs[0], true),
        super::Changes {
            geometry: true,
            material: true
        }
    );
    let published = rows.transforms[0];
    inputs[0].previous = inputs[0].transform;
    inputs[0].transform = Affine3A::from_translation(Vec3::new(-9.0, 2.0, 7.0));
    assert_eq!(
        write(&mut rows, &inputs[0], false),
        super::Changes::default()
    );
    assert_eq!(rows.transforms[0], published);
    assert_eq!(
        write(&mut rows, &inputs[0], true),
        super::Changes::default()
    );
    assert_eq!(rows.transforms[0], Mat4::from(inputs[0].transform));
    assert_eq!(rows.previous[0], published);
}

#[test]
fn final_emitter_probability_marks_geometry_without_material_or_transform_changes() {
    let (slots, inputs) = fixture(1);
    let mut rows = Records::new(1);
    let sources = vec![(GpuInstanceGeometryIds::default(), 7)];
    retained(&mut rows, &inputs, &slots, &sources, &[0], true);
    assert_eq!(
        retained(&mut rows, &inputs, &slots, &sources, &[], false),
        (vec![], vec![])
    );
    let before = rows.transforms.clone();
    let mut dirty = Vec::new();
    super::set_light_probability(0, 0.75, &mut rows.geometry, &mut dirty);
    assert_eq!(dirty, [0]);
    assert_eq!(rows.geometry[0].light_probability, 0.75);
    assert_eq!(rows.materials, [7]);
    assert_eq!(rows.transforms, before);
    dirty.clear();
    super::set_light_probability(0, 0.75, &mut rows.geometry, &mut dirty);
    assert!(dirty.is_empty());
    super::set_light_probability(0, 0.0, &mut rows.geometry, &mut dirty);
    assert_eq!(dirty, [0]);
}
