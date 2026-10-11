use super::*;
use std::{hint::black_box, time::Instant};

fn bank(count: usize) -> (InstanceManager, Vec<SceneSlot>, Vec<UntypedAssetId>) {
    let mut assets = Assets::<crate::StandardMaterial>::default();
    let handles: Vec<_> = (0..8)
        .map(|_| assets.add(crate::StandardMaterial::default()))
        .collect();
    let ids: Vec<_> = handles.iter().map(|h| h.id().untyped()).collect();
    let mut manager = InstanceManager::new();
    manager.slots.begin();
    let slots: Vec<_> = (1..=count as u64)
        .map(|i| manager.slots.touch(Entity::from_bits(i)))
        .collect();
    manager
        .instance_material_assets
        .resize(manager.slots.capacity(), ids[0]);
    manager.bvh_depths.resize(manager.slots.capacity(), 1);
    manager
        .instance_material_ids
        .get_mut()
        .resize(manager.slots.capacity(), 0);
    manager.get_material_id(ids[0]);
    manager.get_material_id(ids[1]);
    for &slot in &slots {
        manager.update_slot_metadata(slot, ids[0], 1);
        manager.slots.activate(slot);
        manager.dirty_indices.push(slot.index);
    }
    manager.refresh_material_ids();
    manager.dirty_indices.clear();
    (manager, slots, ids)
}

fn assign(manager: &mut InstanceManager, slot: SceneSlot, asset: UntypedAssetId) {
    manager.update_slot_metadata(slot, asset, 1);
    manager.instance_material_assets[slot.index as usize] = asset;
    manager.dirty_indices.push(slot.index);
}

fn compare_dense(manager: &InstanceManager) {
    let mut present = HashSet::default();
    let mut unresolved = 0;
    for &index in manager.slots.active_indices() {
        let expected = manager
            .material_id_lookup
            .get(&manager.instance_material_assets[index as usize])
            .copied()
            .unwrap_or(0);
        assert_eq!(
            manager.instance_material_ids.get()[index as usize],
            expected
        );
        if expected == 0 {
            unresolved += 1;
        } else {
            present.insert(expected);
        }
    }
    assert_eq!(manager.unresolved_material_instances, unresolved);
    assert_eq!(manager.material_ids_present_in_scene, present);
}

#[test]
fn unrelated_mapping_does_not_visit_resolved_rows() {
    let (mut manager, _, ids) = bank(1024);
    manager.get_material_id(ids[2]);
    assert_eq!(manager.refresh_material_ids(), 0);
    compare_dense(&manager);
}

#[test]
fn pending_mapping_tracks_source_changes_and_readiness_reentry() {
    let (mut manager, slots, ids) = bank(1024);
    assign(&mut manager, slots[7], ids[2]);
    assign(&mut manager, slots[8], ids[3]);
    manager.refresh_material_ids();
    compare_dense(&manager);
    manager.dirty_indices.clear();
    manager.deactivate_slot(slots[7]);
    assign(&mut manager, slots[8], ids[1]);
    manager.get_material_id(ids[2]);
    manager.get_material_id(ids[3]);
    assert_eq!(manager.refresh_material_ids(), 1);
    compare_dense(&manager);
    manager.dirty_indices.clear();
    // The retained address returns with a different material; its old pending
    // mapping must never override the new authoritative source.
    assign(&mut manager, slots[7], ids[4]);
    manager.slots.activate(slots[7]);
    manager.refresh_material_ids();
    compare_dense(&manager);
    manager.dirty_indices.clear();
    manager.get_material_id(ids[4]);
    assert_eq!(manager.refresh_material_ids(), 1);
    compare_dense(&manager);
}

#[test]
#[ignore = "Full three-million-row release material queue diagnostic"]
fn full_forest_material_mapping_cost() {
    let (mut manager, slots, ids) = bank(3_000_000);
    for generation in 0..3 {
        assign(&mut manager, slots[17], ids[generation + 2]);
        manager.refresh_material_ids();
        manager.dirty_indices.clear();
        for offset in 0..64 {
            assign(
                &mut manager,
                slots[100 + offset * 40000],
                ids[generation % 2],
            );
        }
        manager.get_material_id(ids[generation + 2]);
        let start = Instant::now();
        let visited = black_box(manager.refresh_material_ids());
        let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
        compare_dense(&manager);
        eprintln!(
            "material queue generation={generation} rows=3000000 visited={visited} milliseconds={milliseconds:.6}"
        );
        manager.dirty_indices.clear();
    }
}
