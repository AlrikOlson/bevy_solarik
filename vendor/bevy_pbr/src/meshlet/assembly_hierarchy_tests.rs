//! Sparse publication work is bounded by changed roots, at the real forest scale.
use super::*;

fn owner(id: u64) -> SceneInstance {
    SceneInstance {
        entity: Entity::from_bits(id),
        part: 1,
    }
}
fn record(manager: &InstanceManager, key: SceneInstance) -> (u32, u32, Vec<u32>) {
    let allocation = &manager.hierarchy.allocations[&key];
    let group = manager.assembly_groups.get()[allocation.group as usize];
    let count = group.count & !(1 << 31);
    (
        allocation.group,
        group.first,
        manager.assembly_members.get()[group.first as usize..(group.first + count) as usize]
            .to_vec(),
    )
}
fn clear_dirty(manager: &mut InstanceManager) {
    manager.group_dirty_indices.clear();
    manager.member_dirty_indices.clear();
}

#[test]
fn partial_republication_shrink_growth_and_owner_reuse_keep_other_ranges() {
    let mut manager = InstanceManager::new();
    let bounds = Some(MeshletAabb::default());
    publish(&mut manager, owner(1), bounds, &[10, 11, 12]);
    publish(&mut manager, owner(2), bounds, &[20, 21, 22]);
    let retained = record(&manager, owner(2));
    clear_dirty(&mut manager);
    publish(&mut manager, owner(1), bounds, &[10, 11, 12]);
    assert!(manager.group_dirty_indices.is_empty() && manager.member_dirty_indices.is_empty());
    publish(&mut manager, owner(1), None, &[10, 12]);
    assert_eq!(record(&manager, owner(1)).2, [10, 12]);
    let group = manager.assembly_groups.get()[record(&manager, owner(1)).0 as usize];
    assert_eq!(group.count, (1 << 31) | 2, "pending roots fail open");
    publish(
        &mut manager,
        owner(1),
        bounds,
        &[10, 11, 12, 13, 14, 15, 16],
    );
    assert_eq!(record(&manager, owner(2)), retained);
    let old = record(&manager, owner(1));
    remove(&mut manager, owner(1));
    assert_eq!(manager.assembly_groups.get()[old.0 as usize].count, 0);
    publish(&mut manager, owner(3), bounds, &[30, 31, 32]);
    assert_eq!(
        record(&manager, owner(3)).0,
        old.0,
        "retired group is reused"
    );
    assert_eq!(record(&manager, owner(2)), retained);
    assert!(!manager.hierarchy.allocations.contains_key(&owner(1)));
}

#[test]
fn full_forest_sparse_churn_updates_only_changed_records_and_reuses_capacity() {
    const ROOTS: u32 = 262_144;
    const PARTS: u32 = 12;
    const CHANGED: u32 = 64;
    let mut manager = InstanceManager::new();
    for root in 0..ROOTS {
        let members: Vec<_> = (root * PARTS..(root + 1) * PARTS).collect();
        publish(
            &mut manager,
            owner(u64::from(root + 1)),
            Some(MeshletAabb::default()),
            &members,
        );
    }
    let original: Vec<_> = (0..ROOTS)
        .map(|root| {
            let a = &manager.hierarchy.allocations[&owner(u64::from(root + 1))];
            (a.group, a.members.start)
        })
        .collect();
    let capacity = manager.hierarchy.members.initial_range().end;
    let started = std::time::Instant::now();
    for cycle in 0..32u32 {
        clear_dirty(&mut manager);
        for changed in 0..CHANGED {
            let root = changed * (ROOTS / CHANGED);
            let old = if cycle == 0 {
                root + 1
            } else {
                ROOTS + (cycle - 1) * CHANGED + changed + 1
            };
            remove(&mut manager, owner(u64::from(old)));
        }
        for changed in 0..CHANGED {
            let key = ROOTS + cycle * CHANGED + changed + 1;
            let count = [1, 7, 12, 3][cycle as usize % 4];
            let parts: Vec<_> = (key * PARTS..key * PARTS + count).collect();
            publish(
                &mut manager,
                owner(u64::from(key)),
                Some(MeshletAabb::default()),
                &parts,
            );
        }
        assert_eq!(manager.hierarchy.allocations.len(), ROOTS as usize);
        assert!(manager.group_dirty_indices.len() <= 2 * CHANGED as usize);
        assert!(manager.member_dirty_indices.len() <= (CHANGED * PARTS) as usize);
        assert_eq!(manager.hierarchy.members.initial_range().end, capacity);
    }
    for root in 0..ROOTS {
        if root % (ROOTS / CHANGED) == 0 {
            continue;
        }
        let actual = record(&manager, owner(u64::from(root + 1)));
        assert_eq!((actual.0, actual.1), original[root as usize]);
        assert_eq!(
            actual.2,
            (root * PARTS..(root + 1) * PARTS).collect::<Vec<_>>()
        );
    }
    let groups = manager.assembly_groups.get().len();
    let members = manager.assembly_members.get().len();
    println!(
        "HIERARCHY_SPARSE roots={ROOTS} parts={} changed_per_cycle={CHANGED} cycles=32 max_member_writes={} elapsed_ms={:.3} groups={groups} members={members} capacity={capacity}",
        ROOTS * PARTS,
        CHANGED * PARTS,
        started.elapsed().as_secs_f64() * 1000.0
    );
    let owners: Vec<_> = manager.hierarchy.allocations.keys().copied().collect();
    for key in owners {
        remove(&mut manager, key);
    }
    assert!(
        manager
            .assembly_groups
            .get()
            .iter()
            .all(|group| group.count == 0)
    );
    for root in 0..1024 {
        publish(&mut manager, owner(root + 1), None, &[1, 2, 3]);
    }
    assert_eq!(manager.hierarchy.members.initial_range().end, capacity);
    assert_eq!(manager.assembly_groups.get().len(), groups);
    assert_eq!(manager.assembly_members.get().len(), members);
}
