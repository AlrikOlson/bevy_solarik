//! Extract only changed shared roots; retain other source parts by stable identity.
use super::super::{MeshletAssembly3d, MeshletAssemblyPart};
use super::*;
use alloc::sync::Arc;
use bevy_math::Affine3A;
use bevy_render::scene_slots::SceneInstance;

pub(super) type AssemblyRows<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static MeshletAssembly3d,
        &'static GlobalTransform,
        Option<&'static PreviousGlobalTransform>,
        Option<&'static RenderLayers>,
        Has<NotShadowReceiver>,
        Has<NotShadowCaster>,
    ),
>;
type Row<'a> = (
    Entity,
    &'a MeshletAssembly3d,
    &'a GlobalTransform,
    Option<&'a PreviousGlobalTransform>,
    Option<&'a RenderLayers>,
    bool,
    bool,
);
pub(super) struct AssemblyInput {
    pub(super) parts: Arc<[MeshletAssemblyPart]>,
    transform: Affine3A,
    previous: Affine3A,
    layers: RenderLayers,
    flags: (bool, bool),
    pub(super) slots: Vec<SceneSlot>,
    pub(super) ready: bool,
}
impl AssemblyInput {
    fn new(row: Row<'_>) -> Self {
        Self {
            parts: row.1.0.clone(),
            transform: row.2.affine(),
            previous: row.3.map_or(row.2.affine(), |p| p.0),
            layers: row.4.cloned().unwrap_or_default(),
            flags: (row.5, row.6),
            slots: Vec::with_capacity(row.1.0.len()),
            ready: true,
        }
    }
    fn matches(&self, row: Row<'_>) -> bool {
        Arc::ptr_eq(&self.parts, &row.1.0)
            && self.transform == row.2.affine()
            && self.previous == row.3.map_or(row.2.affine(), |p| p.0)
            && self.layers == *row.4.unwrap_or_default()
            && self.flags == (row.5, row.6)
    }
}
pub(super) fn targets(
    manager: &InstanceManager,
    rows: &AssemblyRows<'_, '_>,
    mut changed: HashSet<Entity>,
    force: bool,
) -> Vec<Entity> {
    // Account for known arrivals/departures before deciding that the receipt
    // needs a full reconciliation. Ordinary streaming remains a root delta.
    let expected = changed
        .iter()
        .fold(manager.assembly_inputs.len(), |count, root| {
            count - usize::from(manager.assembly_inputs.contains_key(root))
                + usize::from(rows.contains(*root))
        });
    if force || rows.iter().len() != expected {
        changed.extend(manager.assembly_inputs.keys().copied());
        changed.extend(rows.iter().map(|row| row.0));
    }
    changed.extend(manager.assembly_pending.iter().copied());
    let mut roots: Vec<_> = changed
        .into_iter()
        .filter(|root| manager.assembly_inputs.contains_key(root) || rows.contains(*root))
        .collect();
    roots.sort_unstable_by_key(|root| root.to_bits());
    roots
}
type Prepared = (u32, MeshletAabb, u32, Vec4);
struct Sources<'a> {
    server: &'a AssetServer,
    assets: &'a mut Assets<MeshletMesh>,
    meshes: &'a mut MeshletMeshManager,
    bindings: &'a RenderMaterialBindings,
    atlas: &'a MeshletCutoutAtlas,
    images: &'a Assets<Image>,
    geometry: HashMap<AssetId<MeshletMesh>, Option<(u32, MeshletAabb, u32)>>,
    alpha: HashMap<(Option<(u32, u32)>, bool), Option<Vec4>>,
}
impl Sources<'_> {
    fn prepare(&mut self, part: &MeshletAssemblyPart) -> Option<Prepared> {
        if !self.bindings.contains_key(&part.material.id().untyped()) {
            return None;
        }
        let key = (
            part.cutout.as_ref().map(|c| (c.layer, c.cutoff.to_bits())),
            part.double_sided,
        );
        let alpha = *self.alpha.entry(key).or_insert_with(|| {
            metadata(part.cutout.as_ref(), self.atlas, self.images).map(|mut m| {
                m.z = f32::from(part.double_sided);
                m
            })
        });
        let (root, bounds, depth) = (*self.geometry.entry(part.mesh.id()).or_insert_with(|| {
            if self.server.is_managed(part.mesh.id())
                && !self.server.is_loaded_with_dependencies(part.mesh.id())
            {
                return None;
            }
            Some(
                self.meshes
                    .queue_upload_if_needed(part.mesh.id(), self.assets),
            )
        }))?;
        Some((root, bounds, depth, alpha?))
    }
}
#[allow(clippy::too_many_arguments)]
pub(super) fn extract(
    manager: &mut InstanceManager,
    rows: &AssemblyRows<'_, '_>,
    roots: &[Entity],
    force: bool,
    server: &AssetServer,
    assets: &mut Assets<MeshletMesh>,
    meshes: &mut MeshletMeshManager,
    bindings: &RenderMaterialBindings,
    atlas: &MeshletCutoutAtlas,
    images: &Assets<Image>,
) -> (usize, usize, Vec<(SceneInstance, SceneSlot)>) {
    let mut sources = Sources {
        server,
        assets,
        meshes,
        bindings,
        atlas,
        images,
        geometry: HashMap::default(),
        alpha: HashMap::default(),
    };
    let mut counts = (0, 0);
    let mut removed = Vec::new();
    for &root in roots {
        let delta = update_root(
            manager,
            root,
            rows.get(root).ok(),
            force,
            bindings,
            &mut |part| sources.prepare(part),
            &mut removed,
        );
        counts.0 += delta.0;
        counts.1 += delta.1;
    }
    for (name, value) in [
        ("meshlet.assembly_roots", manager.assembly_inputs.len()),
        ("meshlet.assembly_roots_visited", roots.len()),
        ("meshlet.assembly_parts_visited", counts.1),
    ] {
        bevy_render::diagnostic::profile_value(name, value as f64, "count");
    }
    (counts.0, counts.1, removed)
}
fn retire_tail(
    entity: Entity,
    old: &AssemblyInput,
    keep: usize,
    removed: &mut Vec<(SceneInstance, SceneSlot)>,
) {
    removed.extend(
        old.slots
            .iter()
            .enumerate()
            .skip(keep)
            .map(|(part, &slot)| {
                (
                    SceneInstance {
                        entity,
                        part: part as u32 + 1,
                    },
                    slot,
                )
            }),
    );
}
#[allow(clippy::too_many_arguments)]
fn update_root(
    manager: &mut InstanceManager,
    root: Entity,
    row: Option<Row<'_>>,
    force: bool,
    bindings: &RenderMaterialBindings,
    prepare: &mut impl FnMut(&MeshletAssemblyPart) -> Option<Prepared>,
    removed: &mut Vec<(SceneInstance, SceneSlot)>,
) -> (usize, usize) {
    let old = manager.assembly_inputs.remove(&root);
    let old_count = old.as_ref().map_or(0, |old| old.slots.len());
    let Some(row) = row else {
        if let Some(old) = old {
            retire_tail(root, &old, 0, removed);
        }
        manager.assembly_part_count -= old_count;
        manager.assembly_pending.remove(&root);
        return (0, 0);
    };
    if let Some(old) = old {
        if !force && old.ready && old.matches(row) {
            manager.assembly_inputs.insert(root, old);
            return (0, 0);
        }
        retire_tail(root, &old, row.1.0.len(), removed);
    }
    let mut input = AssemblyInput::new(row);
    let counts = publish(manager, &mut input, root, bindings, prepare);
    manager.assembly_part_count = manager.assembly_part_count - old_count + input.slots.len();
    manager.assembly_pending.remove(&root);
    if !input.ready {
        manager.assembly_pending.insert(root);
    }
    manager.assembly_inputs.insert(root, input);
    counts
}
fn publish(
    manager: &mut InstanceManager,
    input: &mut AssemblyInput,
    entity: Entity,
    bindings: &RenderMaterialBindings,
    prepare: &mut impl FnMut(&MeshletAssemblyPart) -> Option<Prepared>,
) -> (usize, usize) {
    let mut added = 0;
    for (index, part) in input.parts.iter().enumerate() {
        let slot = manager.slots.touch_part(SceneInstance {
            entity,
            part: index as u32 + 1,
        });
        input.slots.push(slot);
        let Some((root, aabb, depth, cutout)) = prepare(part) else {
            manager.deactivate_slot(slot);
            input.ready = false;
            continue;
        };
        added += usize::from(!manager.slots.is_active(slot));
        let transform = GlobalTransform::from(input.transform * part.transform);
        let previous = PreviousGlobalTransform(input.previous * part.transform);
        manager.add_material_instance(
            slot,
            entity.into(),
            root,
            aabb,
            depth,
            &transform,
            Some(&previous),
            Some(&input.layers),
            part.material.id().untyped(),
            bindings,
            input.flags.0,
            input.flags.1,
            cutout,
        );
    }
    (added, input.slots.len())
}
#[cfg(test)]
#[path = "assembly_extract_tests.rs"]
mod tests;
