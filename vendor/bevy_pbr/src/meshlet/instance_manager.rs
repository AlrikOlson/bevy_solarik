use super::{MeshletCutoutAtlas, MeshletDoubleSided, MeshletVisibilityCutout, cutout::metadata};
use super::{MeshletMesh, MeshletMesh3d, meshlet_mesh_manager::MeshletMeshManager};
use crate::DUMMY_MESH_MATERIAL;
use crate::{
    MaterialBindingId, MeshFlags, MeshTransforms, MeshUniform, PreviousGlobalTransform,
    RenderMaterialBindings, RenderMaterialInstances, meshlet::asset::MeshletAabb,
};
use bevy_asset::{AssetEvent, AssetId, AssetServer, Assets, UntypedAssetId};
use bevy_camera::visibility::RenderLayers;
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::{Entities, Entity, EntityHashMap},
    message::MessageReader,
    query::Has,
    resource::Resource,
    system::{Local, Query, Res, ResMut, SystemState},
};
use bevy_image::Image;
use bevy_light::{NotShadowCaster, NotShadowReceiver};
use bevy_math::Vec4;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{
    MainWorld,
    render_resource::StorageBuffer,
    renderer::RenderQueue,
    scene_slots::{SceneInstance, SceneSlot, SceneSlots},
    sync_world::MainEntity,
};
use bevy_transform::components::GlobalTransform;
use core::ops::DerefMut;
#[path = "assembly_changes.rs"]
mod assembly_changes;
#[path = "assembly_extract.rs"]
mod assembly_extract;
#[path = "assembly_hierarchy.rs"]
mod assembly_hierarchy;
#[path = "scene_metadata.rs"]
mod scene_metadata;

/// Persistent raw scene receipt, including temporarily suppressed instances.
/// A warm frame compares these values without resolving materials, cloning
/// layers, validating shared alpha images, or constructing another scene vector.
#[derive(Clone, PartialEq)]
pub(crate) struct InstanceInput {
    entity: Entity,
    mesh: AssetId<MeshletMesh>,
    transform: [u32; 12],
    previous: [u32; 12],
    slot: SceneSlot,
    layers: RenderLayers,
    not_shadow_receiver: bool,
    not_shadow_caster: bool,
    cutout: Option<(u32, u32)>,
    double_sided: bool,
}

type InstanceRow<'a> = (
    Entity,
    &'a MeshletMesh3d,
    &'a GlobalTransform,
    Option<&'a PreviousGlobalTransform>,
    Option<&'a RenderLayers>,
    bool,
    bool,
    Option<&'a MeshletVisibilityCutout>,
    bool,
);

impl InstanceInput {
    fn from_row(row: InstanceRow<'_>, slot: SceneSlot) -> Self {
        let (entity, mesh, transform, previous, layers, receiver, caster, cutout, double_sided) =
            row;
        Self {
            entity,
            mesh: mesh.id(),
            transform: transform.affine().to_cols_array().map(f32::to_bits),
            previous: previous
                .map_or(transform.affine(), |t| t.0)
                .to_cols_array()
                .map(f32::to_bits),
            slot,
            layers: layers.cloned().unwrap_or_default(),
            not_shadow_receiver: receiver,
            not_shadow_caster: caster,
            cutout: cutout.map(|c| (c.layer, c.cutoff.to_bits())),
            double_sided,
        }
    }

    fn matches(&self, row: InstanceRow<'_>) -> bool {
        let (entity, mesh, transform, previous, layers, receiver, caster, cutout, double_sided) =
            row;
        self.entity == entity
            && self.mesh == mesh.id()
            && self.transform == transform.affine().to_cols_array().map(f32::to_bits)
            && self.previous
                == previous
                    .map_or(transform.affine(), |t| t.0)
                    .to_cols_array()
                    .map(f32::to_bits)
            && self.layers == *layers.unwrap_or_default()
            && self.not_shadow_receiver == receiver
            && self.not_shadow_caster == caster
            && self.cutout == cutout.map(|c| (c.layer, c.cutoff.to_bits()))
            && self.double_sided == double_sided
    }
}

/// Manages data for each entity with a [`MeshletMesh`].
#[derive(Resource)]
pub struct InstanceManager {
    /// Amount of instances in the scene.
    pub scene_instance_count: u32,
    /// The max BVH depth of any instance in the scene. This is used to control the number of
    /// dependent dispatches emitted for BVH traversal.
    pub max_bvh_depth: u32,

    /// Per-instance [`MainEntity`], [`RenderLayers`], and [`NotShadowCaster`].
    pub instances: Vec<Option<(MainEntity, RenderLayers, bool)>>,
    /// Per-instance [`MeshUniform`].
    pub instance_uniforms: StorageBuffer<Vec<MeshUniform>>,
    /// Per-instance model-space AABB.
    pub instance_aabbs: StorageBuffer<Vec<MeshletAabb>>,
    /// Per-instance material ID.
    pub instance_material_ids: StorageBuffer<Vec<u32>>,
    /// Per-instance index to the root node of the instance's BVH.
    pub instance_bvh_root_nodes: StorageBuffer<Vec<u32>>,
    pub instance_cutouts: StorageBuffer<Vec<Vec4>>,
    /// Per-view per-instance visibility bit. Used for [`RenderLayers`] and [`NotShadowCaster`] support.
    pub view_instance_visibility: EntityHashMap<StorageBuffer<Vec<u32>>>,

    // Exact receipts follow ECS query order for sequential stationary reads.
    inputs: Vec<Option<InstanceInput>>,
    query_slots: Vec<SceneSlot>,
    input_positions: Vec<usize>,
    query_publication: u64,
    assembly_inputs: EntityHashMap<assembly_extract::AssemblyInput>,
    assembly_part_count: usize,
    assembly_pending: HashSet<Entity>,
    material_receipt: (u64, u64),
    material_changes: Vec<bool>,
    binding_receipt: HashMap<UntypedAssetId, u32>,
    pub(crate) slots: SceneSlots,
    pub active_indices: StorageBuffer<Vec<u32>>,
    pub assembly_groups: StorageBuffer<Vec<assembly_hierarchy::AssemblyGroup>>,
    pub assembly_members: StorageBuffer<Vec<u32>>,
    pub(crate) group_dirty_indices: Vec<u32>,
    pub(crate) member_dirty_indices: Vec<u32>,
    pub(crate) dirty_indices: Vec<u32>,
    pub(crate) active_dirty_indices: Vec<u32>,
    binding_slots: Vec<u32>,
    bvh_depths: Vec<u32>,
    instance_material_assets: Vec<UntypedAssetId>,
    /// Includes shared parts without a standalone MeshMaterial3d owner.
    scene_metadata: scene_metadata::SceneMetadata,
    unresolved_material_instances: usize,
    /// Persistent instance buffers only need publication after exact scene changes.
    pub instance_upload_dirty: bool,
    pub rebuild_next_extract: bool,
    material_mapping_dirty: bool,

    /// Next material ID available.
    next_material_id: u32,
    /// Map of material asset to material ID.
    material_id_lookup: HashMap<UntypedAssetId, u32>,
    /// Set of material IDs used in the scene.
    material_ids_present_in_scene: HashSet<u32>,
}

impl InstanceManager {
    pub fn new() -> Self {
        Self {
            scene_instance_count: 0,
            max_bvh_depth: 0,

            instances: Vec::new(),
            instance_uniforms: {
                let mut buffer = StorageBuffer::default();
                #[cfg(feature = "graphics_debug")]
                buffer.add_usages(bevy_render::render_resource::BufferUsages::COPY_SRC);
                buffer.set_label(Some("meshlet_instance_uniforms"));
                buffer
            },
            instance_aabbs: {
                let mut buffer = StorageBuffer::default();
                #[cfg(feature = "graphics_debug")]
                buffer.add_usages(bevy_render::render_resource::BufferUsages::COPY_SRC);
                buffer.set_label(Some("meshlet_instance_aabbs"));
                buffer
            },
            instance_material_ids: {
                let mut buffer = StorageBuffer::default();
                #[cfg(feature = "graphics_debug")]
                buffer.add_usages(bevy_render::render_resource::BufferUsages::COPY_SRC);
                buffer.set_label(Some("meshlet_instance_material_ids"));
                buffer
            },
            instance_bvh_root_nodes: {
                let mut buffer = StorageBuffer::default();
                #[cfg(feature = "graphics_debug")]
                buffer.add_usages(bevy_render::render_resource::BufferUsages::COPY_SRC);
                buffer.set_label(Some("meshlet_instance_bvh_root_nodes"));
                buffer
            },
            view_instance_visibility: EntityHashMap::default(),
            instance_cutouts: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_cutouts"));
                #[cfg(feature = "graphics_debug")]
                buffer.add_usages(bevy_render::render_resource::BufferUsages::COPY_SRC);
                buffer
            },

            inputs: Vec::new(),
            query_slots: Vec::new(),
            input_positions: Vec::new(),
            query_publication: 0,
            assembly_inputs: EntityHashMap::default(),
            assembly_part_count: 0,
            assembly_pending: HashSet::default(),
            material_receipt: (0, 0),
            material_changes: Vec::new(),
            binding_receipt: HashMap::default(),
            slots: SceneSlots::default(),
            active_indices: {
                let mut b = StorageBuffer::default();
                b.set_label(Some("meshlet_active_indices"));
                #[cfg(feature = "graphics_debug")]
                b.add_usages(bevy_render::render_resource::BufferUsages::COPY_SRC);
                b
            },
            assembly_groups: StorageBuffer::default(),
            assembly_members: StorageBuffer::default(),
            group_dirty_indices: Vec::new(),
            member_dirty_indices: Vec::new(),
            dirty_indices: Vec::new(),
            active_dirty_indices: Vec::new(),
            binding_slots: Vec::new(),
            bvh_depths: Vec::new(),
            instance_material_assets: Vec::new(),
            scene_metadata: scene_metadata::SceneMetadata::default(),
            unresolved_material_instances: 0,
            instance_upload_dirty: true,
            rebuild_next_extract: false,
            material_mapping_dirty: true,
            next_material_id: 0,
            material_id_lookup: HashMap::default(),
            material_ids_present_in_scene: HashSet::default(),
        }
    }

    pub fn add_instance(
        &mut self,
        slot: SceneSlot,
        instance: MainEntity,
        root_bvh_node: u32,
        aabb: MeshletAabb,
        bvh_depth: u32,
        transform: &GlobalTransform,
        previous_transform: Option<&PreviousGlobalTransform>,
        render_layers: Option<&RenderLayers>,
        mesh_material_ids: &RenderMaterialInstances,
        render_material_bindings: &RenderMaterialBindings,
        not_shadow_receiver: bool,
        not_shadow_caster: bool,
        cutout: Vec4,
    ) {
        self.add_material_instance(
            slot,
            instance,
            root_bvh_node,
            aabb,
            bvh_depth,
            transform,
            previous_transform,
            render_layers,
            mesh_material_ids.mesh_material(instance),
            render_material_bindings,
            not_shadow_receiver,
            not_shadow_caster,
            cutout,
        );
    }

    fn add_material_instance(
        &mut self,
        slot: SceneSlot,
        instance: MainEntity,
        root_bvh_node: u32,
        aabb: MeshletAabb,
        bvh_depth: u32,
        transform: &GlobalTransform,
        previous_transform: Option<&PreviousGlobalTransform>,
        render_layers: Option<&RenderLayers>,
        mesh_material: UntypedAssetId,
        render_material_bindings: &RenderMaterialBindings,
        not_shadow_receiver: bool,
        not_shadow_caster: bool,
        cutout: Vec4,
    ) {
        // Build a MeshUniform for the instance
        let transform = transform.affine();
        let previous_transform = previous_transform.map(|t| t.0).unwrap_or(transform);
        let mut flags = if not_shadow_receiver {
            MeshFlags::empty()
        } else {
            MeshFlags::SHADOW_RECEIVER
        };
        if transform.matrix3.determinant().is_sign_positive() {
            flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }
        let transforms = MeshTransforms {
            world_from_local: transform.into(),
            previous_world_from_local: previous_transform.into(),
            flags: flags.bits(),
        };

        let mesh_material_binding_id = if mesh_material != DUMMY_MESH_MATERIAL.untyped() {
            render_material_bindings
                .get(&mesh_material)
                .cloned()
                .unwrap_or_default()
        } else {
            // Use a dummy binding ID if the mesh has no material
            MaterialBindingId::default()
        };

        let mesh_uniform = MeshUniform::new(
            &transforms,
            0,
            mesh_material_binding_id.slot,
            None,
            None,
            None,
            None,
        );

        let i = slot.index as usize;
        self.update_slot_metadata(slot, mesh_material, bvh_depth);
        extend_to(&mut self.instances, i, None);
        extend_to(&mut self.binding_slots, i, 0);
        extend_to(&mut self.bvh_depths, i, 0);
        extend_to(
            &mut self.instance_material_assets,
            i,
            DUMMY_MESH_MATERIAL.untyped(),
        );
        extend_to(self.instance_uniforms.get_mut(), i, mesh_uniform.clone());
        extend_to(self.instance_aabbs.get_mut(), i, MeshletAabb::default());
        extend_to(self.instance_material_ids.get_mut(), i, 0);
        extend_to(self.instance_bvh_root_nodes.get_mut(), i, 0);
        extend_to(self.instance_cutouts.get_mut(), i, Vec4::ZERO);
        self.instances[i] = Some((
            instance,
            render_layers.cloned().unwrap_or_default(),
            not_shadow_caster,
        ));
        self.instance_uniforms.get_mut()[i] = mesh_uniform;
        self.instance_aabbs.get_mut()[i] = aabb;
        self.instance_material_assets[i] = mesh_material;
        self.binding_slots[i] = mesh_material_binding_id.slot.0;
        self.bvh_depths[i] = bvh_depth;
        self.instance_bvh_root_nodes.get_mut()[i] = root_bvh_node;
        self.instance_cutouts.get_mut()[i] = cutout;
        self.slots.activate(slot);
        self.dirty_indices.push(slot.index);
        self.instance_upload_dirty = true;
        self.slots.changed();
    }

    /// Get the material ID for a [`crate::Material`].
    pub fn get_material_id(&mut self, material_asset_id: UntypedAssetId) -> u32 {
        *self
            .material_id_lookup
            .entry(material_asset_id)
            .or_insert_with(|| {
                self.material_mapping_dirty = true;
                self.next_material_id = self
                    .next_material_id
                    .checked_add(1)
                    .expect("meshlet material ID capacity");
                self.next_material_id
            })
    }

    pub fn material_present_in_scene(&self, material_id: &u32) -> bool {
        self.material_ids_present_in_scene.contains(material_id)
    }

    fn query_matches<'a>(&self, rows: impl Iterator<Item = InstanceRow<'a>>) -> bool {
        if self.slots.active_indices().len() != self.query_slots.len() + self.assembly_part_count {
            return false;
        }
        let unchanged_slots = self.query_publication == self.slots.publication;
        let mut inputs = self.inputs.iter().zip(&self.query_slots);
        rows.into_iter().all(|row| {
            inputs.next().is_some_and(|(input, slot)| {
                input.as_ref().is_some_and(|input| {
                    input.slot == *slot
                        && (unchanged_slots || self.slots.is_active(*slot))
                        && input.matches(row)
                })
            })
        }) && inputs.next().is_none()
    }

    fn scene_matches<'a>(
        &self,
        rows: impl Iterator<Item = InstanceRow<'a>>,
        materials: &RenderMaterialInstances,
        bindings: &RenderMaterialBindings,
        unchanged_inputs: bool,
    ) -> bool {
        self.material_receipt == materials.instances.receipt()
            && binding_slots_match(&self.binding_receipt, bindings)
            && (unchanged_inputs || self.query_matches(rows))
    }

    // Resolve only the bounded changed-owner journal to persistent addresses.
    // Retained instances then use a direct indexed flag instead of a hash lookup.
    fn changed_materials(&mut self, materials: &RenderMaterialInstances) -> bool {
        let Some(changes) = materials.instances.changes_since(self.material_receipt) else {
            return true;
        };
        self.material_changes.resize(self.slots.capacity(), false);
        self.material_changes.fill(false);
        for entity in changes {
            if let Some(slot) = self.slots.get(entity.id()) {
                self.material_changes[slot.index as usize] = true;
            }
        }
        false
    }

    fn observe_query(&mut self, position: usize, entity: Entity) -> SceneSlot {
        let retained = self.query_slots.get(position).copied();
        let slot = retained
            .filter(|slot| self.slots.touch_known(*slot, entity))
            .unwrap_or_else(|| self.slots.touch(entity));
        self.order_input(position, slot);
        slot
    }

    fn order_input(&mut self, position: usize, slot: SceneSlot) {
        extend_to(&mut self.input_positions, slot.index as usize, usize::MAX);
        let previous = self.input_positions[slot.index as usize];
        let previous = if previous == usize::MAX {
            self.query_slots.push(slot);
            self.inputs.push(None);
            self.query_slots.len() - 1
        } else {
            previous
        };
        self.query_slots.swap(position, previous);
        self.inputs.swap(position, previous);
        self.query_slots[position] = slot;
        self.input_positions[self.query_slots[previous].index as usize] = previous;
        self.input_positions[slot.index as usize] = position;
    }

    /// Material preparation must follow renderer parts as well as ECS mesh owners.
    pub(super) fn material_assets_for_pipeline(
        &self,
        ordinary: &RenderMaterialInstances,
    ) -> HashSet<UntypedAssetId> {
        ordinary
            .instances
            .values()
            .map(|instance| instance.asset_id)
            .chain(self.scene_metadata.materials.keys().copied())
            .collect()
    }

    fn update_slot_metadata(&mut self, slot: SceneSlot, material: UntypedAssetId, depth: u32) {
        let i = slot.index as usize;
        if self.slots.is_active(slot) {
            if self.instance_material_assets[i] == material && self.bvh_depths[i] == depth {
                return;
            }
            self.scene_metadata
                .remove(self.instance_material_assets[i], self.bvh_depths[i]);
        }
        self.scene_metadata.add(material, depth);
    }

    fn deactivate_slot(&mut self, slot: SceneSlot) {
        if self.slots.deactivate(slot) {
            let i = slot.index as usize;
            self.scene_metadata
                .remove(self.instance_material_assets[i], self.bvh_depths[i]);
        }
    }

    fn refresh_material_ids(&mut self) -> usize {
        self.material_ids_present_in_scene.clear();
        self.unresolved_material_instances = 0;
        for (asset, count) in &self.scene_metadata.materials {
            let id = self.material_id_lookup.get(asset).copied().unwrap_or(0);
            if id == 0 {
                self.unresolved_material_instances += count;
            } else {
                self.material_ids_present_in_scene.insert(id);
            }
        }
        // A new mapping can resolve any previously unprepared row. Otherwise,
        // every changed source row is already in the publication journal.
        let rows = if self.material_mapping_dirty {
            self.slots.active_indices().to_vec()
        } else {
            self.dirty_indices.clone()
        };
        for &slot in &rows {
            if self.slots.entity(slot).is_none() {
                continue;
            }
            let i = slot as usize;
            let material_id = self
                .material_id_lookup
                .get(&self.instance_material_assets[i])
                .copied()
                .unwrap_or(0);
            if self.instance_material_ids.get()[i] != material_id {
                self.instance_material_ids.get_mut()[i] = material_id;
                self.dirty_indices.push(slot);
                self.instance_upload_dirty = true;
            }
        }
        self.material_mapping_dirty = false;
        rows.len()
    }

    fn refresh_scene_metadata(&mut self) {
        self.scene_instance_count = self.slots.active_indices().len() as u32;
        self.max_bvh_depth = self.scene_metadata.max_depth();
    }

    fn finish_extract(
        &mut self,
        queue: &RenderQueue,
        mut retiring: Vec<(SceneInstance, SceneSlot)>,
    ) -> usize {
        retiring.extend(self.slots.unobserved_from(&self.query_slots));
        for &(owner, slot) in &retiring {
            if self.slots.get_part(owner) == Some(slot) {
                self.deactivate_slot(slot);
            }
        }
        let removed = self.slots.remove_parts(queue, retiring);
        for slot in &removed {
            if let Some(position) = self.input_positions.get_mut(slot.index as usize) {
                *position = usize::MAX;
            }
            if let Some(row) = self.instances.get_mut(slot.index as usize) {
                *row = None;
            }
        }
        let active = self.slots.active_indices();
        if self.active_indices.get().as_slice() != active {
            self.active_dirty_indices
                .extend(self.active_indices.set_indices(active));
            self.instance_upload_dirty = true;
        }
        self.refresh_scene_metadata();
        self.query_publication = self.slots.publication;
        removed.len()
    }
}

pub fn extract_meshlet_mesh_entities(
    mut meshlet_mesh_manager: ResMut<MeshletMeshManager>,
    mut instance_manager: ResMut<InstanceManager>,
    // TODO: Replace main_world and system_state when Extract<ResMut<Assets<MeshletMesh>>> is possible
    mut main_world: ResMut<MainWorld>,
    mesh_material_ids: Res<RenderMaterialInstances>,
    render_material_bindings: Res<RenderMaterialBindings>,
    mut system_state: Local<
        Option<
            SystemState<(
                Query<(
                    Entity,
                    &MeshletMesh3d,
                    &GlobalTransform,
                    Option<&PreviousGlobalTransform>,
                    Option<&RenderLayers>,
                    Has<NotShadowReceiver>,
                    Has<NotShadowCaster>,
                    Option<&MeshletVisibilityCutout>,
                    Has<MeshletDoubleSided>,
                )>,
                assembly_extract::AssemblyRows<'static, 'static>,
                assembly_changes::AssemblyChanges<'static, 'static>,
                Res<AssetServer>,
                ResMut<Assets<MeshletMesh>>,
                MessageReader<AssetEvent<MeshletMesh>>,
                Res<MeshletCutoutAtlas>,
                Res<Assets<Image>>,
                MessageReader<AssetEvent<Image>>,
                super::instance_changes::InstanceChanges<'static, 'static>,
            )>,
        >,
    >,
    render_entities: &Entities,
    render_queue: Res<RenderQueue>,
) {
    let _source = main_world
        .get_resource::<bevy_diagnostic::FrameCount>()
        .and_then(|f| bevy_render::diagnostic::profile_source_frame(f.0));
    let _cpu_profile = bevy_render::diagnostic::profile_scope("meshlet.extract");
    let readiness = main_world
        .get_resource::<bevy_render::scene_readiness::SceneGeometryReadiness>()
        .cloned();
    // Get instances query
    if system_state.is_none() {
        *system_state = Some(SystemState::new(&mut main_world));
    }
    let system_state = system_state.as_mut().unwrap();
    let (
        instances_query,
        assemblies_query,
        mut assembly_changes,
        asset_server,
        mut assets,
        mut asset_events,
        atlas,
        images,
        mut image_events,
        mut input_changes,
    ) = system_state.get_mut(&mut main_world).unwrap();

    // View layer/shadow bits are cheap and depend on this frame's views.
    // Instance transforms/geometry/material buffers persist across camera motion.
    instance_manager
        .view_instance_visibility
        .retain(|entity, _| render_entities.contains(*entity));
    for buffer in instance_manager.view_instance_visibility.values_mut() {
        buffer.get_mut().clear();
    }
    let mut force_rebuild = instance_manager.rebuild_next_extract || atlas.is_changed();
    for event in image_events.read() {
        let (AssetEvent::Added { id }
        | AssetEvent::Modified { id }
        | AssetEvent::Removed { id }
        | AssetEvent::Unused { id }
        | AssetEvent::LoadedWithDependencies { id }) = event;
        if *id == atlas.0.id() {
            force_rebuild = true;
        }
    }
    instance_manager.rebuild_next_extract = false;
    instance_manager.dirty_indices.clear();

    // Free GPU buffer space for any modified or dropped MeshletMesh assets
    for asset_event in asset_events.read() {
        if let AssetEvent::Unused { id }
        | AssetEvent::Modified { id }
        | AssetEvent::Removed { id } = asset_event
        {
            if let Some(readiness) = &readiness {
                readiness.revoke(id.untyped());
            }
            meshlet_mesh_manager.remove(id);
            force_rebuild = true;
        }
    }

    if let Some(readiness) = &readiness {
        for id in readiness.requested::<MeshletMesh>() {
            if assets.contains(id) {
                meshlet_mesh_manager.queue_upload_if_needed(id, &mut assets);
            }
        }
    }
    let material_changed =
        instance_manager.material_receipt != mesh_material_ids.instances.receipt();
    let bindings_changed =
        !binding_slots_match(&instance_manager.binding_receipt, &render_material_bindings);
    let assembly_roots = assembly_extract::targets(
        &instance_manager,
        &assemblies_query,
        assembly_changes.collect(),
        force_rebuild || bindings_changed,
    );
    let unchanged_inputs = input_changes.unchanged()
        && instances_query.iter().len() == instance_manager.query_slots.len()
        && instance_manager.query_publication == instance_manager.slots.publication
        && instance_manager.query_slots.len() + instance_manager.assembly_part_count
            == instance_manager.slots.active_indices().len();
    if !force_rebuild
        && assembly_roots.is_empty()
        && instance_manager.scene_matches(
            instances_query.iter(),
            &mesh_material_ids,
            &render_material_bindings,
            unchanged_inputs,
        )
    {
        instance_manager.query_publication = instance_manager.slots.publication;
        record_extraction(&instance_manager, 0, 0, 0, true);
        return;
    }
    let all_materials_changed =
        material_changed && instance_manager.changed_materials(&mesh_material_ids);
    instance_manager.material_receipt = mesh_material_ids.instances.receipt();
    if bindings_changed {
        instance_manager.binding_receipt.clear();
        instance_manager.binding_receipt.extend(
            render_material_bindings
                .iter()
                .map(|(id, binding)| (*id, binding.slot.0)),
        );
    }
    instance_manager.slots.begin();
    let previous_publication = instance_manager.slots.publication;
    let mut added = 0usize;
    let mut changed = 0usize;
    let mut mesh_metadata: HashMap<AssetId<MeshletMesh>, Option<(u32, MeshletAabb, u32)>> =
        HashMap::default();
    // Shared alpha metadata is validated once per exact source layer/cutoff/sidedness.
    let mut alpha_metadata: HashMap<(Option<(u32, u32)>, bool), Option<Vec4>> = HashMap::default();

    // Iterate over every instance
    let mut query_count = 0;
    for (
        query_position,
        (
            instance,
            meshlet_mesh,
            transform,
            previous_transform,
            render_layers,
            not_shadow_receiver,
            not_shadow_caster,
            cutout,
            double_sided,
        ),
    ) in instances_query.iter().enumerate()
    {
        let row = (
            instance,
            meshlet_mesh,
            transform,
            previous_transform,
            render_layers,
            not_shadow_receiver,
            not_shadow_caster,
            cutout,
            double_sided,
        );
        query_count = query_position + 1;
        let slot = instance_manager.observe_query(query_position, instance);
        let i = slot.index as usize;
        let known = instance_manager.slots.is_active(slot)
            && instance_manager
                .inputs
                .get(query_position)
                .and_then(Option::as_ref)
                .is_some_and(|input| input.entity == instance && input.slot == slot);
        let material_changed = material_changed
            && (all_materials_changed
                || instance_manager
                    .material_changes
                    .get(i)
                    .copied()
                    .unwrap_or(true));
        let material = if material_changed || !known {
            mesh_material_ids.mesh_material(instance.into())
        } else {
            instance_manager.instance_material_assets[i]
        };
        let binding = if bindings_changed
            || !known
            || instance_manager.instance_material_assets[i] != material
        {
            render_material_bindings
                .get(&material)
                .map_or(0, |b| b.slot.0)
        } else {
            instance_manager.binding_slots[i]
        };
        let reused = !force_rebuild
            && known
            && instance_manager
                .inputs
                .get(query_position)
                .and_then(Option::as_ref)
                .is_some_and(|old| old.matches(row))
            && instance_manager.instance_material_assets.get(i) == Some(&material)
            && instance_manager.binding_slots.get(i) == Some(&binding);
        if reused {
            continue;
        }
        let input = InstanceInput::from_row(row, slot);
        changed += 1;
        let alpha = *alpha_metadata
            .entry((input.cutout, double_sided))
            .or_insert_with(|| {
                metadata(cutout, &atlas, &images).map(|mut value| {
                    value.z = f32::from(double_sided);
                    value
                })
            });
        instance_manager.inputs[query_position] = Some(input);
        let Some(cutout) = alpha else {
            instance_manager.deactivate_slot(slot);
            continue;
        };
        // Resolve shared geometry and asset readiness once per prototype rather
        // than once for every leaf/part instance in the forest.
        let mesh_info = mesh_metadata.entry(meshlet_mesh.id()).or_insert_with(|| {
            if asset_server.is_managed(meshlet_mesh.id())
                && !asset_server.is_loaded_with_dependencies(meshlet_mesh.id())
            {
                return None;
            }
            Some(meshlet_mesh_manager.queue_upload_if_needed(meshlet_mesh.id(), &mut assets))
        });
        let Some((root_bvh_node, _, _)) = *mesh_info else {
            instance_manager.deactivate_slot(slot);
            continue;
        };
        let (_, aabb, bvh_depth) = mesh_info.unwrap();
        added += usize::from(!instance_manager.slots.is_active(slot));
        instance_manager.add_instance(
            slot,
            instance.into(),
            root_bvh_node,
            aabb,
            bvh_depth,
            transform,
            previous_transform,
            render_layers,
            &mesh_material_ids,
            &render_material_bindings,
            not_shadow_receiver,
            not_shadow_caster,
            cutout,
        );
    }
    let (assembly_added, assembly_changed, retiring) = assembly_extract::extract(
        &mut instance_manager,
        &assemblies_query,
        &assembly_roots,
        force_rebuild || bindings_changed,
        &asset_server,
        &mut assets,
        &mut meshlet_mesh_manager,
        &render_material_bindings,
        &atlas,
        &images,
    );
    added += assembly_added;
    changed += assembly_changed;
    let removed = instance_manager.finish_extract(&render_queue, retiring);
    instance_manager.query_slots.truncate(query_count);
    instance_manager.inputs.truncate(query_count);
    assembly_hierarchy::rebuild(&mut instance_manager);
    let reused =
        previous_publication == instance_manager.slots.publication && changed == 0 && removed == 0;
    record_extraction(&instance_manager, added, changed, removed, reused);
}

fn record_extraction(
    manager: &InstanceManager,
    added: usize,
    changed: usize,
    removed: usize,
    reused: bool,
) {
    for (name, value) in [
        ("meshlet.scene_reused", usize::from(reused)),
        ("meshlet.slots_added", added),
        ("meshlet.slots_removed", removed),
        ("meshlet.slots_changed", changed),
        ("meshlet.slot_capacity", manager.slots.capacity()),
        ("meshlet.slots_live", manager.slots.counts().0),
        ("meshlet.slots_free", manager.slots.counts().1),
        ("meshlet.slots_retired", manager.slots.counts().2),
    ] {
        bevy_render::diagnostic::profile_value(name, value as f64, "count");
    }
    let serialized = serialization_bytes(manager);
    bevy_render::diagnostic::profile_value(
        "meshlet.cpu_serialization_bytes",
        serialized as f64,
        "bytes",
    );
}

fn extend_to<T: Clone>(values: &mut Vec<T>, index: usize, empty: T) {
    if values.len() <= index {
        values.resize(index + 1, empty);
    }
}

fn serialization_bytes(manager: &InstanceManager) -> usize {
    manager.instance_uniforms.cpu_backing_bytes()
        + manager.instance_aabbs.cpu_backing_bytes()
        + manager.instance_material_ids.cpu_backing_bytes()
        + manager.instance_bvh_root_nodes.cpu_backing_bytes()
        + manager.instance_cutouts.cpu_backing_bytes()
        + manager.active_indices.cpu_backing_bytes()
        + manager.assembly_groups.cpu_backing_bytes()
        + manager.assembly_members.cpu_backing_bytes()
}

fn binding_slots_match(
    slots: &HashMap<UntypedAssetId, u32>,
    bindings: &RenderMaterialBindings,
) -> bool {
    slots.len() == bindings.len()
        && bindings
            .iter()
            .all(|(id, binding)| slots.get(id) == Some(&binding.slot.0))
}

#[cfg(test)]
#[path = "instance_manager_tests.rs"]
mod tests;

/// For each entity in the scene, record what material ID its material was assigned in the `prepare_material_meshlet_meshes` systems,
/// and note that the material is used by at least one entity in the scene.
pub fn queue_material_meshlet_meshes(
    mut instance_manager: ResMut<InstanceManager>,
    frame: Option<Res<bevy_diagnostic::FrameCount>>,
) {
    let _source = frame.and_then(|f| bevy_render::diagnostic::profile_source_frame(f.0));
    let _cpu_profile = bevy_render::diagnostic::profile_scope("meshlet.queue_material");
    let instance_manager = instance_manager.deref_mut();

    if !instance_manager.instance_upload_dirty && !instance_manager.material_mapping_dirty {
        bevy_render::diagnostic::profile_value(
            "meshlet.unresolved_material_instances",
            instance_manager.unresolved_material_instances as f64,
            "count",
        );
        return;
    }
    let visited = instance_manager.refresh_material_ids();
    bevy_render::diagnostic::profile_value(
        "meshlet.material_rows_visited",
        visited as f64,
        "count",
    );
    bevy_render::diagnostic::profile_value(
        "meshlet.unresolved_material_instances",
        instance_manager.unresolved_material_instances as f64,
        "count",
    );
}
