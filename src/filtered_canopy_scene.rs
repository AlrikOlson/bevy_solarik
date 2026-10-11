//! Extraction and GPU publication; cell contents stay immutable during camera motion.
use super::{
    Bounds, Canopy3d, CanopyRegion, CanopySources,
    arena::Arena,
    asset::{GpuRegion, Instance},
};
use alloc::sync::Arc;
use bevy_camera::visibility::InheritedVisibility;
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_math::{Affine3A, Mat3, Mat4, UVec4, Vec3};
use bevy_render::{
    Extract,
    render_asset::RenderAssets,
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages, StorageBuffer},
    renderer::{RenderDevice, RenderQueue},
    scene_readiness::SceneGeometryReadiness,
};
use bevy_transform::components::GlobalTransform;
#[cfg(feature = "graphics_debug")]
#[path = "filtered_canopy_scene_diagnostic.rs"]
pub(super) mod diagnostic;
#[derive(Resource, Default)]
pub(super) struct Extracted {
    values: Vec<Instance>,
}
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PrepareCanopy;
#[derive(Resource, Default)]
pub(crate) struct CanopyGpu {
    pub sources: Option<Buffer>,
    pub regions: Option<Buffer>,
    pub scene: StorageBuffer<Vec<UVec4>>,
    pub active: bool,
    pub bindings_changed: bool,
    pub history_global_changed: bool,
    pub changed_bounds: Vec<[Vec3; 2]>,
    published: std::collections::BTreeMap<u64, (Instance, u32, Bounds)>,
    source: Option<Arc<super::SourceProfile>>,
}
pub(super) fn extract(
    roots: Extract<
        Query<(
            Entity,
            &Canopy3d,
            &GlobalTransform,
            Option<&InheritedVisibility>,
            Option<&super::CanopyRayMask>,
        )>,
    >,
    mut output: ResMut<Extracted>,
) {
    let mut values = Vec::with_capacity(output.values.len());
    for (entity, region, transform, visibility, mask) in &roots {
        if visibility.is_some_and(|v| !v.get()) {
            continue;
        }
        let identity = entity.to_bits();
        let transform = transform.affine();
        let previous = output
            .values
            .binary_search_by_key(&identity, |v| v.identity)
            .map_or(transform, |i| output.values[i].transform);
        values.push(Instance {
            id: region.0.id(),
            transform,
            previous,
            identity,
            rays: mask.is_none_or(|m| m.0),
        });
    }
    values.sort_unstable_by_key(|v| v.identity);
    output.values = values;
}
pub(super) fn prepare(
    extracted: Res<Extracted>,
    sources: Res<CanopySources>,
    assets: Res<RenderAssets<GpuRegion>>,
    mut gpu: ResMut<CanopyGpu>,
    mut arena: ResMut<Arena>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    readiness: Res<SceneGeometryReadiness>,
    frame: Res<FrameCount>,
) {
    arena.ensure(&device);
    gpu.bindings_changed =
        gpu.regions.as_ref().map(|b| b.id()) != arena.buffer.as_ref().map(|b| b.id());
    gpu.regions = arena.buffer.clone();
    let identity = sources.0.as_ref().map(|s| s.identity);
    gpu.history_global_changed = identity != gpu.source.as_ref().map(|s| s.identity);
    if gpu.sources.is_none() || identity != gpu.source.as_ref().map(|s| s.identity) {
        gpu.bindings_changed = true;
        let sentinel = [[0u32; 4]];
        let rows = sources.0.as_ref().map_or(sentinel.as_slice(), |s| s.rows());
        gpu.sources = Some(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("finite canopy source responses"),
            contents: bytemuck::cast_slice(rows),
            usage: BufferUsages::STORAGE,
        }));
        gpu.source = sources.0.clone();
    }
    for id in readiness.requested::<CanopyRegion>() {
        readiness.publish(
            id.untyped(),
            assets
                .get(id)
                .is_some_and(|a| Some(a.source_identity) == identity),
        );
    }
    let mut items = Vec::new();
    for instance in &extracted.values {
        let Some(asset) = assets.get(instance.id) else {
            continue;
        };
        if Some(asset.source_identity) != identity {
            continue;
        }
        let Some(bounds) = transformed_bounds(asset.minimum, asset.maximum, instance.transform)
        else {
            continue;
        };
        items.push((*instance, asset.allocation.range.start, bounds));
    }
    // The app's finer residency budget normally binds first; this is a metadata bound.
    if items.len() > 4096 {
        items.clear();
        tracing::error!("finite canopy instance limit exceeded");
    }
    update_history(&mut gpu, &items);
    let mut rows = pack(&mut items, frame.0);
    rows[1][1] = background_guide_control();
    gpu.active = !items.is_empty();
    gpu.scene
        .set(rows.into_iter().map(UVec4::from_array).collect());
    gpu.bindings_changed |= gpu.scene.write_buffer_changed(&device, &queue).allocated;
}
// Diagnostic only: retain opaque guides while physical canopy rays still run.
fn background_guide_control() -> u32 {
    #[cfg(feature = "graphics_debug")]
    {
        static MODE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            if std::env::var("SOLARIK_CANOPY_CONDITIONED").is_ok_and(|value| value == "1") {
                tracing::warn!("Finite canopy diagnostic: conditioned radiance enabled");
                return 2;
            }
            let enabled =
                std::env::var("SOLARIK_CANOPY_BACKGROUND_GUIDES").is_ok_and(|value| value == "1");
            if enabled {
                tracing::warn!("Finite canopy diagnostic: opaque background guides enabled");
            }
            u32::from(enabled)
        })
    }
    #[cfg(not(feature = "graphics_debug"))]
    0
}

fn update_history(gpu: &mut CanopyGpu, items: &[Item]) {
    let current: std::collections::BTreeMap<_, _> =
        items.iter().map(|i| (i.0.identity, *i)).collect();
    let mut changed = Vec::new();
    for (id, old) in &gpu.published {
        if current.get(id).is_none_or(|new| {
            old.0.id != new.0.id
                || old.0.transform != new.0.transform
                || old.0.rays != new.0.rays
                || old.1 != new.1
        }) {
            changed.push([old.2.low, old.2.high]);
        }
    }
    for (id, new) in &current {
        if gpu.published.get(id).is_none_or(|old| {
            old.0.id != new.0.id
                || old.0.transform != new.0.transform
                || old.0.rays != new.0.rays
                || old.1 != new.1
        }) {
            changed.push([new.2.low, new.2.high]);
        }
    }
    gpu.changed_bounds = changed;
    gpu.published = current;
}
fn transformed_bounds(low: Vec3, high: Vec3, transform: Affine3A) -> Option<Bounds> {
    let axes = [Vec3::X, Vec3::Y, Vec3::Z].map(|v| transform.transform_vector3(v));
    let rotation = Mat3::from_cols(axes[0], axes[1], axes[2]);
    let error = rotation.transpose() * rotation - Mat3::IDENTITY;
    if !transform.is_finite()
        || error.to_cols_array().iter().any(|x| x.abs() > 1e-4)
        || (rotation.determinant() - 1.).abs() > 1e-4
    {
        return None;
    }
    let mut lower = Vec3::splat(f32::INFINITY);
    let mut upper = Vec3::splat(f32::NEG_INFINITY);
    for x in [low.x, high.x] {
        for y in [low.y, high.y] {
            for z in [low.z, high.z] {
                let p = transform.transform_point3(Vec3::new(x, y, z));
                lower = lower.min(p);
                upper = upper.max(p);
            }
        }
    }
    let pad = lower.abs().max(upper.abs()) * (8. * f32::EPSILON) + Vec3::splat(0.001);
    let bounds = Bounds {
        low: lower - pad,
        high: upper + pad,
    };
    (bounds.low.is_finite() && bounds.high.is_finite()).then_some(bounds)
}
type Item = (Instance, u32, Bounds);
fn pack(items: &mut [Item], frame: u32) -> Vec<[u32; 4]> {
    let mut nodes = Vec::new();
    let mut instances = Vec::new();
    if !items.is_empty() {
        node(items, &mut nodes, &mut instances);
    }
    let mut rows = vec![
        [
            nodes.len() as u32 / 2,
            2,
            nodes.len() as u32 + 2,
            instances.len() as u32,
        ],
        [frame, 0, 0, 0],
    ];
    rows.extend(nodes);
    for (instance, base, bounds) in instances {
        for transform in [
            instance.transform,
            instance.transform.inverse(),
            instance.previous,
        ] {
            rows.extend(
                Mat4::from(transform)
                    .to_cols_array_2d()
                    .map(|r| r.map(f32::to_bits)),
            );
        }
        rows.push([
            bounds.low.x.to_bits(),
            bounds.low.y.to_bits(),
            bounds.low.z.to_bits(),
            base,
        ]);
        rows.push([
            bounds.high.x.to_bits(),
            bounds.high.y.to_bits(),
            bounds.high.z.to_bits(),
            u32::from(instance.rays),
        ]);
    }
    rows
}
fn node(items: &mut [Item], nodes: &mut Vec<[u32; 4]>, instances: &mut Vec<Item>) {
    let bounds = items
        .iter()
        .map(|i| i.2)
        .reduce(Bounds::union)
        .expect("nonempty canopy instance node");
    let start = nodes.len();
    nodes.extend([[0; 4]; 2]);
    let mut leaf = u32::MAX;
    if items.len() == 1 {
        leaf = instances.len() as u32;
        instances.push(items[0]);
    } else {
        let extent = bounds.high - bounds.low;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        items.sort_unstable_by(|a, b| {
            (a.2.low[axis] + a.2.high[axis])
                .total_cmp(&(b.2.low[axis] + b.2.high[axis]))
                .then_with(|| a.0.identity.cmp(&b.0.identity))
        });
        let middle = items.len() / 2;
        let (a, b) = items.split_at_mut(middle);
        node(a, nodes, instances);
        node(b, nodes, instances);
    }
    nodes[start] = [
        bounds.low.x.to_bits(),
        bounds.low.y.to_bits(),
        bounds.low.z.to_bits(),
        nodes.len() as u32 / 2,
    ];
    nodes[start + 1] = [
        bounds.high.x.to_bits(),
        bounds.high.y.to_bits(),
        bounds.high.z.to_bits(),
        leaf,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Quat;
    #[test]
    fn rigid_bounds_and_scene_addresses_preserve_rebased_pose() {
        let transform = Affine3A::from_rotation_translation(
            Quat::from_rotation_z(0.8),
            Vec3::new(1e5, -2e5, 3e5),
        );
        let bounds = transformed_bounds(-Vec3::ONE, Vec3::ONE, transform).unwrap();
        for x in [-1., 1.] {
            for y in [-1., 1.] {
                for z in [-1., 1.] {
                    let p = transform.transform_point3(Vec3::new(x, y, z));
                    assert!(p.cmpge(bounds.low).all() && p.cmple(bounds.high).all());
                }
            }
        }
        assert!(
            transformed_bounds(-Vec3::ONE, Vec3::ONE, Affine3A::from_scale(Vec3::splat(2.)))
                .is_none()
        );
        let instance = Instance {
            id: Default::default(),
            transform,
            previous: Affine3A::IDENTITY,
            identity: 77,
            rays: true,
        };
        let rows = pack(&mut [(instance, 123, bounds)], 42);
        assert_eq!(rows[0], [1, 2, 4, 1]);
        assert_eq!(rows[1][0], 42);
        assert_eq!(rows[16][3], 123);
        let actual =
            Mat4::from_cols_array_2d(&std::array::from_fn(|i| rows[4 + i].map(f32::from_bits)));
        assert_eq!(actual, Mat4::from(transform));
    }
    #[test]
    fn history_ignores_previous_pose_but_tracks_support_and_retirement() {
        let mut gpu = CanopyGpu::default();
        let mut instance = Instance {
            id: Default::default(),
            transform: Affine3A::IDENTITY,
            previous: Affine3A::IDENTITY,
            identity: 7,
            rays: true,
        };
        let bounds = Bounds {
            low: -Vec3::ONE,
            high: Vec3::ONE,
        };
        update_history(&mut gpu, &[(instance, 3, bounds)]);
        assert_eq!(gpu.changed_bounds.len(), 1);
        instance.previous = Affine3A::from_translation(Vec3::X);
        update_history(&mut gpu, &[(instance, 3, bounds)]);
        assert!(gpu.changed_bounds.is_empty());
        instance.transform = Affine3A::from_translation(Vec3::X);
        update_history(&mut gpu, &[(instance, 3, bounds)]);
        assert_eq!(gpu.changed_bounds.len(), 2);
        update_history(&mut gpu, &[]);
        assert_eq!(gpu.changed_bounds.len(), 1);
    }
}
