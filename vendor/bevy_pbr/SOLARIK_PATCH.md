# Bevy PBR 0.19.1 patch

This directory vendors the published `bevy_pbr` 0.19.1 crate, retaining its
MIT and Apache-2.0 licenses. The source archive SHA-256 is
`244ae7d618b51a59c913c36b0564a295cd85ea91a5e777b542bb7bdb846a23d6`.
The normalized manifest is retained so registry dependencies resolve outside
Bevy's monorepo. Cargo cache metadata and the crate's nested lockfile are omitted.

Behavioral upstream source changes include the material bind-group index in
`meshlet_prepass` and `meshlet_deferred_gbuffer_prepass`, in
`src/meshlet/material_shade_nodes.rs`: use group 3 instead of group 2.
Their pipeline layouts reserve group 2 for meshlet visibility/instance data
and group 3 for material data, matching the opaque meshlet shading pass.
Binding material data at group 2 produces a WGPU validation failure before
the first rendered frame when meshlets participate in a prepass.

`src/meshlet/mod.rs` also orders meshlet depth rasterization before the ordinary
early prepass, and meshlet material prepasses after the ordinary late deferred
prepass. Without these edges, the ordinary passes can overwrite VG depth or
G-buffer data. The native no-plant-rays control exposed missing VG silhouettes;
the corrected ordering retains them and passes the settled rebase comparison.
This establishes the Solarik `SkipDeferredLighting` path. Ordinary Bevy deferred
lighting pass-ID/stencil ordering is not established by this probe.

Material shading also samples merged scene depth at group 2 binding 8 and
discards a visibility-buffer fragment whose reversed-Z depth is farther away.
This prevents VG material shading from painting over nearer ordinary geometry.
The binding is wired in `resource_manager.rs` and `meshlet_bindings.wgsl`; the
comparison is in `visibility_buffer_resolve.wgsl`. Both depths use the exact
packed float written by the VG depth resolve; no scene-dependent epsilon is used.
The native red-box control verifies the actual occluder region, since a whole
image average alone masked this defect. Acceptance requires at least 98% of
ordinary red-box pixels to remain red in VG, with a valid control area of at
least 1% of the image, in addition to the full-image and rebase comparisons.

The processor's `from_mesh.rs` rounds signed positions to the nearest integer
before fixed-point conversion. Upstream's addition of 0.5 followed by truncation
shifted negative grid-aligned vertices one cell toward zero, diverging from
shared ray geometry and exceeding the documented half-cell quantization error.
Two tests decode the actual packed vertex bitstream and verify signed-grid
identity and the half-cell bound. Processor-enabled owning gates cover this path.

`asset.rs` exposes read-only `meshlet_count` and `storage_bytes` accessors for
generated prototype measurements. They count all stored LODs and packed geometry
and culling payloads; they do not estimate GPU allocation capacity or residency.
These measurements distinguish payload growth from cluster slot pressure in
downstream native fixtures without exposing mutable private buffers.

`MeshletMesh::bvh_node_count()` also exposes the immutable number of stored
BVH nodes, including the root. Downstream bounded fixtures use it together
with all-LOD meshlet counts for conservative admission before publishing
instances. The accessor does not add GPU overflow handling or prove a budget.

The opt-in `MeshletCutoutAtlas` and `MeshletVisibilityCutout` path rejects
single-mip RGBA alpha before visibility-buffer atomic writes in both hardware
and software rasterizers, including shadow views. Hardware interpolates UVs
perspectively; software interpolates UV/w and 1/w. UV transforms must be baked.
The atlas is bounded to 64 layers, 4096 pixels per dimension and 256 MiB.
Invalid or unloaded CPU inputs suppress cutout instances; missing/invalid GPU
atlases suppress their pixels. Opaque instances retain their existing path.
Raster material shading remains opaque, so callers must supply corresponding
alpha-tested ray and ordinary control materials independently. This does not
enable arbitrary Bevy MASK materials, mip chains, transmission or production
streaming. Native hardware/software/shadow acceptance is still required.

Meshlet asset format v4 stores authored MikkTSpace tangents through all LODs
and GPU resolve. The loader also accepts v3 with the legacy zero-tangent
fallback; malformed v4 tangent counts, values and handedness are rejected.
2026-10-07: CPU/GPU residency shares bit-identical authored tangents across
meshlets and LODs using a Vec4 palette and u32 indices. Material group 2 binding
10 carries indices rebased to each asset's palette allocation. Both allocations
are released together. The v4 wire format is unchanged: save expands indices,
and validated v3/v4 loads intern exact float bits, including signed zero and
handedness. No tangent quantization or geometry/LOD change is introduced.
CPU checks, tests and lint pass. The original-alpha grass native probe runs,
but its reverse appearance agreement is 97.5023%, below the unchanged 98%
acceptance requirement. This opt-in capability remains experimental; neither
full foliage appearance nor production residency/wind is accepted here.

Hardware visibility rasterization now rejects back-facing fragments before
alpha testing and atomic depth writes, including its shadow variants. This
matches the existing software rasterizer's winding rule. Material cull flags
do not configure the shared visibility pipeline; without this rejection,
explicit opposed plant surfaces can compete at equal depth and select an
opposite material normal. Same-release native forest and grassland captures
pass their unchanged 0.01 restored-view RGB bound at 0.002703 and 0.002606;
standard rebase/camera-cut calibration also passes. These are bounded live
placement checks, not full foliage streaming, wind or residency acceptance.

The owning workspace's `cargo fmt --all` also formats the vendored Rust files
with its toolchain's default layout. Additional Rust source differences from
the archive are that formatter output; they are retained under the workspace's
format-forward policy. Manifests and both license files remain byte-identical.

Perspective meshlet LOD now converts simplification error into world units,
as the orthographic path already does. Without this conversion enlarged
instances lose detail too early. The space-sim `meshlet_lod_gpu` Vulkan
regression executes the production predicate for equal projected trees at
scales 0.1, 1 and 10; it fails before the repair and passes afterward.

The processor also includes UVs in its attribute error. It uses meshoptimizer's
documented reciprocal square root of mean absolute UV triangle area as the
automatic UV weight, without an extra tuning factor. A planar texture-kink
regression demonstrates the previous normal-only error accepting texture
damage at a 0.001 limit. Original positions, UVs, tangents and alpha images
remain unchanged; this is not an aggregate foliage coverage guarantee.

`from_mesh_preserve_area` is an explicit leaf-surface opt-in. Total-area
redistribution follows the meshoptimizer author's clusterlod approach, with
locked group seams and a moderate-shrinkage guard. Separate LOD vertices keep
source geometry immutable. Bounds include displaced vertices; the area path
uses max(inherited error, local simplification error + dilation displacement),
following the monotonic merge in pinned meshoptimizer clusterlod c313abae.
This is a perceptual LOD estimate, not a Hausdorff bound. Native near/far crown
coverage passes the unchanged 98% requirement (98.95% / 98.24%).
It solves triangle area along bounded boundary directions rather
than compounding first-order area overshoot. This is approximate foliage
coverage, not an exact alpha-visibility guarantee. Solid meshes retain the
ordinary conversion path. `finest_lod_only` is a diagnostic copy for isolating
LOD loss and deliberately forfeits LOD savings.

The conservation objective includes interior triangles: selecting only
boundary-adjacent triangle area made an unchanged retriangulation expand and
restored only 3/4 of a grid's surface. Both cases have RED/GREEN regressions.
Only unlocked open-boundary vertices move, consistent with Epic's documented
Nanite Preserve Area operation. Total area does not guarantee projected cover.

meshopt 0.6.2 is now vendored at `../meshopt`, with its upstream ErrorClamped
attribute-quadric correction exposed only to foliage. See its SOLARIK_PATCH.md
for pinned provenance and licenses. Opt-in debug counters report local,
inherited and dilation errors plus finite/terminal queue costs.

The opt-in foliage path partitions triangle centroids and meshlet centers by
balanced longest-axis spatial splits. Shared-vertex-only METIS grouping has
no locality constraint for disconnected leaves; redistribution must remain
local to avoid exchanging coverage between unrelated branches. Existing
126-triangle and eight-meshlet capacities remain. Solid meshes retain METIS.

`MeshletMesh::workset_profile` supplies a conservative distance-dependent
queue-write bound for perspective views. It includes every BVH child and
meshlet, with no assumed frustum or occlusion savings. Consumers must bound
all active view projections and scales, use all-LOD admission for unsupported
views, and account for positional uncertainty. This does not change GPU queue
capacity, preserve aggregate foliage coverage, or establish frame-time targets.

Regression evidence is the space-sim native vegetation virtual-geometry probe
with WGPU validation enabled, deferred shading, Solarik, and DLSS. Headless
geometry tests cannot exercise this GPU binding failure. Acceptance requires
rerunning that probe after this patch, plus the owning workspace's gates.
Downstream application roots must patch `bevy_pbr` as well as `bevy_render`;
Cargo does not inherit dependency-level `[patch.crates-io]` sections.
