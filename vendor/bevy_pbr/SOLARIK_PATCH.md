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

The owning workspace's `cargo fmt --all` also formats the vendored Rust files
with its toolchain's default layout. Additional Rust source differences from
the archive are that formatter output; they are retained under the workspace's
format-forward policy. Manifests and both license files remain byte-identical.

Regression evidence is the space-sim native vegetation virtual-geometry probe
with WGPU validation enabled, deferred shading, Solarik, and DLSS. Headless
geometry tests cannot exercise this GPU binding failure. Acceptance requires
rerunning that probe after this patch, plus the owning workspace's gates.
Downstream application roots must patch `bevy_pbr` as well as `bevy_render`;
Cargo does not inherit dependency-level `[patch.crates-io]` sections.
