# Solarik's Bevy renderer fork

Development Solarik includes `vendor/bevy_render` as part of its Cargo workspace.
It starts from the published Bevy 0.19.1 crate, under its original MIT OR Apache-2.0
licenses. The license files remain beside its manifest. Solarik owns subsequent
changes; synchronizing this fork with upstream is not a maintenance requirement.

Applications must apply the root Cargo patch shown in the README. A dependency's
`[patch]` table is not inherited by its consumers. Solarik directly depends on the
local renderer too, so the application and Bevy's transitive dependencies must
resolve to that same crate. Use `cargo tree -i bevy_render` to inspect resolution.

## GPU preparation diagnostics

Atmosphere and generated environment maps encode work during the outer
`Render` schedule. Starting the diagnostics frame in `RenderGraph::Begin`
erased those earlier spans and reused their timestamp indices. The recorder
now starts before `RenderSystems::ExtractCommands`; resolution and submission
remain after graph rendering. This retains preparation and environment-filter
GPU timings. The playground enables inside-pass timestamps for diagnostic runs,
which Bevy's pass spans also require. The atmosphere capture audit checks that
moving-generation and environment-filter spans are present before accepting
performance totals.

## Missing geometry during asynchronous loading

The CPU-visible entity list is sorted by **main-world entity**. Its old lookup
used a binary search over the complete `(render entity, main entity)` tuple, whose
primary ordering is the render-world entity. These IDs need not have the same
order. A visible entity waiting for a material could therefore be filtered out
of both pending specialization and queue retries. With a fixed camera it could
remain absent indefinitely; more lighting warmup could not recover it.

The lookup now searches by main entity and verifies the corresponding render
entity. GPU-only visibility still uses its existing exact pair lookup.

Two tests exercise the real visibility and pending-queue APIs with independently
allocated IDs in different orders. The original implementation fails both:
only one of four pending visible meshes survives. The corrected implementation
retains all four and rejects stale render IDs and removed objects.

```text
cargo test -p bevy_render --lib visibility_retry_tests
cargo check --workspace --all-targets
cargo nextest run --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Bistro evidence

At the cafe camera (timeline frame 750 of 1800), unpatched 4K captures varied
between launches: the Vespa body and several potted shrubs were missing, while
the separate headlamp remained. Raw HDR before reconstruction and the depth
buffer also lacked the geometry. Disabling lens or fog effects did not reliably
fix it. The two patched 4K launches restored the complete Vespa and shrubs.
Temporary instrumentation identified the Vespa body as main entity `5106v0`,
and recorded that the old lookup rejected its actually-visible pair while the
corrected lookup recovered it. The diagnostic instrumentation was then removed.
Two further production-build
4K captures at 128 and 512 warmup frames also retained the complete Vespa and
shrubs. At dusk (timeline frame 1520), the 128-frame result differed from the
512-frame result by -1.29% on pavement, +0.40% on the facade and -0.99% on the
cafe wall. These are mean display-linear luminance differences after tone
mapping, at fixed camera and exposure, not a radiometric accuracy claim.

Captures use the original Bistro exterior plus its matching original interior,
Vulkan, RTX 4090, 3840x2160, DLAA ray reconstruction, no DLSS neural rendering,
128 or 512 warmup frames, and the preset fog, motion blur, exposure and lens settings.
This fix changes visibility bookkeeping; lighting and post-processing settings
are not used to conceal missing meshes. Development changes are not in v0.1.0.

Original scene: Amazon Lumberyard Bistro, distributed through NVIDIA ORCA under
CC BY 4.0. The complete local diagnostic evidence is under
`~/Documents/Solarik-Short/moped-fix/`.
