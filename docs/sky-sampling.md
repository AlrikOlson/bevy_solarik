# Sky importance sampling

Measured 2026-09-06 on Windows 11, RTX 4090 (591.86), Vulkan,
Bevy 0.19.1, Bistro at 1920×1080. Baseline: `d6c22c7`.

The first realtime GI bounce now draws one direction from a 50/50 mixture
of the sky distribution and a cosine hemisphere. Each frame, two compute
dispatches rebuild a 6×128×128 cubemap CDF: parallel inclusive scans within
rows, then a marginal scan over 768 rows. The buffer uses 396,288 bytes per
view. Cells are weighted by luminance and their midpoint cubemap Jacobian.
Uniform jitter inside the selected cell uses the exact Jacobian at the
sampled direction when converting the PDF to solid angle.

Both hit and miss paths use the reciprocal mixture PDF (one-sample balance
MIS). Below-surface sky draws remain zero-valued samples with confidence
one; discarding their count would condition temporal reuse on a successful
draw. Black skies fall back to cosine sampling. No extra visibility ray is
introduced. The world cache, specular paths and pathtracer keep their
existing sampling.

This follows the solid-angle PDF and MIS treatment in
[PBRT's infinite area lights](https://pbr-book.org/4ed/Light_Sources/Infinite_Area_Lights)
and [importance sampling](https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling).

## Brightness

Frame 140, sun on at 32 degrees, default **60-frame warmup**, NR off.
Linear captures use EV100 **11.74**, the exposure logged at the captured
pose. The last exposure printed by `photometry.sh` is from later flushing
frames and must not be used. Reference: the existing power-weighted
8192-frame pathtracer capture documented in [light sampling](light-sampling.md).

| Patch | Rectangle | Reference nits | Baseline RR EV | Sky MIS RR EV | Baseline raw EV | Sky MIS raw EV |
|---|---|---:|---:|---:|---:|---:|
| Near cobbles | 780,975,180,80 | 125.6454 | -0.014 | +0.015 | -0.101 | -0.156 |
| Right cobbles | 1700,950,180,80 | 90.3219 | +0.024 | +0.044 | -0.284 | -0.325 |

Both paths pass the 0.5 EV criterion with no clipped pixels in these
patches. **The current baseline already passes.** These single captures
do not demonstrate faster convergence or a brightness improvement; raw
means are slightly farther from the reference. The older roadmap report
of a one-stop deficit is not reproduced by this baseline.

## Cost

Median of the last three diagnostic readings before capture starts:

| GPU pass | Baseline RR ms | Sky MIS RR ms | Baseline raw ms | Sky MIS raw ms |
|---|---:|---:|---:|---:|
| Sky CDF build | — | 0.122 | — | 0.118 |
| Diffuse indirect lighting | 14.102 | 11.947 | 2.506 | 7.348 |

The added CDF pass costs about **0.12 ms**. End-to-end GI timings vary
substantially between runs; these readings do not establish a speedup.
Sampling also adds two binary searches and PDF evaluation to the initial
bounce. A controlled multi-run convergence/cost study remains follow-up
work, including whether 50/50 is the best mixture for an occluded sky.

## Validation and reproduction

`tests/sky_sampling.rs` runs the production WGSL on a Vulkan GPU with
analytic scene inputs: black, uniform, a bright spherical cap, a cap over
a black background, and black again after the bright cases. It checks
finite positive weights, cube direction roundtrips and Monte Carlo
irradiance against independent projected-disk integrals (2% tolerance).
The GPU test is explicitly ignored by ordinary nextest runs so machines
without Vulkan can run the library suite. Run it separately:

```sh
cargo test --test sky_sampling -- --ignored
```

Workspace check, nextest, Python tests and clippy all pass. The integrated
Bistro logs contain no shader validation errors. Captures and the local
asserting verifier are retained in the development workspace; they are
not distributed with this crate.

From that workspace root, with builds and GPU captures strictly sequential:

```sh
# Before building the change:
PHOTO_MODES="solari solari_raw" PHOTO_TAG=sky_baseline_140 SCENE_DIAG=1 bevy-sponza/tools/photometry.sh bistro 140
cargo build --manifest-path bevy-sponza/Cargo.toml --release --locked
PHOTO_MODES="solari solari_raw" PHOTO_TAG=sky_final_140 SCENE_DIAG=1 bevy-sponza/tools/photometry.sh bistro 140
python .magistr/verify-sky-sampling.py
```

The reference is `bevy-sponza/preview/photo_bistro_pathtracer_power_8192_140.png`;
final images use `photo_bistro_{solari,solari_raw}_sky_final_140.png`.
