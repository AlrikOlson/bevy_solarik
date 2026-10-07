# Vendored meshopt 0.6.2

Base: crates.io meshopt 0.6.2, package checksum
`e01e77ead21976b3a9f01ec1724f766923da74f0726364e2b0f425658935d71a`,
meshopt-rs source commit `585fe7f5120df13f494bcde7134d8b115fc28213`.
Original Rust binding authors and MIT/Apache-2.0 licensing retained.
Embedded meshoptimizer retains Arseny Kapoulkine's MIT attribution.

2026-10-07: backport only `meshopt_SimplifyErrorClamped` from
https://github.com/zeux/meshoptimizer/blob/c313abae7de39cc928966e53b7cb405f04bde45e/src/simplifier.cpp
and expose `SimplifyOptions::ErrorClamped` (bit 8) in Rust.
Attribute quadric error is limited to cumulative area weight before adding
positional error, including seam and complex-wedge cases. Disabled behavior
is unchanged. bevy_pbr enables this only for opt-in foliage conversion.
Regression lives in bevy_pbr meshlet/lod_attributes.rs: a unit-area surface
with high UV variation formerly reported 141350.64 units of error.

The vendor path avoids changing the shared Cargo registry cache. This is a
narrow correction, not a wholesale meshoptimizer upgrade.

The build script explicitly tracks all compiled C++ sources and the public
header with Cargo rerun-if-changed so local vendor corrections invalidate
the native archive. The regression initially exposed stale native output
without these dependencies; the rebuilt correction passes.
