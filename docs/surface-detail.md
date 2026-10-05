# Scanned detail at shading time

`SurfaceDetailPlugin` registers `DetailedMaterial`, an extension of
`StandardMaterial`, and `DetailedRayMaterials`, its ray-proxy map. Solarik
installs this plugin. Attach `RaytracingMaterial3d` with the proxy handle and
register the same extension against that handle in the map. Remove the entry
on eviction. Custom raster entities must remove the default
`MeshMaterial3d<StandardMaterial>` that `RaytracingMesh3d` requires.

Two RGBA coverage textures represent eight physical surface-area fractions.
The application decides these from its surface model. The shared colour and
normal/roughness arrays supply measured or explicitly authored microstructure.
Colour uses sRGB storage around a **linear** mean of 0.5; normal xy and
roughness use linear storage, roughness also around mean 0.5. The base material
sets macroscopic reflectance and roughness. Complete mip chains and a repeating
trilinear sampler are required. Inputs are dimensionless except scan sizes and
local mesh positions, both in metres. Mesh transforms must be rigid.

`DetailCoordinates::new` splits each material-frame origin divided by scan
size into an integer cell and fractional phase in f64. Only the fractional
phase and local displacement become f32. Neighbouring origins therefore share
the same scan field without relying on camera-relative world coordinates.
The cell must fit i32; divide very large astronomical coordinates into a
body-local material frame first. Local-coordinate precision remains limited
by the float mesh vertices; this API cannot recover already rounded vertices.

## Sampling model

Three orthogonal projections use normalized weights proportional to
`abs(n_axis)^4`. Their scale is the scan's recorded physical width. Three
translated scan patches blend with squared barycentric weights inside each
triangular cell. This is a stationary, by-example microstructure
approximation: nearby unknown surfaces are assumed to share the scan's local
statistics. The lattice and translations are generated placement choices,
not a geological process. Linear patch blending preserves the mean but reduces
variance; it is not the full histogram-preserving transform described by
[Burley (2019)](https://jcgt.org/published/0008/04/02/).

Normal maps represent height derivatives. With tangent gradient `g` and base
normal `n`, the shading normal is
`normalize(n - (g - n*dot(n,g)))`. This surface-gradient formulation follows
[Mikkelsen (2020)](https://jcgt.org/published/0009/03/04/) and allows the scans
to follow steep surfaces without stretching radial UV coordinates. It changes
shading, not silhouette, occlusion or physical geometry. Very steep normal-map
slopes are bounded by a reconstructed z floor of 0.1.

The raster footprint is the largest local-position screen derivative, in
metres. Mip level is `log2(footprint * image_resolution / scan_size)`, bounded
below by zero. Detail fades to its mean between footprints of 0.1 and 0.5 scan
widths. These are filtering choices, not material properties. Coverage maps
are independently filtered. The shared shader modulates the base mean by the
coverage-weighted scan variation: this assumes the relative variation is
representative across the unresolved mixture, rather than tracing each grain.

Ray hits call the same sampler at the interpolated local mesh position.
They evaluate a point sample; stochastic ray/pixel sampling performs the
integration. Raster derivatives have no equivalent on the ray path yet.
This can increase ray variance and does not constitute a ray-cone texture
filter. Colour, roughness, scale, projection and normal equations are shared.

## Verification and native driver issue

`tests/surface_detail.rs` checks neighbouring f64 origins at Earth radius,
preserving millimetre changes to better than 0.01 mm, and executes the
production sampler on Vulkan. GPU normals are compared to the cross product
of analytically displaced surface tangents, not another copy of the shader's
projection equation. Constant scans must preserve their input mean.

On NVIDIA 616.56, passing a dynamically selected texture-array descriptor as
a function argument crashed inside `nvoglv64.dll`. A small GPU test reproduced
the access violation. The ray variant passes descriptor **indices** and
indexes the global arrays at the actual sample instead. The same test now
passes. Atlas descriptors have a separate capacity of 64, independent of the
5,000 ordinary material texture slots.

The consuming Space Sim fixture rendered all six 1920x1080 approach poses
with Solarik, DLSS RR DLAA and DLSS5 after this change. Final recorded gates
and renderer revision are retained by the consuming project's Magistr cycle.
