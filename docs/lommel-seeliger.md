# Dark particulate disk scattering

`LommelMaterial = ExtendedMaterial<StandardMaterial, LommelSeeliger>` selects
reciprocal Lommel–Seeliger scattering for an opaque particulate surface.
Register the matching standard-material proxy in `LommelRayMaterials` and attach
`RaytracingMaterial3d` to the mesh. The caller removes registry entries when
proxies are retired. `SurfaceDetail.lommel = Vec4::X` provides the same raster
selection for scanned materials. `ScenePlugin` installs the material plugin.
Use Solarik's deferred opaque render path. The Bevy forward fallback remains
standard PBR and does not implement this disk law; it is not an equivalent mode.

Base colour means normal full-phase radiance factor rho. It is clamped to
0..0.25 per channel, rather than interpreted as Lambert hemispherical albedo.
The BRDF is `2 rho / (pi (mu0 + mu))` for positive incoming/outgoing cosines.
The shader helper returns BRDF times incoming cosine, as Solarik estimators
expect. At parallel full phase the outgoing radiance is `rho E / pi`.
Directional-hemispherical reflectance is
`4 rho [1 - mu0 ln(1 + 1/mu0)] <= 4 rho <= 1`.
Sampling uses the existing diffuse direction proposal with its matching PDF;
there is no additional smooth-interface specular lobe in this mode.

This is a low-albedo single-scattering disk approximation, not a full Hapke
model. See [Hicks et al. 2011](https://doi.org/10.1029/2010JE003733).
It supplies neither measured wavelength/phase behaviour nor opposition surge.
Do not use it for transmissive particles, bright ice or a smooth dielectric.
The alpha-masked geometry path remains the caller's responsibility.

Raster deferred flag bit 4 (packed bit 28) selects the same law as ray-material
flag bit 5. Bevy 0.19.1 uses deferred bits 0..2; Solarik Gaussian mode uses bit 3.
Resolved material layout and fixture constructors carry the explicit selection.
Particulate secondary hits use directional scattering rather than a Lambert-only
world irradiance cache; other materials keep the existing cache path.

`tests/lommel.rs` integrates hemispherical energy and checks reciprocity and full
phase. `tests/lommel_gpu.rs` reads the production helper back over cosine grids.
Run GPU fixtures serially with `--include-ignored --test-threads=1`. The consuming
space-sim fixture additionally compares native DLSS5 captures at identical
camera, light, exposure, projection and tone mapping.
