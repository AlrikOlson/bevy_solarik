# Collimated emission

`scene::collimated` provides a circular directional radiance profile for an
equivalent optical exit aperture. It represents a virtual image at infinity,
without tracing internal lenses or a light source. It contains no weapon or
gameplay configuration.

Construct `CollimatedEmission::new(world_axis, angular_radius_radians)` and
register it in `CollimatedMaterials` under a `StandardMaterial` asset ID.
The axis must be finite and nonzero, and the radius must be finite and in
`(0, 0.1)` radians. Callers update world orientation and remove stale entries.
The material's emissive RGB is peak radiance; existing alpha coverage still
weights the emitted contribution. A missing profile preserves ordinary
emission.

The shared WGSL function `bevy_solarik::collimated::collimated_weight` returns
one inside 90% of the tangent radius and smoothly falls to zero at the rim.
It returns zero behind the aperture. It uses cross-product magnitude divided
by the forward dot product, avoiding small-angle cosine cancellation.
Only triangles facing the emission axis emit. Geometry clips the aperture.

Ray hits, the reference pathtracer, primary glass and next-event samples use
this law. Packed light samples preserve the direction and angular radius;
emitter-selection power estimates account for the restricted solid angle.
Material GPU storage grows from 64 to 80 bytes. Light-sample storage is
unchanged. Angular emission is evaluated at the receiver, after light samples
are reused, rather than being baked into a camera-dependent light value.

Custom raster materials must evaluate the same law in their own shader.
`CollimatedEmissionPlugin` registers the library even in raster-only apps.
`RaytracingMaterial3d` supplies a separate `StandardMaterial` for a mesh whose
raster material is custom. Attach `RaytracingMesh3d` first, then replace its
default required raster material. Flush deferred insertions before replacing
materials so a late required-component insertion cannot restore that default.
Custom raster deferred emission must already include angular weighting;
the gbuffer has no additional angular-profile channel.

The profile does not add an isotropic light. For a planar aperture normal to
the axis, `flux_upper_bound(luminance, area)` bounds output by
`luminance * area * pi * sin(radius)^2`. Narrow cones can have high apparent
brightness and very little total power. Uniform area sampling can miss their
footprint at finite sample counts; this implementation does not add a
specialized narrow-beam importance sampler.

Validation includes CPU direction/eye-translation/flux tests, the GPU angular
integral test, and production emitter-sampling quadrature at a small angular
radius with nonuniform emitter-selection probability. Existing primary glass,
glossy glass, foliage and shadow GPU fixtures remain ordinary emitters and
retain their prior expected transport.

The integrated playground provides the optical presentation and capture
matrix. Its static world optic validates ray transport; its animated view
skin remains outside Solarik's current ray-geometry contract.
