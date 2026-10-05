# Coloured stellar sources

`StellarDisksPlugin` adds up to two physical stellar disks independently of
the atmosphere. Attach `StellarDisks` to the HDR camera: unit world directions
and angular radii in radians in `direction`, RGB irradiance in lux in
`irradiance`. A zero irradiance disables a source. Inputs must be finite,
nonnegative and radii positive and at most0.1rad. Supply the same irradiance
to direct surface lights. With this pass active set
`AtmosphereState::render_disks=false` to avoid counting disk emission twice.

For radius r and pixel half-width w, angular coverage is
K(theta)=1-smoothstep(r-w,r+w,theta). Radiance per unit illuminance is
K/(2pi integral(theta K dtheta)). The code evaluates the radial integral
analytically, including the resolved constant interior. This is a pixel
reconstruction approximation for a uniform stellar disk, not limb darkening.
It conserves unresolved flux; the small-angle approximation differs from
the projected sin(theta)cos(theta) integral by less than0.5% for r<=0.1rad.
The GPU test integrates that exact projected measure across six disk sizes.
Geometry depth occludes the disk. Atmospheric attenuation is composed later.

The two atmosphere sources independently carry RGB colour, lux and angular
radius. Clear-air and cloud single/diffuse scattering sum each source's
contribution, with its own path transmittance, phase angle and cloud shadow.
Colours are linear RGB per photopic Y; no exposure belongs in input lux.
This is RGB radiative transport, not a full spectral solver. Cloud multiple
scattering retains the approximation and validity of docs/atmosphere.md.

`tests/atmosphere_luts.rs` executes the production shaders with mapped numeric
results. It checks two-source colour linearity and disk integrated flux.
Run ignored Vulkan tests serially with `--test-threads=1` and no other GPU jobs.
