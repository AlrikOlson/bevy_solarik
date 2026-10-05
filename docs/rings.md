# Finite particulate ring transport

`RingPlugin` is installed by the raytracing scene plugin. Add `RingView(RingData)`
to an HDR camera and supply `RingShadow(RingData)` for directional-light shadowing.
Both must refer to the same physical ring. View centre is camera-relative;
shadow centre is relative to the scene render origin. Subtract large coordinates
in f64 before packing. Axis and normal must be orthogonal unit vectors.

All geometry is metres. Each uniformly spaced profile bin contains normal
optical depth, particle single-scattering albedo in [0,1], Henyey–Greenstein g
in (-1,1), and unused padding. Profile spacing and first radius are explicit.
The host is responsible for valid measured or physically generated profiles.
An inactive ring has half thickness zero; the default storage array still
contains one zero element. A view supports one annulus and its central ellipsoid.

Transport solves the once-scattered radiative-transfer equation with exact
homogeneous-cell integration, Beer–Lambert transmission, a normalized HG phase
function and measured/host-provided radial extinction. Finite slab and cylinder
intersections clip radial edges, thickness and the central hole. Computing
thickness relative to the ring-plane crossing avoids subtracting two large
camera distances to recover a ten-metre path. Radial paths use at most64 cells;
ordinary inclined crossings normally need one cell.

Planet-on-ring shadows use the specified oblate ellipsoid and the analytic area
of a uniform small stellar disk above a locally straight limb. This approximation
requires star angular radius much smaller than the planet's angular extent.
Ring-on-surface shadows attenuate actual Solarik sampled directional-light rays.
The same profile and finite path integration serve both. Four fixed subpixel rays
integrate thin ringlets in HDR before exposure/tone mapping.

Tests compile the production compositor and run production WGSL readback against
independent f64 numerical source integration and geometric column limits,
including a10m layer viewed from100Mm, inclined rays, in-plane radial exit and
off-plane grazing misses. Run `cargo test --test rings -- --include-ignored
--test-threads=1` serially on Vulkan.

This is a homogeneous, once-scattered continuum model, not a simulation of
colliding resolved particles. It omits interparticle multiple scattering,
correlated self-gravity wakes, diffraction and polarization. Regolith scattering
within individual particles is represented by supplied albedo/phase parameters.
The physical interpretation and validity of those inputs belong to the caller.
Single-scattering ring photometry: Bradley et al., Icarus (2013),
https://www.sciencedirect.com/science/article/pii/S0019103513001681 .
