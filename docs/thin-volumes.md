# Optically thin physical volumes

Install `ThinVolumePlugin` and attach `ThinVolumes` to an HDR perspective camera.
`GaussianKernel::axial` accepts a camera-relative centre in metres, a world-axis
direction, transverse and longitudinal standard deviations in metres, and the
volume integral of the linear RGB radiance density. Density j is cd/m³;
integrating j ds gives cd/m². Recompute camera-relative centres in double
precision when the camera moves. No astronomical rescaling is performed.

For covariance Sigma and integrated source J, density is
j(x)=J exp(-q²/2)/((2 pi)^1.5 sqrt(det Sigma)). Transform a ray into unit-Gaussian
coordinates q+t v. Complete the square and integrate the one-dimensional
Gaussian with erf between camera t=0 and the opaque depth distance. Six-sigma
perpendicular truncation has relative peak error below exp(-18). The erf
approximation has approximately 1.5e-7 absolute error. Radiance is added before
atmosphere and optics, after stellar and diffuse sky sources.

The caller supplies scattering phase, incident illumination or emissivity.
This renderer is a first-order optically thin approximation: no extinction,
self-shadowing, multiple scattering or illumination gradients inside a kernel.
The caller must ensure optical depth is small and select spatial quadrature
that resolves its physical density. A Gaussian is an integration kernel, not
an implied fluid simulation. Perspective cameras only.

A normalized half-pixel Gaussian reconstruction kernel broadens unresolved
sources in image sampling and adjusts density normalization to preserve total
emission. It is an isotropic spatial approximation to a screen filter near the
kernel centre; fully resolved kernels are unchanged to pixel-order accuracy.
It prevents missed subpixel cores and does not alter physical input dimensions.

`tests/thin_volume.rs` validates the complete production compositor with the GPU
compiler, compares 64 finite rays against independent 32768-step f64 midpoint
quadrature, and checks normalized source power, rotated axes and invalid inputs.
Run its ignored GPU case serially with `--include-ignored --test-threads=1`.
