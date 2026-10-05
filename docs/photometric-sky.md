# Photometric sky composition

`RadianceSkyPlugin` and `PointSkyPlugin` provide generic camera-background
composition. They own no astronomical catalogue or galaxy model. Both run after
opaque geometry, before atmospheric background transport, and require HDR,
depth prepass, MSAA off and storage-capable camera output. Nonzero reverse-Z
depth occludes their contributions. They do not illuminate secondary rays.

`RadianceSky` takes an equirectangular RGBA32Float image whose RGB values are
linear radiance in cd/m². Longitude starts at +X and increases toward -Z;
latitude runs from +Y at the top to -Y at the bottom. Manual bilinear sampling
wraps longitude and clamps the poles. The shader reconstructs the world ray
from the actual view matrices and multiplies by Bevy's view exposure once.

`PointSky` takes an immutable shared array of `PhotometricPoint` and an observer
position. All positions use the caller's same physical length unit. Position.w
is illuminance in lux at one such unit. Colour.xyz is linear RGB normalized to
photopic Y. Colour.w=0 means a positioned source; colour.w=1 means a directional
measurement with unknown distance, so observer translation is ignored.
Callers must supply finite positive flux and colour and nonzero directions.

The shader computes F=I/r², projects the source through the view matrix, and
converts flux to radiance with the perspective pixel solid angle
Ω=4 cos³(theta)/(width height projection_x projection_y). A discrete normalized
9×9 Gaussian with sigma 0.65 pixels reconstructs an unresolved point. This is
a reconstruction footprint, not an enlarged physical source. Point contributions
use atomic compare/exchange floating addition, then compose into HDR output.
Half-float headroom is limited to 65000; flux conservation applies below clipping.
Sources outside the viewport and partially occluded footprints lose the
corresponding flux. Large resolved sources require geometry or another model.

Each view retains one source buffer plus 12 bytes per output pixel of atomic
accumulation. Immutable Arc identity changes rebuild the source buffer; camera
resolution changes reallocate accumulation. Removed views release allocations.

Ignored GPU tests execute the production shader bodies: `radiance_sky` checks
sampling against a CPU reference including the seam and poles; `point_sky`
checks overlapping sources, inverse-square distance and depth occlusion by
reading numeric GPU output. Run them serially with `--test-threads=1` and an
isolated fixture target directory. The point test uses RGBA32Float read/write
storage to isolate float arithmetic from half-float quantization and requests
the adapter-specific texture format feature explicitly.
