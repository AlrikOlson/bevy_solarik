# Clear-sky atmosphere

## Optional spherical-world mode (development)

Insert `PlanetaryAtmosphere` alongside `AtmosphereState` to supply a sphere's
radius, finite shell height, body-fixed observer and optional spherical source
occluder. Public distances are metres; source and camera directions must use
the same body-fixed axes. Validate settings before insertion. Removing the
resource restores the existing local-ground mode. Existing constructors and
their radiometry remain unchanged; the Bistro images still depict that mode.

Planetary mode clips rays to shell entry/exit and opaque scene depth, using
64 (`Balanced`) or 128 (`High`) integration segments per pixel. A foreground
object outside the atmosphere receives no aerial effect. Ground mode retains
its compact aerial volume. The same 256×64 transmittance and 32×32 multiple
scattering fields now accept the configured radius/height. Medium or geometry
changes rebuild them; observer/light changes reuse them. All atmospheric
textures, including the environment cube, occupy 9,191,424 bytes. Planet mode
does not automatically bind its observer-dependent cube as global lighting.

Optional deterministic cloud density uses body-fixed, warped 3D value noise,
a finite 2–8 km layer, and integration-footprint filtering. No cloud textures
or baked sky assets are loaded. Coverage, extinction and u16 seed are explicit.
Extinction is validated over 0–0.2 m⁻¹, which covers liquid water cloud
(roughly 0.01–0.1 m⁻¹).

The planetary view integrates the cloud shell separately from clear air.
A ray crosses the shell at most twice (before and after passing below the
cloud base); each crossing is marched on uniform midpoints with a 0.25 km
step at the `High` budget (0.5 km at `Balanced`), at least 8 and at most the
view sample count, and stops once transmittance falls below 1e-4. Clear-air
segments before, between and after the crossings receive samples in
proportion to their length. Segments are composited in ray order. Lookup
fields (sky view, lighting cube, aerial volume) are clear air only: their
sample counts cannot resolve the shell, so clouds do not appear in the
lighting environment.
Weather is fictional and static. Cloud scattering uses a normalized mixture
of 35% Cornette–Shanks (g=0.7) and 65% isotropic phase, with albedo 0.99 and a
bounded plane-column self-shadow approximation. This is not a converged
multiple-scattering cloud solution. The optional eclipse uses uniform angular
disk overlap and this atmosphere's nominal 0.004675-radian solar radius.

This is an opaque-depth presentation approximation: secondary rays do not
integrate atmospheric/cloud transport, cloud shadows are not injected into
surface BRDFs, and transparent layers retain the existing opaque-endpoint
limitation. A consumer must separately supply attenuated surface illumination
where required. One observer and one atmosphere are supported, with validated
observer/occluder lengths up to 1e9 m; ground interception uses a sphere.

The Vulkan orbital test in `tests/atmosphere_luts.rs` compares 32 azimuth/limb
samples against an independent f64 metre-based reference with 2,048 view and
512 solar segments. Maximum clear-air radiance error is 0.158% (limit 10%,
0.1 cd/m² denominator floor), and absolute transmittance error is 0.000756
(limit 0.01). It writes 8×4 reference/GPU PPMs under `target/planetary-reference`,
checks foreground depth, eclipse extinction, finite nonnegative cloud output
and exact repeated evaluation. With the dedicated shell march, the 128/512
view-budget comparison has maximum 0.51% radiance-relative or 0–1
transmittance-absolute difference (limit 5%); the shared quadrature it
replaced measured 11.2%. Segmenting a ray at the shell with a vanishing cloud
extinction reproduces the clear-air reference to 0.034% radiance and 0.00017
transmittance. Convergence of the march does not establish cloud physical
accuracy: the lighting model above is unchanged.

At 1920×1080 on the same hardware the shell march raised planetary composite
GPU time from 0.68 to 1.42 ms p50 for a full-disk orbital view and from 1.43
to 3.07 ms for a limb view; clear views are unchanged.

Native 800×600 orbital/limb/day/night/pole/cloud-free/eclipse/vacuum views on
RTX 4090/Vulkan driver 616.56 show a bounded limb and stable return view.
Initial refined orbital composition GPU p50/p95 was 0.283/0.825 ms; limb
0.705/1.036 ms. The medium rebuild measured 0.0123 ms transmittance and
1.787 ms multiple scattering. These are individual dispatch timings from
120 retained samples (87 on the first view), not total frame benchmarks.
The declared composition p95 budget is 5 ms at this resolution.

The [July 2026 Smolder talk](https://advances.realtimerendering.com/s2026/index.html)
describes integrated general volumetric effects in Glacier. Its integration
goals informed explicit layer/depth handling, but its engine implementation
is not a drop-in WGSL atmosphere. Hillaire/Bruneton remain the numerical
baseline here; no unmeasured performance advantage over Smolder is claimed.

Run GPU checks serially with `--ignored --nocapture --test-threads=1`, separate
from captures and builds. Existing local-ground numerical tests remain required.

Development builds provide an optional `AtmospherePlugin`. It generates a
clear terrestrial atmosphere on the GPU and presents the sky and aerial
perspective before temporal reconstruction and tonemapping. The playground
uses it for raster lighting, Solarik and the reference pathtracer.

The transport follows [Hillaire 2020](https://sebh.github.io/publications/egsr2020.pdf):
small transmittance and multiple-scattering LUTs feed a horizon-focused sky-view
LUT and a camera-frustum aerial-perspective volume. Earth density profiles,
ozone absorption and the transmittance coordinate mapping follow
[Bruneton's atmosphere model](https://ebruneton.github.io/precomputed_atmospheric_scattering/).
This is an original implementation of those equations. It does not copy a
third-party shader implementation.

## Units and light consumers

Atmosphere textures contain unexposed linear RGB radiance in cd/m². Source
illuminances are top-of-atmosphere lux. Shader distances are kilometres;
public observer height and aerial distance are metres. Rayleigh scale height
is 8 km, aerosol scale height is 1.2 km, and the ozone layer is triangular,
centred at 25 km with 15 km half-width. Mie scattering uses a 0.9 single
scattering albedo and the Cornette–Shanks phase function.

There is no target-illuminance normalization, zenith calibration, enlarged
celestial disk or radiance gain. Exposure is a separate authored camera curve.
The half-float view target alone clamps at 65,000 to stay representable; the
lighting cubemap remains RGBA32F.

The cubemap excludes sun and moon disks. It feeds `SolarikSkyLight` and Bevy's
`GeneratedEnvironmentMapLight` at the same scale. Solarik GI, its world cache,
glossy/glass rays and its pathtracer already consume that sky binding. The
importance distribution retains its independent 128×128×6 sampling grid and
its existing solid-angle Jacobian/PDF contract. The texture itself is 256² per
face, with Bevy's cube/world Z convention.

Sun and moon are Bevy directional lights carrying the atmosphere's attenuated
RGB illuminance. Encode the largest channel as scalar illuminance and divide
the linear color by that channel; multiplying by luminance again loses energy.
The playground uses `sun_illuminance_rgb()` and `moon_illuminance_rgb()` for
these two rays, with 128 CPU quadrature samples only when state changes.
Solarik's finite directional emitters provide disk energy to reflection and
reference paths. The visible background draws the corresponding angular
disks, with pixel-footprint antialiasing. Solar radius is 0.004675 radians,
lunar radius 0.00452. Visible disk radiance is lux divided by its projected
solid angle and attenuated along the view ray.

A stable procedural star catalogue uses photometric stellar flux and a
normalized Gaussian pixel footprint, attenuated through the same atmosphere.
It is present in both the visible background and the lighting cubemap.
Daylight hides stars through radiance and exposure, without a time-of-day fade.
The moon is an authored full moon opposite the solar azimuth; orbital motion,
lunar phases and an astronomical star catalogue are outside this model.

## Update and ordering contract

| Data | Dimensions / format | Rebuilt when |
|---|---|---|
| Transmittance | 256×64 RGBA32F | Medium changes |
| Multiple scattering | 32×32 RGBA32F | Medium changes |
| Sky view | 512×256 RGBA32F | Sun, moon, observer or atmosphere state changes |
| Lighting cube | 256×256×6 RGBA32F | Sky changes or output image is replaced |
| Aerial scattering and RGB transmittance | Two 32³ RGBA16F textures | Each rendered camera frame |

Medium integration uses 128 transmittance samples and 64 directions × 64
segments for multiple scattering. View quality selects 32 (`Balanced`) or 64
(`High`, default) sky segments. Aerial slices use eight segments, with
quadratic distance spacing and an exact zero-distance slice. The exponential
segment integral has a Taylor limit for vacuum and tiny optical depth.

Generation runs in `RenderSystems::PrepareBindGroups`, before Bevy's generated
environment filtering and Solarik's sky distribution. No image bytes are
regenerated or uploaded per moving frame. An unchanged state skips all four
generation passes. Each dependent GPU pass has its own compute-pass boundary.
Generation waits for all required pipelines and only publishes completed state.

`AtmosphereBackground` runs after opaque rasterization and before Solarik and
raster transparent surfaces. `AtmosphereComposite` runs after surface lighting
and the pathtracer, before early postprocessing. It applies aerial perspective
only to nonzero opaque depth, so glass against the sky retains its composite.
This is a depth-based aerial approximation: transparent layers use the opaque
endpoint distance, and secondary transport does not integrate a separate
participating-medium path along every ray.

The playground preserves realtime history under continuous solar motion.
Parameter edits and solar cuts over one degree request Solarik, RR, TAA and
Neural Rendering resets. The reference accumulator resets for any lighting
change. Existing camera-cut handling remains active. Cache estimates adapt
through their bounded update cadence; resetting is not an instantaneous global
retrace of every inactive cache cell.

## Application setup

Add `AtmospherePlugin` after Bevy's render plugins. Allocate one
`AtmosphereEnvironment::new(&mut images)`, insert it and an `AtmosphereState`,
and give its image handle to `SolarikSkyLight` and/or a camera's
`GeneratedEnvironmentMapLight`. Add `AtmosphereCamera` to the camera. It
requires HDR, depth prepass, MSAA off and storage usage on the main texture;
explicit existing camera components must satisfy those requirements too.
The device must support filtered RGBA32F textures (`FLOAT32_FILTERABLE`) and
RGBA16F read/write storage textures. Validation here uses Vulkan on RTX 4090.
Update the directional lights from the state's RGB illuminance methods and
attach Bevy `SunDisk` components with the matching angular diameters.

One state and observer serve the environment map. This slice supports local
ground-scale scenes, observer heights 1–20,000 m and an Earth radius of
6,360 km with a 100 km atmosphere. It is not a multi-observer planetary renderer.
The virtual planet ground receives direct Lambertian illumination; its feedback
into the multiple-scattering approximation is not a full ground/atmosphere
pathtracing solution. Clouds, weather, terrain occlusion of the sky and orbital
travel are separate work.

## Playground controls

Existing scene files remain valid. An optional section selects physical
medium controls and an authored solar setting:

```toml
[atmosphere]
preset = "timeline" # also day, golden-hour, twilight, night
rayleigh = 1.0      # 0..8
mie = 1.0          # 0..16
ozone = 1.0        # 0..8
mie_g = 0.8        # 0..0.95
aerial_distance = 2000.0 # metres, 100..20000
stars = 1.0        # 0..4
quality = "high"  # high or balanced
```

Preset solar elevations are 40°, 3°, −6° and −18°, respectively, at 90° azimuth.
`timeline` keeps the existing scene sun path. `SCENE_SKY_PRESET` selects the
same presets; an explicit `SCENE_SUN` path takes precedence. The scene's
`[sky] ground_albedo` and `ibl_scale` continue to apply. Controls reject NaN,
infinity and values outside their documented ranges. Scene edits remain live
in fly mode.

`SCENE_SUN_SCALE` and `SCENE_MOON_SCALE` now scale the entire source, including
scattering and its disk, as well as direct lighting. They no longer isolate
only the old direct-light term. Zero both and set IBL scale zero for lamp-only
photometry. Equal `SCENE_DAY_EV`/`SCENE_NIGHT_EV` values provide fixed exposure;
`SCENE_LINEAR=1` retains the rig's untonemapped diagnostic path.

The atmosphere supplies the clear-air medium. The old local fog is off by
default. `SCENE_LOCAL_FOG=1` together with `SCENE_FX=fog` explicitly adds the
scene's separate artistic fog volume; it is additional extinction/scattering.

## Measured cost at 4K

September 6, 2026, Windows/Vulkan/RTX 4090, release build, native 3840×2160
Bistro with Solarik and DLSS Ray Reconstruction. Three independent fixed-sun
runs and three moving-sun runs each render 180 measured frames after 256 fixed
warmup frames. The camera is (4, 1.6, 12), looking at (−3, 3, −3), FOV 50°;
fixed sun is 40° elevation / 30° azimuth, moving sun goes 40° to −14° and
30° to 90°. Extra grading and local fog are off.

Diagnostics begin before GPU preparation and report only fresh measurements.
This includes Bevy's environment filtering and Solarik's importance distribution;
old retained samples from a one-time LUT rebuild are excluded. There are
175–181 asynchronous diagnostic reports in each capture interval.

| Run | Atmosphere GPU median / p95, ms | All sky GPU median / p95, ms | Application frame median / p95, ms |
|---|---:|---:|---:|
| Fixed 1 | 0.236 / 0.254 | 1.050 / 1.075 | 77.53 / 83.93 |
| Fixed 2 | 0.236 / 0.252 | 1.052 / 1.368 | 88.77 / 91.19 |
| Fixed 3 | 0.238 / 0.257 | 1.053 / 1.300 | 79.74 / 82.71 |
| Moving 1 | 0.382 / 0.401 | 1.191 / 1.512 | 83.33 / 89.28 |
| Moving 2 | 0.381 / 0.397 | 1.187 / 1.493 | 87.16 / 90.49 |
| Moving 3 | 0.385 / 0.404 | 1.192 / 1.524 | 87.37 / 90.32 |

Atmosphere includes generation, background, aerial LUT and composition.
“All sky” sums those GPU spans plus environment filtering and the sky CDF;
it is not the GPU duration of the whole renderer. Environment filtering has
0.682–0.691 ms run medians and the CDF 0.124 ms. Moving-sky generation itself
has 0.121 ms sky-view and 0.025 ms cubemap medians. An unchanged state dispatches
neither generation pass, confirmed in all three fixed runs. The one-time
medium rebuild costs 0.011–0.012 ms transmittance plus 1.226–1.227 ms multiple
scattering in these runs.

CPU update batch medians are 0.0090–0.0093 ms in the moving runs, measured in
three batches of 60 updates per run. Fixed states skip quadrature and do not
produce update batches. Both atmosphere alone and all sky work fit the
preselected 3 ms median / 5 ms p95 GPU budget; moving CPU updates fit 0.5 ms.
Filtering and CDF still run for fixed skies; skipping them is follow-up work.

Application timings include the renderer, capture scheduling and three PNG
readbacks/writes per 180-frame interval. Capture-wall-time means were
78.85, 88.81 and 80.03 ms fixed; 83.75, 87.28 and 87.61 ms moving. The previous
CPU-sky build, under the same capture recipe, measured 138.88, 148.56 and
141.15 ms fixed; 129.63, 130.17 and 139.08 ms moving. These are end-to-end
capture comparisons across the integrated change, not an isolated GPU speedup
or a benchmark of interactive play.

The measured executable SHA-256 is
`bdc5c44d207b48fe5959b4d29a9391d84f5163988d800f1012678df503b4e05b`.
These timings use driver 591.86 and precede the HUD registration, normal
capture shutdown and NR resource-cleanup fixes. The final README images use
driver 616.56 and include those fixes; these numbers are not measurements of
the newer driver.
The local rig receipts are `sky-performance-build.json`, `sky-proof-*/timing.json` and
`sky-performance.json` under the organizer's `artifacts/bevy-sponza/`.

## Verification

`tests/atmosphere.rs` checks state validation, bounded and reddening direct
illumination, vacuum/opaque limits and production WGSL mathematics against
an independent double-precision reference. `tests/atmosphere_luts.rs` executes
the actual transmittance, multiple-scattering, sky and cube compute shaders.
It checks all generated texels for finite nonnegative radiance, transmittance
in [0,1], dense control limits, zero scattering in vacuum, Lambertian ground
units, source linearity and nonnegative multiple-scattering contribution.

Reference sky integration uses 2,048 view segments and 512 independently
integrated solar segments per sample, in metres and f64, without production
LUTs. The preselected error limits are 5% away from the horizon and 10% at
grazing angles, with a 0.1 cd/m² denominator floor, and 0.005 absolute
transmittance error. At 60 sampled sky directions across 40°, 3°, −6° and −18°
solar elevations, the measured maximum radiance errors are **0.30% away from
the horizon and 0.87% grazing**. These are single-scattering accuracy checks;
the isotropic multiple-scattering closure is an approximation, not a claim
of agreement with a converged volumetric pathtracer. Integrating the production
256² cubemap in vacuum gives 5.5774×10⁻⁵ lux of stellar flux versus the independent
catalogue sum of 5.6370×10⁻⁵ lux: 1.06% error, within the preselected 15% footprint
and quadrature budget.

Final-build fixed-exposure captures also compare the visible sky with the
reference pathtracer's environment consumer at 40° and 3° solar elevation.
At 1280×720, 256 fixed warmup frames and EV100 12.5, the comparison uses the
upper 450 rows, excluding clipped pixels and the virtual ground. Mean absolute
RGB differences in sRGB-decoded display-linear values are 0.00002263 (day)
and 0.00002407 (golden hour), or 0.00496% and 0.01296% of the reference mean,
well inside the preselected 5% limit. Reference logs confirm stable accumulation.
These quantized image comparisons check consumer consistency; both consumers
share the atmosphere model, so they are not an independent physics oracle.
All four runs exited normally with zero NVIDIA driver events on 616.56.

The 4K day/golden-hour/twilight/night sky and Bistro views, day/night zeniths
and a golden-hour cube-boundary view were inspected. A 180-frame sunset sweep
has a largest adjacent display-linear mean-luminance change of 0.003032, with
no flash in the inspected sequence. The camera cut at timeline frame 600 shows
fresh noise without retained geometry from the previous shot. These earlier
visual audits use the same atmosphere shaders as the final build; their
`sky-preflight-build.json` receipt predates the fixed-pose and shutdown fixes.
Final static images and the reference comparison were recaptured after those fixes.
Local evidence includes `sky-transition.json`, `sky-transition-sheet.png`,
`sky-view-reference.json` and the `sky-*/capture.log` files.

Run GPU tests serially on a Vulkan device, without a simultaneous capture:

```text
cargo test --test atmosphere -- --ignored --nocapture
cargo test --test atmosphere_luts -- --ignored --nocapture
```

The ordinary gate also tests live scene edits, direct RGB light updates,
stationary history preservation and history invalidation. The rig's fixed-path
regression checks 5,400 poses at three observer heights: equal camera endpoints
must produce identical transform bits throughout the timeline, avoiding false
sky invalidation and reference accumulation resets. Capture provenance,
fixed-view settling and refreshed README assets are recorded in
[readme-captures.md](readme-captures.md).
