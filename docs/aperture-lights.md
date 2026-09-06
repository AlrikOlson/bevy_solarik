# Aperture-light audit

September 6, 2026, RTX 4090 / Vulkan. Renderer `bb4104d`; local rig
`6eb7fcf` plus a new generated lantern fixture. This rechecks the historical
0.6–1.2 EV Bistro pavement deficit after the glass-transport corrections.
Light power and renderer code were not adjusted to fit these measurements.

## Isolated lantern

The local rig's `python tools/make_lightbox.py --lantern` generates a grey
floor beneath an 800 lm point light at y=2 m. Five black, opaque, double-sided
panels enclose a 0.4 m wide box from y=1.75 to 2.25 m, leaving its bottom
open. Geometry tests check the five transformed boundaries and the fixed
camera/preset. Sky, sun and moon illumination are disabled.

At 640×360 and EV100 2, the reference uses 4,096 warm-up frames; realtime
uses 2,048. Linear capture disables presentation effects. Pixel rectangles
are `x,y,width,height` from the upper left:

| Floor patch | Rectangle | Reference nits | RR difference | Raw difference |
|---|---|---:|---:|---:|
| Centre | 280,160,80,40 | 2.2916 | −0.02 EV | −0.00 EV |
| Left | 170,210,60,40 | 1.2968 | −0.01 EV | −0.02 EV |
| Right | 410,210,60,40 | 1.2970 | −0.03 EV | −0.01 EV |

All patch means meet the 0.25 EV target. This fixture does not reproduce a
general aperture-light energy loss. It does not isolate every possible
reuse/cache configuration or certify arbitrary aperture widths.

## Bistro

The frame-400 comparison uses 960×540, EV100 8 and identical camera and
lighting. Raising EV100 from 5 prevents the measured reference patches from
clipping; the whole image still contains bright emitters. Only the named
patches are photometric evidence. Realtime warm-up is 2,048 frames.

The final reference uses 16,384 warm-up frames (16,385 including the saved
frame under the audited capture contract).

| Pavement patch | Rectangle | Reference nits | RR difference | Raw difference |
|---|---|---:|---:|---:|
| Left of post | 258,443,20,30 | 4.6022 | +0.16 EV | −0.35 EV |
| Railing shadows | 340,502,25,20 | 1.1240 | +0.06 EV | −0.28 EV |
| Post contact shadow | 282,491,16,20 | 0.3645 | −0.19 EV | +0.33 EV |

All these final patches are unclipped and within half a stop. The older
deficit does not reproduce on current code in these explicitly located
patches. The historical record did not retain exact rectangles, so these
are a new reproducible measurement of the named regions, not a byte-exact
rerun of the old comparison.

At 8,192 warm-up frames, reference patch means differ from the longer run
by +0.04, +0.20 and +0.08 EV respectively. The 4,096-frame differences are
+0.03, +0.48 and +0.29 EV; its railing patch has one clipped sample. This
remaining reference noise warrants caution, but the current RR comparison
meets the half-stop target against both longer references. It does not
support tuning light intensity or blaming a specific reuse/cache/RR pass.

Presentation screenshots retain the known night blotching/foliage limits;
passing patch energy tests does not establish image quality or temporal
stability in motion.

## Reproduction and interpretation

Use the separate local rig after its release build. Run captures and builds
sequentially. Set `SCENE_LINEAR=1`, `NR_CAPTURE=1`, `NR_START=400`,
`NR_RES=960x540`, `SCENE_NIGHT_EV=8`. Select `SCENE_PATHTRACE=1` for the
reference, normal Solarik for RR, and `SCENE_RR=0` for raw. Keep a separate
output directory and capture log for every run. Use 16,384 warm-up frames
for the reference and 2,048 for realtime. Measure with
`tools/photometry.py --ev100 8 --patch name=x,y,w,h reference=... rr=... raw=...`.

ReSTIR [visibility reuse can introduce bias](https://cwyman.org/papers/hpg21_rearchitectingReSTIR.pdf),
but that possibility alone is not a diagnosis. Clipped display values can
[hide differences in radiance](https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Path_Tracing).
The current measurements are patch averages, not an independent-seed
variance estimate. A single raw frame remains noisy.
