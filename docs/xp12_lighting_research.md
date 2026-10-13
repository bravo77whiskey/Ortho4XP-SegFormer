# X-Plane 12 lighting: what the engine actually does, and what we can change

2026-10-12. Research into XP12's sim-wide lighting complaints: flat lighting
(light looks like a brightness boost), the whole frame washing out toward the
dominant content, a milky atmosphere, cockpit vs exterior exposure, and harsh
or slow lighting transitions. Everything below was read from the installed
build (X-Plane.exe md5 d456a701…, shaders dated 2026-07-04).

## Outcome (2026-10-13, tested in-sim on X-Plane 12.4.2)

**Kept** (`scripts/xp12_lighting/profile.json`, applied with
`python scripts/xp12_lighting/shader_patch.py profile`):

| Patch | Shader archive | Setting | Effect |
|---|---|---|---|
| Sky | `dome.xsa` (3 variants) | radiance ×1.22, saturation ×1.16 | Deeper, more vivid sky. Fitted from screenshots so the stock tone curve reproduces the sky of the contrast-1.3 curve. Glossy pavement reflects it; matte ground and ambient light are unchanged. |
| Haze | `scatter_render_atmosphere.xsa` (9 variants) | ×0.7 | Less milky distant terrain. This pass discards sky pixels, so the sky is untouched. |
| Eye adaptation | `lumi_histo.xsa` variant 2 | ×2 faster both ways | Shorter blown-out/dark phase after the view changes. Needs confirming in a real flight. |
| Local tone mapping | `exposure_fusion.xsa` variants 1, 5 | sigma² ×1.5 | Slightly better cockpit vs exterior balance. |

**Rejected**:
- Every global tone-curve variant (contrast 1.2–1.4, exposure +0.4–0.75, saturation 0.85–1.2). The user preferred stock ground in every case. Per-channel contrast also raises apparent saturation.
- Centre-weighted metering (forced mode 1, radius 0.3). Stock looked better in the cockpit.
- 4 km shadow distance, even with fade ratio 0.9. The edge where shadows switch on stays visible on trees while moving.

**What doesn't work**: `settings.txt` art-control overrides for values the engine recomputes every frame, whether added as new lines or appended to Laminar's. Values read once at startup, like shadow distance, do work.

**After an X-Plane update or Steam file verification**, Laminar's shaders come back. Re-run the `profile` command. It recognises the new stock files, archives the old pristine copies and re-applies. If a patch no longer finds its code pattern, it stops and leaves that file untouched.

Tooling:
- `scripts/xp12_lighting/shader_patch.py`: SPIR-V patches, profile and restore. Backups go to `E:\Ortho4XP_Review\xp12_lighting\backups`.
- `scripts/xp12_lighting/art_controls.py`: decodes art-control defaults and does settings.txt backup/restore.

Sections 1–5 below are the original research notes. The presets in section 5 turned out not to take effect (see above).

## 1. The levers that exist

- **Art controls.** X-Plane.exe registers 8,285 named tuning values. 1,270 are
  floats whose defaults can be decoded from the registration call
  (`lea rdx,[name]` with the default in `xmm3`). `Resources/settings.txt` maps UI
  settings to art-control values through `SETTING_1` lines, so any of them can be
  overridden there. Of the lighting ones, Laminar's file only touches SSAO
  and MSAA, so the defaults below are the live values.
- **Shaders.** `Resources/shaders/bin/spv/*.xsa` are zip archives of SPIR-V with
  debug names intact. They are readable (a minimal disassembler is in the
  session scratchpad) and could in principle be replaced. Whether XP verifies
  them has not been tested.
- **Not reachable:** the CPU-side logic, such as how auto-exposure turns the
  luminance histogram into an EV, how weather profiles are blended, and when
  the ambient probe re-renders. Only its parameters can be tuned.

## 2. XP12's display pipeline, as decoded from the shaders

Every `hdr_*`/`resolve_*` variant applies the same chain (the `u_tonemap_mode`
switch has been specialised away, so it has only one case):

```
x  = scene_rgb * exposure_rat                         # camera exposure
k  = pow(slope * mid, 1/power) / mid                  # ASC-CDL pivoted on mid-grey
y  = pow(k * x + offset, power)                       #   (slope/offset/power per channel)
y' = WCS * y                                          # "AgX" inset matrix
t  = y' / (y' + D * (1 - mid))                        # rational (Reinhard-type) curve, per channel
z  = WCS_inv * t                                      # outset matrix
z  = mix(luma(z), z, saturation)
out= clamp(pow(z * output_gain, output_gamma), 0, 1) * D
```

The exposure-fusion pass builds 4 synthetic exposures of luma with the same
curve, spaced by the factor `D`, and writes them sRGB-encoded for the
multi-scale blend.

What this means:

- **The tone curve has no toe and a long, soft shoulder.** `t = y/(y+c)` is
  nearly linear in the shadows and compresses everything above mid-grey hard.
  With mid-grey preserved, a white surface in full sun ends up well below
  display white. This is the classic flat Reinhard look: highlights don't pop,
  so a lit surface reads as a brighter version of the same colour, not as
  light. ACES/filmic and "punchy" AgX looks add an S-curve (toe plus
  midtone contrast) for exactly this reason.
- **The "AgX" in 12.2 is the inset/outset matrix pair**, which desaturates
  highlights by design. The practitioner consensus on AgX is that it is flat
  and grey out of the box without a look applied.
- **ASC-CDL `power` is a free midtone-contrast knob.** The pivot is `mid`, so
  power > 1 steepens the curve around mid-grey without changing exposure. The
  defaults are all 1.0, so no contrast look is applied.

## 3. Physical references vs XP's defaults

| Quantity | Real world | XP12 default | Note |
|---|---|---|---|
| Exposure for bright sun | EV100 ≈ 15 (sunny 16) | — | |
| Exposure for sand/snow/beach | EV100 ≈ 16 | `photometric/ev100_max` = **14.8** | The camera cannot stop down enough for very bright scenes, so they stay overexposed by about 1 stop. A strong candidate for "bright ground washes out everything". |
| Direct sun vs clear-sky diffuse | ~120 klx vs ~20 klx (≈6:1, ~2.6 stops) | `skyc/*_ratio_*` (not decodable) | Too much ambient relative to direct light makes a flat, shadowless look. |
| Multiple scattering | ×1.0 (physical) | `scattering/multi_rat` = **1.25** | An artistic boost that lifts sky light and haze: milky atmosphere, lifted ambient. |
| Ground albedo for sky | ~0.1–0.3 (desert 0.3–0.4) | `scattering/earth_albedo` = 0.15 | Fixed, not per-location. |
| Shadow distance | unlimited | `shadow/csm/far_limit_exterior` = **1000 m** (Laminar's settings.txt at max quality) | Beyond 1 km no building, tree or terrain cast shadows: the landscape is lit by N·L only. This is a major part of the sim-wide flatness from the air. |
| Exposure-fusion shadow lift | — | `exposure_fusion/shadows` = 0, `sigma` 5.5, `max_mip` 6 | In Wronski's formulation "shadows" brightens the brightest synthetic exposure; 0 means local tone mapping only compresses and never lifts. |
| Auto-exposure | — | `autoexposure/trim_lo/hi` = 0, `gain_lo/hi` = 0.5/0.4 | No histogram percentile trimming, so the dominant content sets the exposure (the "averaged with the most dominant thing" effect). |
| Adaptation | — | `exposure/speed_up` 3, `speed_down` 5, `photometric/speed` 100 | Units unknown. |
| Ambient probe | — | `cube/recalc_degs_cos` 0.99985 (=1°), `cube/recalc_all_degs_cos` 0.9 (≈26°), `lighting/probe_resolution` 256 | The probe re-renders on thresholds, which fits "lighting stays stale, then pops". |
| SSAO | — | `ssao/exterior_strength` 1, `ssao/detail_shadow_strength` 0 | |

## 4. Causes of the complaints, ranked by evidence

1. **Flat lighting (sim-wide).** There are four compounding causes:
   - the toe-less rational tone curve with no contrast look;
   - no cast shadows beyond 1 km;
   - ambient light too strong relative to direct (multi-scatter ×1.25 plus
     the sky profiles);
   - many surfaces with no roughness or normal data. Ortho4XP's `.ter` writes
     only `BASE_TEX_NOWRAP`, and Laminar's 12.2 notes say terrain now
     expects proper roughness/normals.

   The first three apply to everything in the frame, not just orthos.
2. **Washout follows the dominant content.** Untrimmed histogram metering plus
   the EV 14.8 ceiling: bright ground both drags the meter and exceeds the range
   the camera can compensate for.
3. **Milky atmosphere.** Multi-scatter ×1.25, the AgX inset desaturation, and
   the same shoulder compressing the sky.
4. **Cockpit vs exterior.** Exposure fusion exists but runs with `shadows` = 0.
5. **Transitions.** Adaptation speed is tunable. Probe re-render thresholds are
   tunable, but more frequent re-renders cost GPU time.

## 5. Test plan (in-sim, one hypothesis at a time)

`python scripts/xp12_lighting/art_controls.py apply <preset>`, restart X-Plane,
compare with `restore`. Every apply and restore backs up settings.txt first.

0. `validate`: confirms that settings.txt overrides reach controls Laminar
   never set (expect a brighter, red-tinted frame). **Run this first.** If
   nothing changes, the whole art-control route is dead and only shader
   replacement remains.
1. `contrast`: CDL power 1.2 (H1, flat look).
2. `exposure_range`: ev100_max 16 (H2, washout over desert/snow/haze).
3. `multiscatter`: multi_rat 1.0 (H3, milky atmosphere, lifted ambient).
4. `local_tonemap`: exposure_fusion/shadows 1.0 (H4, cockpit vs exterior).
5. `adaptation`: adaptation speeds ×2 (H5; direction of units unknown).
6. `ambient_occlusion`: exterior SSAO 1.5, detail shadows 0.5 (H6).

Suggested test scenes: a desert airport at noon (washout, transitions), a
hazy coastal city (atmosphere), cockpit view with the sun outside (exposure
fusion), and mountains from 3–5 km AGL (shadow distance, flatness).

Not covered by presets yet:
- **Shadow distance.** `far_limit_exterior` is set on Laminar's
  shadow-quality line, so the preset block could be overridden. It needs a
  direct edit of that line (backed up). Expect a cost in shadow resolution
  and performance.
- **Tone-curve shape.** A toe or real S-curve cannot be made from CDL alone;
  that needs a replacement `hdr`/`resolve` shader.
- **Ortho roughness/normals** in `.ter`: a separate Ortho4XP change.

## Sources

- Laminar, What Is a Photometric Renderer? — https://developer.x-plane.com/2021/11/photometric-lighting-what-is-it-and-why-do-we-need-it/
- Laminar, X-Plane 12.2.0 beta notes (AgX, cube maps, terrain roughness) — https://developer.x-plane.com/?p=41039
- Wronski, Exposure Fusion – local tonemapping for real-time rendering — https://bartwronski.com/2022/02/28/exposure-fusion-local-tonemapping-for-real-time-rendering/
- Mertens, Kautz, Van Reeth, Exposure Fusion (2007) — https://www.cs.princeton.edu/courses/archive/spring14/cos426/papers/Mertens07.pdf
- Daylight illuminance — https://en.wikipedia.org/wiki/Daylight
- Sunny 16 rule — https://en.wikipedia.org/wiki/Sunny_16_rule
- AgX look discussions — https://discourse.threejs.org/t/is-agx-tonemapping-implemented-correctly/60609 , https://discuss.pixls.us/t/filmic-vs-sigmoid-vs-agx-some-thoughts/54716
