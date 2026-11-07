"""Tell farm false positives and greenhouses apart from buildings.

The building detector fires on farmland: polytunnel arrays, plastic-covered
fields, furrowed plots, field roads.  Placed as ordinary houses and
warehouses they carpet the countryside.  This module sorts the surviving
YOLO detections of one texture into three kinds, using the landcover
SegFormer map the overlay already has, a few pixel statistics and, for a
handful of candidates, HeightNet:

* ``building``   -- place as before;
* ``greenhouse`` -- a covered crop structure: place a greenhouse asset;
* ``ground``     -- no structure (field, field road, plastic mulch): drop.

Rules (fitted on 420 hand-labelled placed detections from Wenzhou and
Yuncheng, then checked on a held-out Shouguang sample and 90 long grey
roofs from cities and greenhouse belts; Esri ZL16):

1. **Unsupported on farmland.**  Less than ``SEG_BUILDING_FRAC`` of the OBB
   is building class and interior plus a ring around it is mostly
   agriculture.  Real buildings there are small compact farm sheds (19 of
   21 labelled were <= 250 m2 and aspect <= 2.5), so those stay buildings;
   anything larger is a greenhouse when its interior is desaturated (film),
   otherwise open ground.  Dark blue-grey interiors are solar panels
   (``looks_like_solar_panels``), also ground.
2. **Unsupported on a road.**  Mostly road class inside: ground.
3. **Supported, greenhouse-shaped, beside farmland.**  The landcover model
   reads long film roofs as buildings.  Long (aspect >= 3.5), >= 400 m2,
   grey and not blue-tinted, and either with farmland around it or with some farmland and a low
   HeightNet height: greenhouse.
4. **Greenhouse fields.**  In dense belts (Shouguang) the model labels
   whole fields of greenhouses as buildings, so there is no farmland ring.
   A long grey low supported detection with two greenhouses (from 1-3)
   within ``FIELD_RADIUS_M`` becomes one too, repeated a few times so a
   field fills in from its edges.
5. **Ridged roofs.**  Rules 1-4 need grey film and long shapes; Korean
   vinyl houses (Gimhae, around RKPK) are teal or blue-grey multi-span
   blocks, often squat, with no farmland ring the landcover model can see.
   What every covered crop block shows is its tunnel ridges: a regular
   light/dark stripe with a 4-12 m pitch across the block.  A detection
   whose cross-ridge profile repeats at that pitch (``ridge_pattern``),
   and whose colour is film (grey, or teal to blue) rather than a painted
   roof (pale, or red to green), is a greenhouse whatever its landcover,
   provided HeightNet calls it low and it stands among fields (farmland
   around it, or from ZL17 on hardly any streets or yards, or stripes
   regular enough to need no surroundings): both keep apartment balconies
   and factory skylight strips out.  From ZL17 on a box one to three
   tunnels wide is judged by a wider window across it; weaker stripes
   join rule 4's field growth.  Checked on 678 hand-labelled Esri ZL18
   detections north of RKPK: score >= 0.3 found 84 of 119 greenhouse
   blocks with 1 building before the wider windows took most of the rest.

Everything is metric; only rule 5's looser tests depend on the texture
resolution (``RIDGE_FINE_MAX_MPP``), because ZL16 aliases roofs into stripes.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

KIND_BUILDING = "building"
KIND_GREENHOUSE = "greenhouse"
KIND_GROUND = "ground"

# SegFormer (OpenEarthMap) classes, as in O4_Landcover_Inference.
CLASS_DEVELOPED = 3
CLASS_ROAD = 4
CLASS_TREE = 5
CLASS_WATER = 6
CLASS_AGRICULTURE = 7
CLASS_BUILDING = 8

SEG_BUILDING_FRAC = 0.2   # below: the landcover model sees no roof here
RING_M = 8.0              # context ring around the OBB
AGRI_CONTEXT = 1.0        # interior + ring agriculture fractions, summed
ROAD_FRAC = 0.4           # unsupported and at least this much road: ground
SHED_MAX_M2 = 250.0       # unsupported farmland detections up to this...
SHED_MAX_ASPECT = 2.5     # ...and this compact stay buildings
FILM_MAX_SATURATION = 55.0  # mean HSV saturation (0-255): film is grey
# Rules 3-4 skip blue-tinted roofs: painted steel sheds and halls read
# grey-blue, film is neutral or green-tinted (removed 13 of 62 buildings
# from the candidates, 4 of 152 greenhouses).
ROOF_MAX_BLUE_TINT = 5.0  # mean B - R, uint8
# Rule 3
GH_MIN_ASPECT = 3.5
GH_MIN_M2 = 400.0
GH_RING_AGRI = 0.3        # this much farmland around: no height needed
GH_SOME_AGRI = 0.01       # ...or any farmland and a low roof
GH_MAX_HEIGHT_M = 11.0    # HeightNet raw metres
# Rule 4
FIELD_MIN_ASPECT = 3.0
FIELD_MAX_HEIGHT_M = 12.0
FIELD_RADIUS_M = 60.0     # centre distance, plus half the candidate's length
FIELD_MIN_NEIGHBOURS = 2
FIELD_ROUNDS = 3
# Rule 5
RIDGE_MIN_PERIOD_M = 4.0
RIDGE_MAX_PERIOD_M = 12.0
RIDGE_MIN_PERIODS = 4.0   # the profile must hold this many ridges
RIDGE_TREND_M = 18.0      # moving average removed from the profile
RIDGE_TRIM_M = 2.0        # dropped at each end of the profile
RIDGE_MIN_SCORE = 0.3     # greenhouse on its own
RIDGE_FIELD_SCORE = 0.1   # candidate for field growth (rule 4)
RIDGE_MIN_M2 = 250.0
RIDGE_FIELD_MIN_M2 = 60.0    # tunnel-shaped boxes; compact ones need RIDGE_MIN_M2
# HeightNet reads big greenhouse blocks as 10-20 m; apartment slabs, whose
# balconies and shadows also repeat, are far taller
RIDGE_MAX_HEIGHT_M = 20.0
# Covered crops stand among fields: at 2 m/px the skylight strips of grey
# factory roofs repeat like tunnels too, but nothing around them is
# farmland (Gimhae ZL16: 0 of 91 city hits had any within this radius,
# 81-100% of greenhouse hits had at least this share)
RIDGE_FARMLAND_RADIUS_M = 150.0
RIDGE_MIN_FARMLAND = 0.1
# Textures this fine (ZL17 on) resolve single tunnels; at ZL16 a tunnel is
# 3-4 px and rows of houses or skylight strips alias into the same stripes,
# so the looser tests below stay off there.
RIDGE_FINE_MAX_MPP = 1.2
# A dense belt squeezed between a river and a road has no field the model
# sees, but no streets or yards either: on fine textures an unbuilt
# neighbourhood will do.  Shenzhen/Kaohsiung ZL17-19 city hits are 15-30%
# developed space + road, the Gimhae riverbank belt 3%.
RIDGE_MAX_BUILT = 0.08
# Blocks beside a village fail both (streets count as built); from ZL17 on
# stripes this regular need no surroundings: 5 roofs on 6 Shenzhen and
# Kaohsiung ZL17-19 city textures, 17 greenhouse blocks on one Gimhae ZL17
RIDGE_SURE_SCORE = 1.2
# Korean tunnels are 6-8 m wide: on fine textures a box narrower than this
# (one to three tunnels) cannot hold four ridges and is judged by a window
# this wide across it (Gimhae ZL17/18: ~370 such boxes, 2 of them houses;
# at ZL16 nearly all were houses); boxes this elongated count as
# tunnel-shaped
RIDGE_CONTEXT_M = 36.0
RIDGE_TUNNEL_MIN_ASPECT = 2.0
# film is grey (low saturation) or teal to blue; painted roofs are pale
# (light blue, white) or red, orange, yellow, green
FILM_GREY_SATURATION = 40.0
FILM_MIN_HUE = 120.0      # degrees
FILM_MAX_HUE = 192.0      # painted blue roofs and solar panels lie above
PAINT_MIN_VALUE = 175.0
PAINT_PALE_VALUE = 160.0
PAINT_PALE_SATURATION = 45.0
# Solar panels on farmland are dark, grey and blue-tinted; greenhouse film,
# even teal film, leans green (B < G) or is more saturated.  Measured on a
# Wenzhou reclaimed-land PV farm against Shouguang, Wenzhou and Gimhae
# greenhouses (Esri/Bing ZL16-18).
PANEL_MAX_VALUE = 115.0       # mean HSV value (0-255)
PANEL_MIN_BLUE_RED = -8.0     # mean B - R
PANEL_MIN_BLUE_GREEN = -6.5   # mean B - G
PANEL_MAX_SATURATION = 34.0   # mean HSV saturation (0-255)


def _obb_frame(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lengths))
    ux = edges[i] / max(float(lengths[i]), 1e-6)
    uy = np.array([-ux[1], ux[0]], dtype=np.float32)
    return pts, ux, uy


def _obb_patch(image, points, min_short_px=0.0, shift=0, min_long_px=0.0):
    """The OBB interior resampled axis-aligned, long side along x.

    ``min_short_px`` widens a narrower OBB to that width, about its axis or,
    with ``shift`` -1 / +1, all to one side (the OBB at the other edge);
    ``min_long_px`` lengthens it about its centre.
    """
    pts, ux, uy = _obb_frame(points)
    rel = pts - pts[0]
    xs, ys = rel @ ux, rel @ uy
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    grow = float(min_short_px) - (y1 - y0)
    if grow > 0.0:
        y0 -= grow * (0.5 - 0.5 * shift)
        y1 += grow * (0.5 + 0.5 * shift)
    grow = 0.5 * (float(min_long_px) - (x1 - x0))
    if grow > 0.0:
        x0, x1 = x0 - grow, x1 + grow
    w = max(2, int(round(x1 - x0)))
    h = max(2, int(round(y1 - y0)))
    src = np.float32([
        pts[0] + ux * x0 + uy * y0,
        pts[0] + ux * x1 + uy * y0,
        pts[0] + ux * x0 + uy * y1,
    ])
    dst = np.float32([[0, 0], [w, 0], [0, h]])
    m = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def tunnel_axis(image, points):
    """'long' or 'short': which OBB axis the greenhouse ridges follow."""
    if image is None:
        return "long"
    gray = cv2.cvtColor(_obb_patch(image, points), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)
    # ridges along x make the row means vary; ridges along y the column means
    row_var = float(gray.mean(axis=1).std())
    col_var = float(gray.mean(axis=0).std())
    return "long" if row_var >= col_var else "short"


def ridge_pattern(image, points, m_per_px, across_m=None, shift=0, square=False):
    """How strongly the OBB interior repeats as tunnel ridges (0 = not at all).

    Takes the mean brightness profile across each OBB axis, removes its
    slow trend, and scores the best lag L in the ridge pitch range as
    min(acf(L), acf(2L)) - acf(L/2): a stripe pattern correlates with
    itself one and two pitches on and anti-correlates half a pitch on,
    a roof's light and dark halves or a few vents do not.

    With ``across_m`` the OBB is first widened to that many metres (see
    ``_obb_patch`` for ``shift``) and only ridges along its long axis count:
    a box one to three tunnels wide, judged by the block around it.  With
    ``square`` as well it is widened both ways and both axes count.
    """
    if image is None:
        return 0.0
    mpp = float(m_per_px)
    min_px = 0.0 if across_m is None else float(across_m) / mpp
    gray = cv2.cvtColor(
        _obb_patch(image, points, min_px, shift, min_px if square else 0.0),
        cv2.COLOR_RGB2GRAY,
    )
    gray = gray.astype(np.float32)
    profiles = (gray.mean(axis=1),) if across_m is not None and not square else (
        gray.mean(axis=0), gray.mean(axis=1))
    lags = np.arange(
        max(2, int(math.floor(RIDGE_MIN_PERIOD_M / mpp))),
        int(math.ceil(RIDGE_MAX_PERIOD_M / mpp)) + 1,
    )
    trend = max(3, int(round(RIDGE_TREND_M / mpp)) | 1)
    trim = int(round(RIDGE_TRIM_M / mpp))
    best = 0.0
    for prof in profiles:
        # the box edges often take in a road or field margin: drop them
        prof = prof[trim:len(prof) - trim]
        n = len(prof)
        if n * mpp < RIDGE_MIN_PERIODS * RIDGE_MIN_PERIOD_M:
            continue
        prof = prof - np.convolve(
            np.pad(prof, trend // 2, mode="edge"),
            np.ones(trend, np.float32) / trend, mode="valid",
        )
        spread = 2.5 * 1.4826 * float(np.median(np.abs(prof - np.median(prof))))
        if spread > 0.0:
            prof = np.clip(prof, -spread, spread)
        prof = prof - prof.mean()
        den = float(np.dot(prof, prof))
        if den <= 1e-6:
            continue

        def acf(lag):
            lag = int(round(lag))
            if lag <= 0 or lag >= n:
                return 0.0
            return float(np.dot(prof[:-lag], prof[lag:])) / den * n / (n - lag)

        for lag in lags[lags * RIDGE_MIN_PERIODS <= n]:
            best = max(best, min(acf(lag), acf(2 * lag)) - acf(lag / 2.0))
    return best


def detection_context(detection, veg_map, m_per_px):
    """Landcover fractions inside the OBB and in a ring around it.

    Returns None when the OBB does not reach the map.
    """
    pts = np.asarray(detection.get("points", ()), dtype=np.float32)
    if pts.shape != (4, 2):
        return None
    h, w = veg_map.shape[:2]
    ring_px = max(1, int(round(RING_M / float(m_per_px))))
    x0 = max(0, int(math.floor(float(pts[:, 0].min()))) - ring_px)
    y0 = max(0, int(math.floor(float(pts[:, 1].min()))) - ring_px)
    x1 = min(w, int(math.ceil(float(pts[:, 0].max()))) + ring_px + 1)
    y1 = min(h, int(math.ceil(float(pts[:, 1].max()))) + ring_px + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    inside = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(inside, [np.round(pts - (x0, y0)).astype(np.int32)], 1)
    k = 2 * ring_px + 1
    ring = cv2.dilate(inside, np.ones((k, k), np.uint8)) - inside
    inside = inside.astype(bool)
    ring = ring.astype(bool)
    n_in = int(inside.sum())
    if n_in == 0:
        return None
    n_ring = max(1, int(ring.sum()))
    win = veg_map[y0:y1, x0:x1]
    cls_in = win[inside]
    cls_ring = win[ring]
    return {
        "window": (x0, y0, x1, y1),
        "inside": inside,
        "in_building": float(np.count_nonzero(cls_in == CLASS_BUILDING)) / n_in,
        "in_agri": float(np.count_nonzero(cls_in == CLASS_AGRICULTURE)) / n_in,
        "in_road": float(np.count_nonzero(cls_in == CLASS_ROAD)) / n_in,
        "ring_agri": float(np.count_nonzero(cls_ring == CLASS_AGRICULTURE)) / n_ring,
    }


def interior_colour(image, ctx):
    """(mean HSV saturation, mean blue tint B - R) inside the OBB."""
    x0, y0, x1, y1 = ctx["window"]
    px = image[y0:y1, x0:x1][ctx["inside"]].astype(np.int16)
    if px.size == 0:
        return 0.0, 0.0
    vmax = px.max(axis=1)
    vmin = px.min(axis=1)
    sat = np.where(vmax > 0, (vmax - vmin) * 255 // np.maximum(vmax, 1), 0)
    return float(sat.mean()), float((px[:, 2] - px[:, 0]).mean())


def is_film_coloured(image, ctx):
    """Grey or teal-to-blue and not pale: film, not a painted roof."""
    x0, y0, x1, y1 = ctx["window"]
    px = image[y0:y1, x0:x1][ctx["inside"]]
    if px.size == 0:
        return False
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = float(hsv[:, 1].mean())
    val = float(hsv[:, 2].mean())
    if val >= PAINT_MIN_VALUE or (
        val >= PAINT_PALE_VALUE and sat >= PAINT_PALE_SATURATION
    ):
        return False
    if sat < FILM_GREY_SATURATION:
        return True
    hue = float(np.median(hsv[:, 0])) * 2.0
    return FILM_MIN_HUE <= hue <= FILM_MAX_HUE


def _panels_on():
    return os.environ.get("O4_BLD_GH_PANELS", "1").strip().lower() not in ("0", "false", "no", "off")


def looks_like_solar_panels(image, ctx):
    """Dark blue-grey interior: a solar farm, not greenhouse film."""
    if not _panels_on():
        return False
    x0, y0, x1, y1 = ctx["window"]
    return _panel_pixels(image[y0:y1, x0:x1][ctx["inside"]])


def _panel_pixels(px):
    """Solar-panel test on the mean colour of RGB pixels (N x 3)."""
    if px.size == 0:
        return False
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    r, g, b = px.astype(np.float32).mean(axis=0)
    _h, sat, val = hsv.astype(np.float32).mean(axis=0)
    return bool(val < PANEL_MAX_VALUE and sat < PANEL_MAX_SATURATION and
                b - r > PANEL_MIN_BLUE_RED and b - g > PANEL_MIN_BLUE_GREEN)


class _Countryside:
    """Whether a pixel lies among fields: farmland within
    RIDGE_FARMLAND_RADIUS_M, or (fine textures) hardly anything built.

    The shares are built on first use from the landcover map at a few
    metres per cell.
    """

    def __init__(self, veg_map, m_per_px):
        self.veg_map = veg_map
        self.m_per_px = float(m_per_px)
        self.farmland = None
        self.built = None
        self.step = max(1, int(round(4.0 / self.m_per_px)))

    def _share(self, mask):
        k = int(round(2.0 * RIDGE_FARMLAND_RADIUS_M / (self.m_per_px * self.step))) | 1
        return cv2.boxFilter(
            mask.astype(np.float32), -1, (k, k), borderType=cv2.BORDER_REFLECT
        )

    def __call__(self, x, y):
        cls = self.veg_map[::self.step, ::self.step]
        if self.farmland is None:
            self.farmland = self._share(cls == CLASS_AGRICULTURE)
        h, w = self.farmland.shape
        cy = min(h - 1, max(0, int(y) // self.step))
        cx = min(w - 1, max(0, int(x) // self.step))
        if self.farmland[cy, cx] >= RIDGE_MIN_FARMLAND:
            return True
        if self.m_per_px > RIDGE_FINE_MAX_MPP:
            return False
        if self.built is None:
            self.built = self._share((cls == CLASS_DEVELOPED) | (cls == CLASS_ROAD))
        return bool(self.built[cy, cx] <= RIDGE_MAX_BUILT)


def _ridge_score(detection, ctx, image_fn, m_per_px, countryside):
    """Rule 5 ridge score; 0 when too small, not film-coloured or not
    among fields (``countryside``: a _Countryside)."""
    width_m = float(detection.get("width_m") or 0.0)
    tunnel = ctx["aspect"] >= RIDGE_TUNNEL_MIN_ASPECT
    if ctx["area"] < (RIDGE_FIELD_MIN_M2 if tunnel else RIDGE_MIN_M2):
        return 0.0
    image = image_fn()
    if image is None or not is_film_coloured(image, ctx):
        return 0.0
    pts = detection["points"]
    score = ridge_pattern(image, pts, m_per_px)
    if width_m < RIDGE_CONTEXT_M and float(m_per_px) <= RIDGE_FINE_MAX_MPP:
        # too narrow to hold four ridges: judge it by the block around it,
        # with the box in the middle or at either edge of the block
        if tunnel:
            score = max([score] + [
                ridge_pattern(image, pts, m_per_px, across_m=RIDGE_CONTEXT_M, shift=s)
                for s in (0, -1, 1)
            ])
        else:
            score = max(score, ridge_pattern(
                image, pts, m_per_px, across_m=RIDGE_CONTEXT_M, square=True))
    if score < RIDGE_FIELD_SCORE:
        return 0.0
    if score >= RIDGE_SURE_SCORE and float(m_per_px) <= RIDGE_FINE_MAX_MPP:
        return score
    if not countryside(*_centre_px(detection)):
        return 0.0
    return score


def _aspect(detection):
    length = float(detection.get("length_m") or 0.0)
    width = float(detection.get("width_m") or 0.0)
    return length / width if width > 0.0 else 1.0


def _features(detection, veg_map, m_per_px):
    ctx = detection_context(detection, veg_map, m_per_px)
    if ctx is None:
        return None
    ctx["area"] = float(detection.get("area_m2") or 0.0)
    ctx["aspect"] = _aspect(detection)
    return ctx


def _unsupported_kind(ctx, image_fn):
    """Rules 1-2 for a detection without building class; None = building."""
    if ctx["in_road"] >= ROAD_FRAC:
        return KIND_GROUND, "road"
    if ctx["in_agri"] + ctx["ring_agri"] < AGRI_CONTEXT:
        return None
    if ctx["area"] <= SHED_MAX_M2 and ctx["aspect"] <= SHED_MAX_ASPECT:
        return KIND_BUILDING, "farm_shed"
    image = image_fn()
    if image is None:
        return KIND_GROUND, "farmland"
    saturation, _ = interior_colour(image, ctx)
    if saturation < FILM_MAX_SATURATION:
        if looks_like_solar_panels(image, ctx):
            return KIND_GROUND, "solar_panels"
        return KIND_GREENHOUSE, "farmland_film"
    return KIND_GROUND, "farmland"


def _is_long_grey_roof(ctx, image_fn):
    """Candidate for rules 3-4: supported, long, big enough and grey."""
    if ctx["aspect"] < FIELD_MIN_ASPECT or ctx["area"] < GH_MIN_M2:
        return False
    image = image_fn()
    if image is None:
        return False
    saturation, blue_tint = interior_colour(image, ctx)
    return saturation < FILM_MAX_SATURATION and blue_tint < ROOF_MAX_BLUE_TINT


def _centre_px(detection):
    return np.asarray(detection["points"], dtype=np.float64).reshape(4, 2).mean(axis=0)


def classify_detections(detections, veg_map, image_fn, m_per_px, height_fn=None,
                        seeds=()):
    """Return a list of (kind, reason) aligned with ``detections``.

    ``image_fn()`` returns the RGB texture (or None) and is only called when
    a verdict needs colour.  ``height_fn(dets)`` returns HeightNet heights
    (metres, NaN when unknown) for the long grey supported candidates and
    the ridged ones; without it rule 3 relies on the farmland ring alone,
    rule 4 is off and rule 5 does not check heights.  ``seeds`` are
    greenhouse detections already sorted (an earlier pass) that count as
    neighbours for rule 4.
    """
    n = len(detections)
    out = [(KIND_BUILDING, None)] * n
    feats = [_features(d, veg_map, m_per_px) for d in detections]
    countryside = _Countryside(veg_map, m_per_px)
    cand = []
    ridged = []
    strong = []
    for i, ctx in enumerate(feats):
        if ctx is None:
            continue
        if ctx["in_building"] < SEG_BUILDING_FRAC:
            verdict = _unsupported_kind(ctx, image_fn)
            if verdict is not None:
                out[i] = verdict
        elif _is_long_grey_roof(ctx, image_fn):
            cand.append(i)
        ridge = _ridge_score(detections[i], ctx, image_fn, m_per_px, countryside)
        if ridge >= RIDGE_MIN_SCORE and ctx["area"] >= RIDGE_MIN_M2:
            strong.append(i)
        elif ridge >= RIDGE_FIELD_SCORE and out[i][0] == KIND_BUILDING:
            ridged.append(i)
    in_cand = set(cand)
    ridged = [i for i in ridged if i not in in_cand]
    asked = cand + ridged + [i for i in strong if i not in in_cand]
    if not asked:
        return out
    heights = np.full(len(asked), np.inf)
    if height_fn is not None:
        got = np.asarray(height_fn([detections[i] for i in asked]), dtype=np.float64)
        heights = np.where(np.isfinite(got), got, np.inf)
    height = dict(zip(asked, heights))
    for i in strong:
        # apartment slabs repeat as balconies and shadows: greenhouses are low
        if height_fn is None or height[i] <= RIDGE_MAX_HEIGHT_M:
            out[i] = (KIND_GREENHOUSE, "ridged_roof")
    for i in cand:
        ctx = feats[i]
        if out[i][0] == KIND_GREENHOUSE or ctx["aspect"] < GH_MIN_ASPECT:
            continue
        if ctx["ring_agri"] >= GH_RING_AGRI or (
            ctx["ring_agri"] > GH_SOME_AGRI and height[i] <= GH_MAX_HEIGHT_M
        ):
            out[i] = (KIND_GREENHOUSE, "long_film_roof")
    field = [i for i in cand + ridged if height[i] <= FIELD_MAX_HEIGHT_M]
    if not field:
        return out
    centres = np.array([_centre_px(d) for d in detections]) * float(m_per_px)
    seed_centres = np.array(
        [_centre_px(d) for d in seeds], dtype=np.float64
    ).reshape(-1, 2) * float(m_per_px)
    for _ in range(FIELD_ROUNDS):
        gh = np.array([kind == KIND_GREENHOUSE for kind, _r in out])
        gh_centres = np.vstack([centres[gh], seed_centres])
        if not len(gh_centres):
            break
        grown = []
        for i in field:
            if out[i][0] != KIND_BUILDING:
                continue
            reach = FIELD_RADIUS_M + 0.5 * float(detections[i].get("length_m") or 0.0)
            dist = np.hypot(*(gh_centres - centres[i]).T)
            if int(np.count_nonzero(dist <= reach)) >= FIELD_MIN_NEIGHBOURS:
                grown.append(i)
        if not grown:
            break
        for i in grown:
            out[i] = (KIND_GREENHOUSE, "greenhouse_field")
    return out


def classify_detection(detection, veg_map, image_fn, m_per_px, height_fn=None):
    """(kind, reason) for one detection on its own (no field propagation)."""
    return classify_detections([detection], veg_map, image_fn, m_per_px, height_fn)[0]


# ── Greenhouse placement ────────────────────────────────────────────────────
# Greenhouses are placed from the O4XP greenhouse library (virtual prefix
# o4xp/agri/), two families:
#
# * tunnels_<W>x<L>.obj -- arrays of arched film tunnels, W = bays x
#   TUNNEL_PITCH_M across (OBJ x), L along the tunnels (OBJ z).  South China
#   style; a detection is covered by a grid of such pieces.
# * solar_<L>x<W>.obj -- the north-China "solar" greenhouse: one wide span,
#   L along its east-west ridge (OBJ x), W deep (OBJ z), an earthen wall on
#   the OBJ's -z side.  Used for single-span detections whose long axis runs
#   within SOLAR_MAX_EW_DEG of east-west; the wall is turned to the pole.
#
# Lengths come from one ladder; each piece takes the largest that fits.
TUNNEL_PITCH_M = 8.0
TUNNEL_MAX_BAYS = 12
TUNNEL_LENGTHS_M = tuple(range(10, 155, 5))
TUNNEL_MIN_LENGTH_M = 10.0
SOLAR_WIDTHS_M = tuple(range(8, 22, 2))
SOLAR_MAX_EW_DEG = 35.0
# Solar greenhouses come in two variants (anteroom at either end); a
# piece's variant follows from its position, so reruns place the same one.
SOLAR_VARIANTS = ("a", "b")
# The solar greenhouse is a north-China design.  East of China (Korea,
# Japan) single spans are film tunnels like the rest of the block.
SOLAR_MAX_LON = 123.5
GREENHOUSE_LIBRARY_PREFIX = "o4xp/agri/"


def tunnel_asset_path(bays, length_m):
    width_m = int(round(bays * TUNNEL_PITCH_M))
    return f"{GREENHOUSE_LIBRARY_PREFIX}tunnels_{width_m}x{int(length_m)}.obj"


def solar_asset_path(width_m, length_m, variant="a"):
    suffix = "" if variant == "a" else f"_{variant}"
    return f"{GREENHOUSE_LIBRARY_PREFIX}solar_{int(length_m)}x{int(width_m)}{suffix}.obj"


def _snap_length(length_m):
    """Largest ladder length not above ``length_m`` (None when too short)."""
    fits = [v for v in TUNNEL_LENGTHS_M if v <= length_m + 0.25]
    return float(fits[-1]) if fits else None


def _split(total, max_piece):
    n = max(1, int(math.ceil(total / max_piece - 1e-9)))
    return n, total / n


def _bearing(vec):
    """Compass bearing of an image-space direction (x east, y south)."""
    return math.degrees(math.atan2(float(vec[0]), -float(vec[1]))) % 360.0


def _piece(c, ax, bx, ca, cb, length_m, width_m, mpp, heading, path, **extra):
    center = c + (ax * ca + bx * cb) / mpp
    ha = ax * (length_m / 2.0) / mpp
    hb = bx * (width_m / 2.0) / mpp
    poly = np.array([
        center - ha - hb, center + ha - hb, center + ha + hb, center - ha + hb,
    ], dtype=np.float32)
    piece = {
        "center": (float(center[0]), float(center[1])),
        "bearing": heading,
        "length_m": length_m,
        "width_m": width_m,
        "path": path,
        "poly": poly,
    }
    piece.update(extra)
    return piece


def uses_solar_greenhouses(lon):
    """Whether single-span detections at ``lon`` may be solar greenhouses."""
    return float(lon) <= SOLAR_MAX_LON


def greenhouse_pieces(detection, m_per_px, northern=True, solar=True):
    """Greenhouse objects covering one greenhouse detection.

    Returns a list of dicts: ``center`` (px), ``bearing`` (the OBJ heading,
    degrees clockwise from north), ``length_m``, ``width_m``, ``path``,
    ``kind`` ('tunnels' or 'solar') and the piece's pixel ``poly``.
    ``northern`` turns solar greenhouse walls to the north (else south);
    ``solar=False`` places tunnels only.
    """
    pts, ux, uy = _obb_frame(detection["points"])
    c = pts.mean(axis=0)
    rel = pts - c
    long_px = float(np.ptp(rel @ ux))
    short_px = float(np.ptp(rel @ uy))
    if detection.get("tunnel_axis") == "short":
        ax, bx, along_px, across_px = uy, ux, short_px, long_px
    else:
        ax, bx, along_px, across_px = ux, uy, long_px, short_px
    mpp = float(m_per_px)
    along_m, across_m = along_px * mpp, across_px * mpp
    n_along, seg_m = _split(along_m, TUNNEL_LENGTHS_M[-1])
    length_m = _snap_length(seg_m)
    if length_m is None or length_m < TUNNEL_MIN_LENGTH_M:
        return []
    axis = _bearing(ax) % 180.0
    ew_off = abs(axis - 90.0)
    pieces = []
    if (
        solar and
        ew_off <= SOLAR_MAX_EW_DEG and
        SOLAR_WIDTHS_M[0] - 1.0 <= across_m <= SOLAR_WIDTHS_M[-1] + 2.0 and
        along_m >= 2.0 * across_m
    ):
        width_m = float(min(SOLAR_WIDTHS_M, key=lambda w: abs(w - across_m)))
        # OBJ x runs along the ridge (bearing heading + 90); its -z wall
        # faces bearing ``heading``: keep that within 90 deg of the pole.
        heading = (axis - 90.0) % 360.0
        if (northern and 90.0 < heading < 270.0) or (
            not northern and not 90.0 < heading < 270.0
        ):
            heading = (heading + 180.0) % 360.0
        for j in range(n_along):
            ca = -along_m / 2.0 + seg_m * (j + 0.5)
            centre = c + ax * ca / mpp
            pick = int(math.floor(centre[0] / 7.0) + math.floor(centre[1] / 7.0))
            variant = SOLAR_VARIANTS[pick % len(SOLAR_VARIANTS)]
            pieces.append(_piece(
                c, ax, bx, ca, 0.0, length_m, width_m, mpp, heading,
                solar_asset_path(width_m, length_m, variant), kind="solar",
            ))
        return pieces
    total_bays = max(1, int(round(across_m / TUNNEL_PITCH_M)))
    n_across, bays_f = _split(total_bays, TUNNEL_MAX_BAYS)
    bays_left = total_bays
    start_b = -total_bays * TUNNEL_PITCH_M / 2.0
    for i in range(n_across):
        bays = int(round(bays_f)) if i < n_across - 1 else bays_left
        bays = max(1, min(bays, bays_left))
        bays_left -= bays
        width_m = bays * TUNNEL_PITCH_M
        cb = start_b + width_m / 2.0
        start_b += width_m
        for j in range(n_along):
            ca = -along_m / 2.0 + seg_m * (j + 0.5)
            # tunnels run along OBJ z and are symmetric end to end
            pieces.append(_piece(
                c, ax, bx, ca, cb, length_m, width_m, mpp, axis,
                tunnel_asset_path(bays, length_m), kind="tunnels", bays=bays,
            ))
        if bays_left <= 0:
            break
    return pieces


# ── Greenhouse fields ───────────────────────────────────────────────────────
# YOLO finds a greenhouse belt piecemeal: some tunnels and solar houses come
# out as boxes, others not at all, others as buildings.  Placed box by box,
# a belt turns into a patchwork of greenhouses, bare ground and houses.
# Following the delineation chain of Koc-San & Sonmez (ISPRS 2016:
# classify, then closing and a sieve, then trace the field outline), the
# sorted greenhouses are joined into fields by a morphological closing;
# inside a field, film-coloured ground no box covers is tiled with rows
# laid out like the greenhouses around it, and low film-coloured buildings
# become greenhouses too.
FIELDS_CLOSE_M = 20.0       # bridges lanes and a missed solar row or a few tunnels
FIELDS_HOLE_M2 = 10000.0    # enclosed holes up to this are part of the field
FIELDS_MIN_SEEDS = 3        # greenhouses a field needs before it is filled
FIELDS_GROW_ROUNDS = 4      # growth through film-coloured ground, FIELDS_CLOSE_M a round
FIELDS_OPEN_M = 2.0         # slivers thinner than about twice this are dropped
FIELDS_SMOOTH_M = 3.0       # colour is judged on the texture blurred this much
FIELDS_MIN_GAP_M2 = 150.0   # uncovered film patches smaller than this stay bare
FIELDS_KEEP_OUT_M = 3.0     # clearance kept around remaining buildings
FIELDS_NEAR_M = 40.0        # greenhouses this close set a patch's row layout
FIELDS_RUN_COVER = 0.7      # share of a strip column that must be film
FIELDS_TUNNEL_BAYS = 2      # tunnel fill strips are this many bays wide
FIELDS_MEMBER_COVER = 0.8   # buildings this much inside a field may convert
FIELDS_MEMBER_FILM = 0.45   # ...when this share of their roof is film-coloured
FIELDS_MEMBER_PAINT = 0.15  # ...and less than this is painted blue
FIELDS_PRIOR_MIN_M2 = 2000.0  # mapped film area that makes a field with one greenhouse in it


def _film_pixels(image, m_per_px):
    """Per-pixel film colour (grey, or teal to blue, not pale paint) on the
    texture smoothed over FIELDS_SMOOTH_M."""
    k = max(1, int(round(FIELDS_SMOOTH_M / float(m_per_px)))) | 1
    img = cv2.blur(image, (k, k)) if k > 1 else image
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    sat = hsv[..., 1]
    val = hsv[..., 2]
    pale = (val >= PAINT_MIN_VALUE + 25) | (
        (val >= PAINT_PALE_VALUE) & (sat >= PAINT_PALE_SATURATION))
    grey = sat < FILM_GREY_SATURATION + 5
    tinted = (hue >= FILM_MIN_HUE) & (hue <= FILM_MAX_HUE + 8) & (sat < 120)
    return (grey | tinted) & ~pale & (val >= 45)


def _det_axis(det):
    """Unit vector (image px) of the greenhouse rows' long direction."""
    pts, ux, uy = _obb_frame(det["points"])
    return uy if det.get("tunnel_axis") == "short" else ux


def _axis_angle(vecs, weights=None):
    """Mean undirected angle (radians, image frame) of unit vectors."""
    v = np.asarray(vecs, dtype=np.float64).reshape(-1, 2)
    a2 = 2.0 * np.arctan2(v[:, 1], v[:, 0])
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=np.float64)
    return 0.5 * math.atan2(float((np.sin(a2) * w).sum()), float((np.cos(a2) * w).sum()))


def _row_layout(seeds, angle, m_per_px, solar):
    """(strip width m, pitch m, phase m, rotated-frame across unit) for a patch.

    Solar houses stand in rows with soil between: width and pitch come from
    the neighbouring greenhouses.  Tunnel blocks are contiguous.
    """
    mpp = float(m_per_px)
    across = np.array([-math.sin(angle), math.cos(angle)])
    offs, widths = [], []
    for det in seeds:
        pts = np.asarray(det["points"], dtype=np.float64).reshape(4, 2) * mpp
        proj = pts @ across
        offs.append((float(proj.min()), float(proj.max())))
        widths.append(float(proj.max() - proj.min()))
    if not solar:
        w = FIELDS_TUNNEL_BAYS * TUNNEL_PITCH_M
        # strip centres half a strip in from the greenhouses' edges
        phase = (float(np.median([o[0] for o in offs])) + 0.5 * w) % w if offs else 0.0
        return w, w, phase, across
    w = float(np.clip(np.median(widths), SOLAR_WIDTHS_M[0], SOLAR_WIDTHS_M[-1]))
    centres = np.sort(np.array([0.5 * (a + b) for a, b in offs]))
    diffs = np.diff(centres)
    diffs = diffs[(diffs >= 0.9 * w) & (diffs <= 4.0 * w)]
    pitch = float(np.percentile(diffs, 25)) if len(diffs) >= 2 else 1.8 * w
    pitch = max(pitch, w + 2.0)
    # phase: circular median of the row centres modulo the pitch
    ang = 2.0 * math.pi * (centres % pitch) / pitch
    phase = (math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
             / (2.0 * math.pi) * pitch) % pitch
    return w, pitch, phase, across


def _strip_runs(mask, angle, width_m, pitch_m, phase_m, m_per_px, offset_px):
    """Rectangles (image px corner lists) tiling ``mask`` in strips.

    Strips run along ``angle``, ``width_m`` wide every ``pitch_m``, their
    centres at ``phase_m`` modulo the pitch (metres along the across axis,
    in texture coordinates offset by ``offset_px``); ``phase_m`` None tries
    phases about a pixel apart and keeps the one whose strips cover the most
    of ``mask``.  A run is a stretch of strip columns at least
    FIELDS_RUN_COVER film and TUNNEL_MIN_LENGTH_M long.
    """
    mpp = float(m_per_px)
    h, w = mask.shape
    ca, sa = math.cos(angle), math.sin(angle)
    ux = np.array([ca, sa])
    uy = np.array([-sa, ca])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
    px = corners @ ux
    py = corners @ uy
    x0, y0 = float(px.min()), float(py.min())
    W = int(math.ceil(px.max() - x0)) + 1
    H = int(math.ceil(py.max() - y0)) + 1
    # rotated frame (u, v) -> image: p = u*ux + v*uy, with u = x0 + col
    m = np.array([[ux[0], uy[0], x0 * ux[0] + y0 * uy[0]],
                  [ux[1], uy[1], x0 * ux[1] + y0 * uy[1]]], dtype=np.float64)
    rot = cv2.warpAffine(mask.astype(np.uint8), m, (W, H),
                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                         borderValue=0)
    if not rot.any():
        return []
    # the across coordinate of row v in texture metres
    off = np.asarray(offset_px, dtype=np.float64)
    v_texture_m = (y0 + off @ uy) * mpp
    wp = width_m / mpp
    min_len = TUNNEL_MIN_LENGTH_M / mpp
    rows = np.nonzero(rot.any(axis=1))[0]
    # column sums of any band of rows: cum[rb] - cum[ra]
    cum = np.zeros((H + 1, W), np.int32)
    np.cumsum(rot, axis=0, out=cum[1:])

    def runs(phase):
        k0 = int(math.floor(((rows[0] * mpp + v_texture_m) - phase) / pitch_m)) - 1
        k1 = int(math.ceil(((rows[-1] * mpp + v_texture_m) - phase) / pitch_m)) + 1
        found, covered = [], 0
        for k in range(k0, k1 + 1):
            vc = (phase + k * pitch_m - v_texture_m) / mpp
            va, vb = vc - wp / 2.0, vc + wp / 2.0
            ra, rb = max(0, int(round(va))), min(H, int(round(vb)))
            if rb - ra < max(1.0, 0.6 * wp):
                continue
            band = cum[rb] - cum[ra]
            cover = band >= FIELDS_RUN_COVER * (rb - ra)
            if not cover.any():
                continue
            d = np.diff(np.concatenate([[0], cover.astype(np.int8), [0]]))
            for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                if e - s < min_len:
                    continue
                covered += int(band[s:e].sum())
                quad = np.array([[s, va], [e, va], [e, vb], [s, vb]], dtype=np.float64)
                found.append((quad[:, :1] + x0) * ux + (quad[:, 1:] + y0) * uy + off)
        return found, covered

    if phase_m is not None:
        return runs(float(phase_m))[0]
    step = max(mpp, float(pitch_m) / 40.0)
    best, best_cov = [], 0
    for ph in np.arange(0.0, float(pitch_m), step):
        found, covered = runs(float(ph))
        if covered > best_cov:
            best, best_cov = found, covered
    return best


def _fill_detection(pts, m_per_px):
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    along = float(np.linalg.norm(p[1] - p[0])) * float(m_per_px)
    across = float(np.linalg.norm(p[3] - p[0])) * float(m_per_px)
    return {
        "points": p.astype(np.float32),
        "confidence": 0.0,
        "length_m": max(along, across),
        "width_m": min(along, across),
        "area_m2": along * across,
        "structure": KIND_GREENHOUSE,
        "tunnel_axis": "long" if along >= across else "short",
        "greenhouse_fill": True,
    }


def _grow_field(seed, cand, mpp):
    """Field mask: the greenhouses closed over FIELDS_CLOSE_M, grown a few
    rounds through film-coloured ground (``cand``) within that reach, and
    with enclosed holes up to FIELDS_HOLE_M2 filled.  Worked at >= 1 m/px."""
    h, w = seed.shape
    f = max(1, int(1.0 / mpp))
    wm = mpp * f
    if f > 1:
        size = (w // f, h // f)
        seed_s = (cv2.resize(seed, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        cand_s = (cv2.resize(cand, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    else:
        seed_s, cand_s = seed, cand
    r = max(1, int(round(FIELDS_CLOSE_M / wm)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    field = cv2.morphologyEx(seed_s, cv2.MORPH_CLOSE, disk)
    for _ in range(FIELDS_GROW_ROUNDS):
        reach = cv2.dilate(field, disk)
        grown = cv2.morphologyEx(field | (cand_s & reach), cv2.MORPH_CLOSE, disk)
        if np.array_equal(grown, field):
            break
        field = grown
    hh, ww = field.shape
    inv = (1 - field).astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(inv, connectivity=4)
    x, y, bw, bh, area = (stats[:, j] for j in range(5))
    hole = ((area <= FIELDS_HOLE_M2 / (wm * wm)) & (x > 0) & (y > 0) &
            (x + bw < ww) & (y + bh < hh))
    hole[0] = False
    field = field | hole[lab].astype(np.uint8)
    if f > 1:
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_NEAREST)
    return field


def greenhouse_fields(greenhouses, buildings, veg_map, image, m_per_px,
                      height_fn=None, solar=True, counts=None, prior=None):
    """Join greenhouses into fields and complete them.

    Returns (buildings kept, buildings turned greenhouse, fill detections).
    Fill detections are OBB-like dicts (``greenhouse_fill`` True) placed
    like any greenhouse detection; converted buildings get
    ``structure='greenhouse'`` and a ``tunnel_axis``.  ``prior`` is an
    optional uint8 mask of mapped greenhouse land (O4_Greenhouse_Map): its
    film-coloured ground seeds fields too.
    """
    if prior is not None and not prior.any():
        prior = None
    if image is None or veg_map is None or (
            prior is None and len(greenhouses) < FIELDS_MIN_SEEDS):
        return list(buildings), [], []
    mpp = float(m_per_px)
    h, w = veg_map.shape[:2]
    seed = np.zeros((h, w), np.uint8)
    for det in greenhouses:
        cv2.fillPoly(seed, [np.round(np.asarray(det["points"])).astype(np.int32)], 1)
    film = _film_pixels(image, mpp)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    painted = ((hsv[..., 0] > (FILM_MAX_HUE + 8.0) / 2.0) & (hsv[..., 1] > 50))
    del hsv

    def filmy(ctx):
        # bright film fails is_film_coloured's paint test; the smoothed
        # per-pixel film share inside the box tells it from paint too.
        # Painted blue steel roofs are never film.
        x0, y0, x1, y1 = ctx["window"]
        inside = ctx["inside"]
        if float(painted[y0:y1, x0:x1][inside].mean()) >= FIELDS_MEMBER_PAINT:
            return False
        return (float(film[y0:y1, x0:x1][inside].mean()) >= FIELDS_MEMBER_FILM
                or is_film_coloured(image, ctx))

    bld = np.zeros((h, w), np.uint8)
    rowlike = np.zeros((h, w), np.uint8)
    for det in buildings:
        poly = [np.round(np.asarray(det["points"])).astype(np.int32)]
        cv2.fillPoly(bld, poly, 1)
        # long film-coloured roofs the rules kept: the field may grow over
        # them, so a row of missed greenhouses can still join it
        if _aspect(det) >= FIELD_MIN_ASPECT:
            ctx = _features(det, veg_map, mpp)
            if ctx is not None and filmy(ctx):
                cv2.fillPoly(rowlike, poly, 1)
    # car parks and yards are grey too: film ground is never road, water,
    # trees or developed land
    ground_ok = ~np.isin(veg_map, (CLASS_ROAD, CLASS_WATER, CLASS_TREE, CLASS_DEVELOPED))
    cand = (film & ground_ok & (bld == 0)).astype(np.uint8)
    grow = cand | rowlike
    mapped = None
    if prior is not None:
        mapped = (prior.astype(bool) & cand.astype(bool)).astype(np.uint8)
        ro = max(1, int(round(FIELDS_OPEN_M / mpp)))
        mapped = cv2.morphologyEx(mapped, cv2.MORPH_OPEN,
                                  np.ones((2 * ro + 1, 2 * ro + 1), np.uint8))
    field = _grow_field(seed if mapped is None else (seed | mapped), grow, mpp)
    # fields need a few greenhouses: one stray seed fills nothing
    n, lab, stats, _c = cv2.connectedComponentsWithStats(field, connectivity=8)
    seeds_per = np.zeros(n, np.int32)
    for det in greenhouses:
        cx, cy = _centre_px(det)
        seeds_per[int(lab[min(h - 1, max(0, int(cy))), min(w - 1, max(0, int(cx)))])] += 1
    good = seeds_per >= FIELDS_MIN_SEEDS
    if mapped is not None:
        # the map is years older than the imagery (2020 Sentinel-2): mapped
        # land needs one sign of greenhouses today too, a detected one or a
        # long film roof, or a drained lagoon would fill
        mapped_px = np.bincount(lab[mapped > 0].ravel(), minlength=n)
        row_px = np.bincount(lab[rowlike > 0].ravel(), minlength=n)
        good |= ((mapped_px * mpp * mpp >= FIELDS_PRIOR_MIN_M2) &
                 ((seeds_per >= 1) | (row_px > 0)))
    good[0] = False
    field = good[lab].astype(np.uint8)
    if not field.any():
        return list(buildings), [], []

    # buildings standing inside a field: low film-coloured roofs join it
    kept, converted = [], []
    asked = []
    for det in buildings:
        ctx = _features(det, field, mpp)
        if ctx is None:
            kept.append(det)
            continue
        x0, y0, x1, y1 = ctx["window"]
        inside = float(field[y0:y1, x0:x1][ctx["inside"]].mean())
        if (inside >= FIELDS_MEMBER_COVER and
                not (ctx["area"] <= SHED_MAX_M2 and ctx["aspect"] <= SHED_MAX_ASPECT) and
                filmy(ctx)):
            asked.append(det)
        else:
            kept.append(det)
    if asked:
        heights = np.full(len(asked), -np.inf)
        if height_fn is not None:
            got = np.asarray(height_fn(asked), dtype=np.float64)
            heights = np.where(np.isfinite(got), got, -np.inf)
        for det, hgt in zip(asked, heights):
            if hgt <= FIELD_MAX_HEIGHT_M:
                det["structure"] = KIND_GREENHOUSE
                det["tunnel_axis"] = tunnel_axis(image, det["points"])
                converted.append(det)
            else:
                kept.append(det)

    # what to fill: film-coloured field ground no greenhouse or building covers
    covered = seed.copy()
    for det in converted:
        cv2.fillPoly(covered, [np.round(np.asarray(det["points"])).astype(np.int32)], 1)
    covered = cv2.dilate(covered, np.ones((3, 3), np.uint8))
    blocked = np.zeros((h, w), np.uint8)
    for det in kept:
        cv2.fillPoly(blocked, [np.round(np.asarray(det["points"])).astype(np.int32)], 1)
    ko = max(1, int(round(FIELDS_KEEP_OUT_M / mpp)))
    blocked = cv2.dilate(blocked, np.ones((2 * ko + 1, 2 * ko + 1), np.uint8))
    gap = (field.astype(bool) & film & ground_ok & ~covered.astype(bool) & ~blocked.astype(bool))
    gap = gap.astype(np.uint8)
    ro = max(1, int(round(FIELDS_OPEN_M / mpp)))
    gap = cv2.morphologyEx(gap, cv2.MORPH_OPEN, np.ones((2 * ro + 1, 2 * ro + 1), np.uint8))

    all_gh = list(greenhouses) + converted
    centres = np.array([_centre_px(d) for d in all_gh], dtype=np.float64).reshape(-1, 2)
    axes = [_det_axis(d) for d in all_gh]
    solar_kind = [
        bool(solar) and any(p["kind"] == "solar" for p in greenhouse_pieces(d, mpp)[:1])
        for d in all_gh
    ]
    near_px = FIELDS_NEAR_M / mpp
    min_gap_px = FIELDS_MIN_GAP_M2 / (mpp * mpp)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(gap, connectivity=8)
    fills = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_gap_px:
            continue
        sel = []
        for grow in (1, 4):
            # a hole deep in a field looks further for its neighbours
            pad = int(math.ceil(near_px * grow))
            wx0, wy0 = x - pad, y - pad
            wx1, wy1 = x + bw + pad, y + bh + pad
            sel = np.nonzero(
                (centres[:, 0] >= wx0) & (centres[:, 0] < wx1) &
                (centres[:, 1] >= wy0) & (centres[:, 1] < wy1))[0].tolist()
            if sel:
                break
        near = [all_gh[j] for j in sel]
        sub = (lab[y:y + bh, x:x + bw] == i)
        if _panels_on() and _panel_pixels(image[y:y + bh, x:x + bw][sub]):
            continue  # a solar farm between panel boxes, not missed film
        if near:
            angle = _axis_angle([axes[j] for j in sel],
                                [float(all_gh[j].get("area_m2") or 1.0) for j in sel])
        else:
            # mapped land with no greenhouse box anywhere: rows along the patch
            ys, xs = np.nonzero(sub)
            (_cx, _cy), (rw, rh), ra = cv2.minAreaRect(
                np.column_stack([xs, ys]).astype(np.float32))
            angle = math.radians(ra if rw >= rh else ra + 90.0)
        is_solar = bool(sel) and sum(solar_kind[j] for j in sel) * 2 > len(sel)
        width_m, pitch_m, _phase, _across = _row_layout(near, angle, mpp, is_solar)
        # the neighbours give width and pitch; the rows' position comes from
        # the gap itself (a missed row rarely sits on the field's mean phase)
        for pts in _strip_runs(sub, angle, width_m, pitch_m, None, mpp, (x, y)):
            fills.append(_fill_detection(pts, mpp))
    if counts is not None:
        counts["agri_greenhouse_field_member"] = (
            counts.get("agri_greenhouse_field_member", 0) + len(converted))
        counts["agri_greenhouse_field_fill"] = (
            counts.get("agri_greenhouse_field_fill", 0) + len(fills))
    return kept, converted, fills


def split_detections(detections, veg_map, image_loader, m_per_px, counts=None,
                     height_fn=None, seeds=()):
    """Split detections into (buildings, greenhouses); ground is dropped.

    ``image_loader`` is called at most once, lazily.  Greenhouse detections
    get ``structure='greenhouse'`` and ``tunnel_axis``.  ``counts`` (a dict)
    receives ``agri_<kind>_<reason>`` tallies.  ``seeds``: see
    ``classify_detections``.
    """
    if veg_map is None or not detections:
        return list(detections or ()), []
    state = {"loaded": False, "image": None}

    def image_fn():
        if not state["loaded"]:
            state["loaded"] = True
            state["image"] = image_loader() if image_loader is not None else None
        return state["image"]

    verdicts = classify_detections(
        detections, veg_map, image_fn, m_per_px, height_fn=height_fn, seeds=seeds
    )
    buildings = []
    greenhouses = []
    for det, (kind, reason) in zip(detections, verdicts):
        if reason is not None and counts is not None:
            key = f"agri_{kind}_{reason}"
            counts[key] = counts.get(key, 0) + 1
        if kind == KIND_BUILDING:
            buildings.append(det)
        elif kind == KIND_GREENHOUSE:
            det["structure"] = KIND_GREENHOUSE
            det["tunnel_axis"] = tunnel_axis(image_fn(), det["points"])
            greenhouses.append(det)
    return buildings, greenhouses
