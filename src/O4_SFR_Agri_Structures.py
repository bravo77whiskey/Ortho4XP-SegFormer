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
   otherwise open ground.
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

Everything is metric, so the texture zoom level does not matter.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

KIND_BUILDING = "building"
KIND_GREENHOUSE = "greenhouse"
KIND_GROUND = "ground"

# SegFormer (OpenEarthMap) classes, as in O4_SFR_Inference.
CLASS_ROAD = 4
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


def _obb_frame(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lengths))
    ux = edges[i] / max(float(lengths[i]), 1e-6)
    uy = np.array([-ux[1], ux[0]], dtype=np.float32)
    return pts, ux, uy


def _obb_patch(image, points):
    """The OBB interior resampled axis-aligned, long side along x."""
    pts, ux, uy = _obb_frame(points)
    rel = pts - pts[0]
    xs, ys = rel @ ux, rel @ uy
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
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


def classify_detections(detections, veg_map, image_fn, m_per_px, height_fn=None):
    """Return a list of (kind, reason) aligned with ``detections``.

    ``image_fn()`` returns the RGB texture (or None) and is only called when
    a verdict needs colour.  ``height_fn(dets)`` returns HeightNet heights
    (metres, NaN when unknown) for the long grey supported candidates;
    without it rule 3 relies on the farmland ring alone and rule 4 is off.
    """
    n = len(detections)
    out = [(KIND_BUILDING, None)] * n
    feats = [_features(d, veg_map, m_per_px) for d in detections]
    cand = []
    for i, ctx in enumerate(feats):
        if ctx is None:
            continue
        if ctx["in_building"] < SEG_BUILDING_FRAC:
            verdict = _unsupported_kind(ctx, image_fn)
            if verdict is not None:
                out[i] = verdict
        elif _is_long_grey_roof(ctx, image_fn):
            cand.append(i)
    if not cand:
        return out
    heights = np.full(len(cand), np.inf)
    if height_fn is not None:
        got = np.asarray(height_fn([detections[i] for i in cand]), dtype=np.float64)
        heights = np.where(np.isfinite(got), got, np.inf)
    height = dict(zip(cand, heights))
    for i in cand:
        ctx = feats[i]
        if ctx["aspect"] < GH_MIN_ASPECT:
            continue
        if ctx["ring_agri"] >= GH_RING_AGRI or (
            ctx["ring_agri"] > GH_SOME_AGRI and height[i] <= GH_MAX_HEIGHT_M
        ):
            out[i] = (KIND_GREENHOUSE, "long_film_roof")
    field = [i for i in cand if height[i] <= FIELD_MAX_HEIGHT_M]
    if not field:
        return out
    centres = np.array([_centre_px(d) for d in detections]) * float(m_per_px)
    for _ in range(FIELD_ROUNDS):
        gh = np.array([kind == KIND_GREENHOUSE for kind, _r in out])
        if not gh.any():
            break
        gh_centres = centres[gh]
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
# Greenhouses are placed from the O4SFR greenhouse library (virtual prefix
# o4sfr/agri/), two families:
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
GREENHOUSE_LIBRARY_PREFIX = "o4sfr/agri/"


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


def greenhouse_pieces(detection, m_per_px, northern=True):
    """Greenhouse objects covering one greenhouse detection.

    Returns a list of dicts: ``center`` (px), ``bearing`` (the OBJ heading,
    degrees clockwise from north), ``length_m``, ``width_m``, ``path``,
    ``kind`` ('tunnels' or 'solar') and the piece's pixel ``poly``.
    ``northern`` turns solar greenhouse walls to the north (else south).
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


def split_detections(detections, veg_map, image_loader, m_per_px, counts=None,
                     height_fn=None):
    """Split detections into (buildings, greenhouses); ground is dropped.

    ``image_loader`` is called at most once, lazily.  Greenhouse detections
    get ``structure='greenhouse'`` and ``tunnel_axis``.  ``counts`` (a dict)
    receives ``agri_<kind>_<reason>`` tallies.
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
        detections, veg_map, image_fn, m_per_px, height_fn=height_fn
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
