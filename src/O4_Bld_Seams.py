"""Biggest piece wins: one placement per building split at a seam.

The trained YOLO reads a texture through 512 px windows that do not overlap,
and each texture on its own, so a building crossing a window edge or a
texture edge is detected as separate pieces, each cut at the seam. Every piece
then gets its own smaller, often oddly shaped asset.

After the whole tile has been placed, pieces of one building that meet across
a seam are grouped and only the largest keeps its placement. On by default;
O4_BLD_SEAM_OWNER=0 turns it off. Judged visually on whole-tile review sets
(Gran Canaria, Taipei); a stitched-view alternative was tried and dropped.
"""
import os

import numpy as np

# Pieces end within this many pixels of the seam they were cut at (measured on
# the review set: YOLO's box edges pile up from -2 to +2 px around a window edge).
SEAM_TOUCH_PX = 2.5
# Mercator pixel space pieces from different zoom levels are compared in.
REF_ZL = 20


def owner_enabled():
    return os.environ.get("O4_BLD_SEAM_OWNER", "1").strip().lower() not in ("0", "false", "no", "off")


def seam_lines_touched(points, img_w, img_h, crop_stride=None, tol=SEAM_TOUCH_PX):
    """Seams a detection box was cut at: [(axis, coord_px, side, kind)].

    ``side`` is -1 when the box lies before the line and +1 after it; kind is
    'texture' for the image border and 'window' for a YOLO window edge.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = []
    for axis, size in ((0, int(img_w)), (1, int(img_h))):
        lo = float(pts[:, axis].min())
        hi = float(pts[:, axis].max())
        if -tol <= lo <= tol:
            out.append((axis, 0.0, 1, "texture"))
        if size - 1 - tol <= hi <= size + tol:
            out.append((axis, float(size), -1, "texture"))
        if crop_stride:
            stride = int(crop_stride)
            for line in range(stride, size, stride):
                if abs(hi - line) <= tol and lo < line - tol:
                    out.append((axis, float(line), -1, "window"))
                if abs(lo - line) <= tol and hi > line + tol:
                    out.append((axis, float(line), 1, "window"))
    return out


class SeamPiece:
    """A placement whose detection box ends on a seam, in global REF_ZL pixels."""

    __slots__ = ("placement", "area_m2", "quad_g", "lines_g", "px_g", "review")

    def __init__(self, placement, til_y, til_x, zl, points_px, area_m2, lines, review=None):
        s = 2.0 ** (REF_ZL - zl)
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        self.placement = placement
        self.area_m2 = float(area_m2)
        self.px_g = s                    # one texture pixel in REF_ZL pixels
        self.quad_g = np.column_stack(((pts[:, 0] + til_x * 256) * s, (pts[:, 1] + til_y * 256) * s))
        origin = (til_x * 256, til_y * 256)
        self.lines_g = [(a, (c + origin[a]) * s, side, k) for a, c, side, k in lines]
        self.review = review


def group_pieces(pieces):
    """Return (pieces to drop, {kind: count}) — all but the biggest of each group.

    Two pieces belong to one building when they end on the same seam line from
    opposite sides and their boxes touch (within a few pixels) across it.
    """
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    if not pieces:
        return [], {}
    polys = []
    for p in pieces:
        poly = Polygon(p.quad_g)
        polys.append(poly if poly.is_valid else poly.buffer(0))
    tree = STRtree(polys)
    parent = list(range(len(pieces)))
    link_kind = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    max_px_g = max(p.px_g for p in pieces)
    for i, piece in enumerate(pieces):
        grown = polys[i].buffer(SEAM_TOUCH_PX * max_px_g)
        for j in tree.query(grown):
            j = int(j)
            if j <= i:
                continue
            other = pieces[j]
            # A few pixels of slack at the coarser of the two textures.
            slack = SEAM_TOUCH_PX * max(piece.px_g, other.px_g)
            if polys[i].distance(polys[j]) > slack:
                continue
            for a, c, side, kind in piece.lines_g:
                if any(
                    oa == a and oside == -side and abs(oc - c) <= slack
                    for oa, oc, oside, _ok in other.lines_g
                ):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
                    link_kind[(i, j)] = kind
                    break
    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(i)
    drop, counts = [], {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda k: (pieces[k].area_m2, -k))
        member_set = set(members)
        kinds = {kind for (a, b), kind in link_kind.items() if a in member_set}
        label = "texture" if "texture" in kinds else "window"
        for k in members:
            if k != keep:
                drop.append(pieces[k])
                counts[label] = counts.get(label, 0) + 1
    return drop, counts
