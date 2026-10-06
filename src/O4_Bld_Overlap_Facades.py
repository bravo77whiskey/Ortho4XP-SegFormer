"""Grazing detections become touching or merged facades instead of one dropping.

Overlap suppression keeps a single detection per building. When two boxes only
graze (their overlap is a small part of either one), the smaller one used to
win and the bigger one was dropped outright, however little they touched:
with two large buildings that threw away the biggest thing in the area over a
corner graze of YOLO's loose boxes.

Now suppression keeps both and records the pair (O4_BLD_OVERLAP_FACADES,
on by default; =0 restores the drop).
After heights are known, each grazing group is resolved into polygon facades:

- merge: neighbours of the same facade class whose heights are within
  O4_BLD_OVERLAP_MERGE_DH_M (6 m) join into one facade (at most
  O4_BLD_OVERLAP_MERGE_MAX detections), taking the height of the biggest;
- split: otherwise the higher-confidence side keeps the overlap and the other
  is trimmed back to meet it. Within O4_BLD_OVERLAP_CONF_TIE (0.05) the
  overlap is cut down the middle instead.

A detection that keeps its whole box stays a normal detection (object or
facade). A trimmed one that keeps less than MIN_KEEP_FRAC of its ground, or
pieces thinner than MIN_SIDE_M, is dropped. Judged visually on the dense
Taipei and Shenzhen review textures (2026-10-06).
"""
import os

import numpy as np

MERGE_DH_M = 6.0
MERGE_MAX = 3
CONF_TIE = 0.05
MIN_KEEP_FRAC = 0.4
MIN_SIDE_M = 3.0
SIMPLIFY_M = 0.25
# Morphological opening of trimmed pieces: removes the hair-thin spikes a cut
# leaves where two box edges nearly coincide.
OPEN_M = 0.5


def enabled():
    return os.environ.get("O4_BLD_OVERLAP_FACADES", "1").strip().lower() not in ("0", "false", "no", "off")


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Node:
    """One grazing detection: its box in texture pixels plus what decides its fate."""

    __slots__ = ("key", "points", "conf", "area_m2", "height_m", "facade_cls")

    def __init__(self, key, points, conf, area_m2, height_m, facade_cls):
        self.key = key
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.conf = float(conf)
        self.area_m2 = float(area_m2)
        self.height_m = height_m
        self.facade_cls = facade_cls


class Outcome:
    """kind: 'merged' (rings, representative), 'member' (absorbed into ``rep``),
    'split' (trimmed rings), 'dropped' (too little left), 'unchanged'."""

    __slots__ = ("kind", "rings", "rep", "members")

    def __init__(self, kind, rings=None, rep=None, members=None):
        self.kind = kind
        self.rings = rings
        self.rep = rep
        self.members = members or []


def _long_axis_cut(geom, through):
    """A line through ``through`` along ``geom``'s long axis, long enough to cross it.

    Returns (line, point on it, unit direction) or None.
    """
    from shapely.geometry import LineString

    rect = geom.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords, dtype=np.float64)[:4]
    if len(coords) < 4:
        return None
    e1 = coords[1] - coords[0]
    e2 = coords[2] - coords[1]
    axis = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    norm = float(np.hypot(*axis))
    if norm <= 1e-9:
        return None
    axis = axis / norm
    minx, miny, maxx, maxy = geom.bounds
    reach = float(np.hypot(maxx - minx, maxy - miny)) + 1.0
    c = np.asarray(through, dtype=np.float64)
    return LineString([tuple(c - axis * reach), tuple(c + axis * reach)]), c, axis


def _split_down_the_middle(inter, geom_a, geom_b):
    """Return (part B loses, part A loses): each half goes to the nearer group."""
    from shapely.ops import split, unary_union

    cut = _long_axis_cut(inter, inter.centroid.coords[0])
    if cut is None:
        return None
    line, c, axis = cut
    try:
        halves = [g for g in split(inter, line).geoms if g.area > 0]
    except Exception:
        halves = []
    if len(halves) < 2:
        return None

    def side(pt):
        v = np.asarray(pt.coords[0], dtype=np.float64) - c
        return float(axis[0] * v[1] - axis[1] * v[0])

    ca, cb = geom_a.centroid, geom_b.centroid
    sa, sb = side(ca), side(cb)
    a_keeps, b_keeps = [], []
    for half in halves:
        hc = half.centroid
        sh = side(hc)
        if sa * sb < 0:
            # the boxes lie on opposite sides of the cut: each keeps its side
            a_side = sh * sa > 0
        else:
            a_side = hc.distance(ca) <= hc.distance(cb)
        (a_keeps if a_side else b_keeps).append(half)
    if not a_keeps or not b_keeps:
        return None
    return unary_union(a_keeps), unary_union(b_keeps)


def _clean_pieces(geom, min_side_px):
    from shapely.geometry import MultiPolygon, Polygon

    if geom.is_empty:
        return []
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else (
        [geom] if isinstance(geom, Polygon) else
        [g for g in getattr(geom, 'geoms', ()) if isinstance(g, Polygon)]
    )
    out = []
    for poly in polys:
        if poly.is_empty or poly.area <= 0:
            continue
        rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords, dtype=np.float64)
        if len(rect) >= 4:
            short = min(np.hypot(*(rect[1] - rect[0])), np.hypot(*(rect[2] - rect[1])))
            if short < min_side_px:
                continue
        out.append(poly)
    return out


def resolve(nodes, edges, m_per_px, merge_dh_m=None, merge_max=None, conf_tie=None,
            min_keep_frac=MIN_KEEP_FRAC, min_side_m=MIN_SIDE_M):
    """Resolve grazing pairs; return {node key: Outcome}.

    ``nodes`` maps key -> Node, ``edges`` lists (key_a, key_b) grazing pairs.
    Nodes absent from every edge are not returned.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    merge_dh_m = _env_float("O4_BLD_OVERLAP_MERGE_DH_M", MERGE_DH_M) if merge_dh_m is None else merge_dh_m
    merge_max = int(_env_float("O4_BLD_OVERLAP_MERGE_MAX", MERGE_MAX)) if merge_max is None else merge_max
    conf_tie = _env_float("O4_BLD_OVERLAP_CONF_TIE", CONF_TIE) if conf_tie is None else conf_tie
    m_per_px = max(float(m_per_px), 1e-6)
    min_side_px = float(min_side_m) / m_per_px

    geoms = {}
    for key, node in nodes.items():
        g = Polygon(node.points)
        if not g.is_valid:
            g = g.buffer(0)
        geoms[key] = g
    pairs = []
    seen = set()
    for a, b in edges:
        if a == b or a not in geoms or b not in geoms:
            continue
        k = (a, b) if str(a) <= str(b) else (b, a)
        if k in seen:
            continue
        seen.add(k)
        inter = geoms[a].intersection(geoms[b]).area
        if inter > 0:
            pairs.append((inter, a, b))
    if not pairs:
        return {}

    # Merge groups: union-find, biggest overlaps first, capped in size and
    # only between neighbours of one facade class and similar height.
    parent = {}
    members = {}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for _inter, a, b in pairs:
        for k in (a, b):
            if k not in parent:
                parent[k] = k
                members[k] = [k]
    for _inter, a, b in sorted(pairs, key=lambda p: -p[0]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        group = members[ra] + members[rb]
        if len(group) > merge_max:
            continue
        heights = [nodes[k].height_m for k in group]
        if any(h is None for h in heights) or max(heights) - min(heights) > merge_dh_m:
            continue
        if len({nodes[k].facade_cls for k in group}) != 1:
            continue
        parent[rb] = ra
        members[ra] = group
        del members[rb]

    roots = list(members)
    group_geom = {}
    group_conf = {}
    for r in roots:
        group_geom[r] = unary_union([geoms[k] for k in members[r]]) if len(members[r]) > 1 else geoms[r]
        group_conf[r] = max(nodes[k].conf for k in members[r])

    # Split what remains between groups: the stronger side keeps the overlap,
    # a near tie is cut down the middle.
    lost = {r: [] for r in roots}
    done = set()
    for _inter, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        k = (ra, rb) if str(ra) <= str(rb) else (rb, ra)
        if k in done:
            continue
        done.add(k)
        ga, gb = group_geom[ra], group_geom[rb]
        inter = ga.intersection(gb)
        if inter.is_empty or inter.area <= 0:
            continue
        ca, cb = group_conf[ra], group_conf[rb]
        halves = None
        if abs(ca - cb) <= conf_tie:
            halves = _split_down_the_middle(inter, ga, gb)
        if halves is not None:
            a_keeps, b_keeps = halves
            lost[rb].append(a_keeps)
            lost[ra].append(b_keeps)
        elif ca >= cb:
            lost[rb].append(inter)
        else:
            lost[ra].append(inter)

    simplify_px = SIMPLIFY_M / m_per_px
    open_px = max(OPEN_M / m_per_px, 0.5)
    out = {}
    for r in roots:
        group = members[r]
        rep = max(group, key=lambda k: (nodes[k].area_m2, nodes[k].conf))
        geom = group_geom[r]
        if lost[r]:
            geom = geom.difference(unary_union(lost[r]))
        if len(group) == 1 and not lost[r]:
            out[r] = Outcome('unchanged')
            continue
        geom = geom.buffer(-open_px, join_style=2).buffer(open_px, join_style=2)
        pieces = _clean_pieces(geom, min_side_px)
        kept_area = sum(p.area for p in pieces)
        if not pieces or kept_area < float(min_keep_frac) * group_geom[r].area:
            for k in group:
                out[k] = Outcome('dropped')
            continue
        rings = []
        for p in pieces:
            p = p.simplify(simplify_px, preserve_topology=True)
            ring = np.asarray(p.exterior.coords, dtype=np.float64)
            if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            if len(ring) >= 3:
                rings.append(ring)
        if not rings:
            for k in group:
                out[k] = Outcome('dropped')
            continue
        kind = 'merged' if len(group) > 1 else 'split'
        out[rep] = Outcome(kind, rings=rings, rep=rep, members=list(group))
        for k in group:
            if k != rep:
                out[k] = Outcome('member', rep=rep)
    return out
