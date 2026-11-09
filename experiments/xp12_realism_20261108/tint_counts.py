"""Tree polygons, vertices and leaf colours per texture for review runs (default: tint off vs on)."""
import collections, json, os, re, sys
RUNS = sys.argv[1:] or ["veg-forest-tint-off", "veg-forest-tint"]
for key in ("szx_e", "khh_n"):
    for run in RUNS:
        f = rf"G:\Ortho4XP_Review\runs\{run}\BI\{key}.js"
        if not os.path.exists(f):
            print(key, run, "missing"); continue
        s = open(f, encoding="utf-8").read()
        d = json.loads(s[s.index("= {", s.index("REVIEW_SAMPLE[")) + 2:].rstrip().rstrip(";"))
        veg = d.get("veg", [])
        nv = sum(sum(len(r) for r in v["r"]) for v in veg if not v.get("pt"))
        cols = collections.Counter(m.group(1) for v in veg for m in [re.search(r"(?:^|[/-])t(\d+)/", v["a"])] if m)
        pts = sum(v.get("pt") or 0 for v in veg)
        print(f"{key:6s} {run:22s} polygons {sum(1 for v in veg if not v.get('pt')):5d}  vertices {nv:7d}  single trees {pts:5d}  "
              f"colours {len(cols)}: " + ", ".join(f"t{k}:{n}" for k, n in cols.most_common()))
