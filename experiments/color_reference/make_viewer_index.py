"""Write <DATA>/color/viewer_index.js listing locations, providers, textures and runs."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refnorm as RN
root = os.path.join(RN.DATA, "color")
# Runs shown in the viewer, newest/best first (first one present is the default).
SHOW = sys.argv[1:] or ["v13gpu", "v14gpufill", "v13", "v14fill", "v2", "v8fill"]
runs = [r for r in SHOW if os.path.isdir(os.path.join(root, "runs", r))]
locs = []
for k in sorted(os.listdir(RN.SET)):
    if not os.path.isfile(os.path.join(RN.SET, k, "meta.json")): continue
    meta = json.load(open(os.path.join(RN.SET, k, "meta.json")))
    provs = {}
    for pv in meta.get("providers", ["BI", "GO2"]):
        b = RN.Block(k, pv)
        provs[pv] = [{"i": i, "j": j, "raw": os.path.relpath(p, root).replace("\\", "/"),
                      "name": os.path.basename(p)} for (i, j), p in sorted(b.paths.items())]
    have = [r for r in runs if os.path.isdir(os.path.join(root, "runs", r, k))]
    have.reverse()   # the viewer lists runs reversed
    locs.append({"key": k, "label": meta["label"], "n": meta.get("n", 1), "providers": provs,
                 "runs": have, "ref": f"set/{k}/ref.png",
                 "refMargin": meta["ref_origin_px"][0] * meta["ref_scale"], "refScale": meta["ref_scale"]})
open(os.path.join(root, "viewer_index.js"), "w").write("window.COLOR_INDEX = " + json.dumps(locs, indent=1) + ";\n")
print(len(locs), "locations,", runs)
