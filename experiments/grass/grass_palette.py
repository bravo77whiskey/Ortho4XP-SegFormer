"""Grass spike step 2: per-texture colour palette -> polygons; count + colour error."""
import os, sys, json, math, time
import numpy as np, cv2
from PIL import Image

SET = r"E:\Ortho4XP_Review\sets\hotspots\BI"
OUT = r"E:\Ortho4XP_Review\_grass_spike"
RANGELAND = 2

def lab(img_rgb):
    return cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)

def smooth_masked(lab_img, mask, sigma_px):
    m = mask.astype(np.float32)
    w = cv2.GaussianBlur(m, (0, 0), sigma_px)
    out = np.empty_like(lab_img)
    for c in range(3):
        out[..., c] = cv2.GaussianBlur(lab_img[..., c] * m, (0, 0), sigma_px) / np.maximum(w, 1e-6)
    return out

def kmeans_palette(samples, k, seed=0):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.05)
    cv2.setRNGSeed(seed)
    _, _, centres = cv2.kmeans(samples.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centres

def assign(lab_px, centres):
    best = np.full(lab_px.shape[0], np.inf, np.float32); lbl = np.zeros(lab_px.shape[0], np.int32)
    for i, c in enumerate(centres):
        d = ((lab_px - c) ** 2).sum(1)
        better = d < best; best[better] = d[better]; lbl[better] = i
    return lbl

def mode_filter(labels, mask, k, win):
    best_n = np.full(labels.shape, -1, np.float32); out = labels.copy()
    for i in range(k):
        n = cv2.boxFilter(((labels == i) & mask).astype(np.float32), -1, (win, win), normalize=False)
        b = n > best_n; out[b] = i; best_n[b] = n[b]
    out[~mask] = -1
    return out

def polygons(labels, k, min_px, eps_px):
    n_out = n_holes = n_verts = 0; lost = 0
    for i in range(k):
        m = (labels == i).astype(np.uint8)
        if not m.any(): continue
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None: continue
        for j, c in enumerate(cnts):
            outer = hier[0][j][3] < 0
            a = cv2.contourArea(c)
            if a < min_px:
                if outer: lost += a
                continue
            s = cv2.approxPolyDP(c, eps_px, True)
            if len(s) < 3: continue
            n_verts += len(s)
            if outer: n_out += 1
            else: n_holes += 1
    return n_out, n_holes, n_verts, lost

def main(fname, bbox, ks=(16, 32, 64), sigma_m=float(os.environ.get("SIGMA_M", 6)), win_m=float(os.environ.get("WIN_M", 11)), min_m2=float(os.environ.get("MIN_M2", 50)), eps_px=float(os.environ.get("EPS_PX", 1))):
    lat_s, lon_w, lat_n, lon_e = bbox
    img = np.asarray(Image.open(os.path.join(SET, fname)).convert("RGB"))
    cls = np.load(os.path.join(OUT, "classmaps", fname[:-4] + "_cls.npy"))
    h, w = cls.shape
    mpp = (lon_e - lon_w) * 111320 * math.cos(math.radians((lat_n + lat_s) / 2)) / w
    grass = cls == RANGELAND
    grass = cv2.morphologyEx(grass.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    L = lab(img)
    S = smooth_masked(L, grass, sigma_m / mpp)
    px = S[grass]
    raw = L[grass]
    rng = np.random.default_rng(0)
    samp = px[rng.choice(len(px), min(200000, len(px)), replace=False)]
    res = {"file": fname, "m_per_px": round(mpp, 3), "grass_frac": round(float(grass.mean()), 3),
           "grass_km2": round(float(grass.sum()) * mpp * mpp / 1e6, 2),
           "detail_lost_dE_median": round(float(np.median(np.linalg.norm(raw - px, axis=1))), 2)}
    # Baseline: one colour per connected grass component.
    ncc, cc = cv2.connectedComponents(grass.astype(np.uint8), connectivity=8)
    ccg = cc[grass]
    sums = np.zeros((ncc, 3)); np.add.at(sums, ccg, px); cnt = np.bincount(ccg, minlength=ncc)[:, None]
    mean = sums / np.maximum(cnt, 1)
    de = np.linalg.norm(px - mean[ccg], axis=1)
    res["one_colour_per_patch"] = {"patches": int(ncc - 1), "dE_median": round(float(np.median(de)), 2),
                                   "dE_p90": round(float(np.percentile(de, 90)), 2)}
    res["palette"] = {}
    min_px = min_m2 / (mpp * mpp); win = max(3, int(round(win_m / mpp)) | 1)
    for k in ks:
        t = time.time()
        centres = kmeans_palette(samp, k)
        lbl = np.full(grass.shape, -1, np.int32); lbl[grass] = assign(px, centres)
        lbl = mode_filter(lbl, grass, k, win)
        de = np.linalg.norm(px - centres[lbl[grass]], axis=1)
        n_out, n_holes, n_verts, lost = polygons(lbl, k, min_px, eps_px)
        res["palette"][k] = {"polygons": n_out, "holes": n_holes, "vertices": n_verts,
                             "area_lost_pct": round(100 * lost / grass.sum(), 2),
                             "dE_median": round(float(np.median(de)), 2), "dE_p90": round(float(np.percentile(de, 90)), 2),
                             "secs": round(time.time() - t, 1)}
        if k == 32:
            np.save(os.path.join(OUT, fname[:-4] + "_pal32_labels.npy"), lbl.astype(np.int8))
            np.save(os.path.join(OUT, fname[:-4] + "_pal32_centres.npy"), centres)
            quant = L.copy(); quant[grass] = centres[lbl[grass]]
            q_rgb = (np.clip(cv2.cvtColor(quant, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)
            # Preview crop around the densest grass 1024 window.
            dens = cv2.boxFilter(grass.astype(np.float32), -1, (1024, 1024))
            y, x = np.unravel_index(np.argmax(dens[512:-512, 512:-512]), (h - 1024, w - 1024))
            crop = lambda a: a[y:y + 1024, x:x + 1024]
            outline = crop(img).copy()
            cnts, _ = cv2.findContours(crop(grass).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outline, cnts, -1, (255, 0, 255), 1)
            Image.fromarray(np.hstack([outline, crop(q_rgb)])).save(os.path.join(OUT, fname[:-4] + "_pal32_preview.jpg"), quality=90)
            res["preview_xy"] = [int(x), int(y)]
    return res

if __name__ == "__main__":
    samples = {s["file"]: s for s in json.load(open(os.path.join(SET, "samples.json")))}
    allres = []
    for f in sys.argv[1:]:
        r = main(f, samples[f]["bbox"]); allres.append(r)
        print(json.dumps(r, indent=1))
    json.dump(allres, open(os.path.join(OUT, "palette_stats_" + os.environ.get("TAG", "a") + ".json"), "w"), indent=1)
