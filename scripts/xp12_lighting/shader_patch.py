"""Patch X-Plane 12's tonemap shaders (SPIR-V inside Resources/shaders/bin/spv/*.xsa).

  status         which archives differ from the pristine backup
  test           mechanism test: clamp the final red channel to 0.5 in every
                 tonemap variant of hdr.xsa -> unmistakable cyan
                 cast in bright areas if X-Plane runs the edited shaders
  curve          replace XP's tone curve t = y/(y+c) with the contrast form
                 t = (gA)^n / ((gA)^n + m^(n-1) c): same mid-grey, darker toe,
                 brighter shoulder (--contrast n, 1.0 = XP's own curve), plus
                 --exposure (stops, g = 2^stops) and --saturation (scales XP's
                 own post-curve saturation mix)
  sky            sky only (dome.xsa): multiply the sky dome's radiance by --gain
                 and scale its saturation by --saturation, before tone mapping
  haze           haze over geometry only (scatter_render_atmosphere.xsa): scale
                 the aerial-perspective output by --amount (1 = stock, 0 = none);
                 sky pixels are discarded by that pass, so the sky is untouched
  meter          exposure metering (lumi_histo.xsa): force XP's own centre-weighted
                 histogram mode with the centre at the screen centre (+ --offset-y,
                 fraction of screen height, positive = lower) and --radius
                 (fraction of screen height); pixels inside count ceil(1/radius)x
  adapt          eye-adaptation speed (lumi_histo.xsa variant 2): XP blends the
                 metered brightness per frame as mix(prev, new, rate); patched to
                 rate' = 1 - (1 - rate)^k, i.e. k x faster at any frame rate
                 (--darker k when the scene gets darker, --brighter k when brighter)
  fusion         local tone mapping strength (exposure_fusion.xsa weight pass):
                 XP weights its 4 synthetic exposures per pixel by
                 exp(-0.5 (luma - 0.5)^2 sigma2); --strength multiplies sigma2
                 (1 = stock; >1 = darker areas lifted / bright areas held more)
  profile        apply the kept set from profile.json (re-run after an X-Plane update
                 or a Steam file verification: those put Laminar's shaders back)
  restore        copy the pristine archives back (all, or hdr / dome / haze / meter /
                 adapt / fusion)

Before any write every archive is copied to the backup directory; the first
copy of each is kept as shaders.original/<name>.  patch_state.json records the
md5 of every archive this tool wrote, so a file that matches neither the
pristine copy nor a recorded patch is recognised as new stock from an X-Plane
update: the old pristine copy is archived and the new file becomes the original.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

XP_ROOT = Path(r"D:\SteamLibrary\steamapps\common\X-Plane 12")
DATA_DIR = Path(r"E:\Ortho4XP_Review\xp12_lighting")
SPV_DIR = XP_ROOT / "Resources" / "shaders" / "bin" / "spv"
# hdr.xsa = final tone map (resolve.xsa only weights MSAA samples; its output stays HDR)
# dome.xsa = sky background drawn from XP's precomputed sky-view table
# scatter_render_atmosphere.xsa = aerial perspective blended over geometry
# lumi_histo.xsa = luminance histogram (variant 1) and eye adaptation (variant 2)
# exposure_fusion.xsa = XP 12.2+ local tone mapping (4 synthetic exposures, Laplacian blend)
ARCHIVES = ("hdr.xsa", "dome.xsa", "scatter_render_atmosphere.xsa", "lumi_histo.xsa", "exposure_fusion.xsa")
RESTORE_NAMES = {"hdr": "hdr.xsa", "dome": "dome.xsa", "haze": "scatter_render_atmosphere.xsa",
                 "meter": "lumi_histo.xsa", "adapt": "lumi_histo.xsa", "fusion": "exposure_fusion.xsa"}
FUSION_MARKER = b"u_sigma2"
METER_MARKER = b"u_spot_center"
ADAPT_MARKER = b"u_blend_speed"
TONEMAP_MARKER = b"u_asc_cdl_slope"
HAZE_MARKER = b"u_gbuffer_data"          # every aerial-perspective fragment variant
SKY_MARKER = b"u_dome_data"
PROFILE = Path(__file__).resolve().parent / "profile.json"            # every dome fragment variant (table-driven and live)

OP_EXT_INST_IMPORT, OP_EXT_INST = 11, 12
OP_TYPE_FLOAT, OP_TYPE_VECTOR = 22, 23
OP_CONSTANT, OP_CONSTANT_COMPOSITE = 43, 44
OP_FUNCTION = 54
OP_NAME, OP_STORE, OP_COMPOSITE_EXTRACT, OP_VECTOR_TIMES_SCALAR, OP_DOT = 5, 62, 81, 142, 148
OP_LOAD, OP_ACCESS_CHAIN, OP_CONVERT_U_TO_F, OP_BITCAST, OP_IEQUAL = 61, 65, 112, 124, 170
OP_TYPE_INT, OP_MEMBER_NAME, OP_TYPE_STRUCT = 21, 6, 30
OP_COMPOSITE_CONSTRUCT, OP_FADD, OP_FSUB, OP_FMUL, OP_FDIV = 80, 129, 131, 133, 136
OP_MATRIX_TIMES_VECTOR = 145
GLSL_FCLAMP, GLSL_FMAX, GLSL_POW, GLSL_FMIX = 43, 40, 26, 46


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _orig_dir():
    return DATA_DIR / "backups" / "shaders.original"


def backup(reason):
    orig = _orig_dir()
    orig.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = DATA_DIR / "backups" / f"shaders_{stamp}_{reason}"
    n = 2
    while snap.exists():
        snap = DATA_DIR / "backups" / f"shaders_{stamp}_{reason}_{n}"
        n += 1
    snap.mkdir(parents=True)
    for name in ARCHIVES:
        src = SPV_DIR / name
        if not (orig / name).exists():
            shutil.copy2(src, orig / name)
        shutil.copy2(src, snap / name)
        if _md5(snap / name) != _md5(src):
            sys.exit(f"backup verification failed for {snap / name}")
    print(f"backup: {snap}")


def _instructions(words):
    i = 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        if wc == 0:
            raise ValueError("malformed SPIR-V (zero word count)")
        yield i, op, wc
        i += wc


def clamp_red(spv, red_max):
    """Replace the max operand of every vec3 FClamp(x, 0, 1) with (red_max, 1, 1)."""
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    floats, consts, composites, glsl, first_fn = set(), {}, {}, None, None
    clamps = []
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_EXT_INST_IMPORT and struct.pack(f"<{wc - 2}I", *a[1:]).startswith(b"GLSL.std.450"):
            glsl = a[0]
        elif op == OP_TYPE_FLOAT and a[1] == 32:
            floats.add(a[0])
        elif op == OP_CONSTANT and a[0] in floats:
            consts[a[1]] = struct.unpack("<f", struct.pack("<I", a[2]))[0]
        elif op == OP_CONSTANT_COMPOSITE:
            composites[a[1]] = (a[0], a[2:])
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        elif op == OP_EXT_INST and a[2] == glsl and a[3] == GLSL_FCLAMP:
            clamps.append(i)
    patched = 0
    bound = words[3]
    float_type = next(iter(floats))
    new_red, new_vec = bound, bound + 1
    vec_type = None
    for i in clamps:
        lo, hi = words[i + 6], words[i + 7]
        if lo in composites and hi in composites:
            (t_lo, c_lo), (t_hi, c_hi) = composites[lo], composites[hi]
            if (len(c_hi) == 3 and all(consts.get(c) == 1.0 for c in c_hi)
                    and all(consts.get(c) == 0.0 for c in c_lo)):
                vec_type = t_hi
                words[i + 7] = new_vec
                patched += 1
    if not patched:
        return None, 0
    one = next(c for c, v in consts.items() if v == 1.0)
    decl = [(4 << 16) | OP_CONSTANT, float_type, new_red, struct.unpack("<I", struct.pack("<f", red_max))[0],
            (6 << 16) | OP_CONSTANT_COMPOSITE, vec_type, new_vec, new_red, one, one]
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = bound + 2
    return struct.pack(f"<{len(words)}I", *words), patched


def _f32(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def contrast_curve(spv, n, exposure=0.0, saturation=1.0):
    """Rewrite XP's tone curve in place.

    XP (all hdr tonemap variants):  A = WCS * cdl(x)          (MatrixTimesVector)
                                    k = D * (1 - mid)
                                    t = A / (A + vec3(k))
    patched:                        An = pow(max(A, 0), n)
                                    t  = An / (An + vec3(mid^(n-1) * k))
    The Hill/Naka-Rushton form keeps mid-grey where it was (A = mid gives the
    same t) while n > 1 adds a toe and pulls the shoulder up toward white.
    Exposure g = 2^stops folds into the constant: (gA)^n/((gA)^n+s) =
    A^n/(A^n + s/g^n).  Saturation scales the factor of XP's own
    mix(luma, colour, sat) that follows the outset matrix.
    """
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    floats, consts, glsl, first_fn = set(), {}, None, None
    defs = {}                                   # result id -> (word index, opcode)
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_EXT_INST_IMPORT and struct.pack(f"<{wc - 2}I", *a[1:]).startswith(b"GLSL.std.450"):
            glsl = a[0]
        elif op == OP_TYPE_FLOAT and a[1] == 32:
            floats.add(a[0])
        elif op == OP_CONSTANT and a[0] in floats:
            consts[a[1]] = struct.unpack("<f", struct.pack("<I", a[2]))[0]
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        if op in (OP_COMPOSITE_CONSTRUCT, OP_FADD, OP_FSUB, OP_FMUL, OP_FDIV, OP_MATRIX_TIMES_VECTOR):
            defs[a[1]] = (i, op)
        elif op == OP_EXT_INST and a[3] == GLSL_FMIX:
            defs[a[1]] = (i, ("fmix", a[4:]))
    # locate t = A / (A + C), C = vec3(k), k = D * (1 - mid), A = matrix * vector
    site = None
    for res, (i, op) in defs.items():
        if op != OP_FDIV:
            continue
        num, den = words[i + 3], words[i + 4]
        if defs.get(den, (0, 0))[1] != OP_FADD or defs.get(num, (0, 0))[1] != OP_MATRIX_TIMES_VECTOR:
            continue
        ia = defs[den][0]
        if words[ia + 3] != num:
            continue
        c = words[ia + 4]
        if defs.get(c, (0, 0))[1] != OP_COMPOSITE_CONSTRUCT:
            continue
        ic = defs[c][0]
        k = words[ic + 3]
        if words[ic + 3:ic + 6] != [k, k, k] or defs.get(k, (0, 0))[1] != OP_FMUL:
            continue
        ik = defs[k][0]
        sub = next((x for x in words[ik + 3:ik + 5] if defs.get(x, (0, 0))[1] == OP_FSUB), None)
        if sub is None or consts.get(words[defs[sub][0] + 3]) != 1.0:
            continue
        mid = words[defs[sub][0] + 4]
        if defs[num][0] > ic:
            continue                            # A must exist before vec3(k) is built
        site = dict(div=i, add=ia, construct=ic, num=num, k=k, mid=mid,
                    vec3=words[ic + 1], f32=words[ik + 1])
        break
    if site is None:
        return None, 0
    # saturation site: mix(luma, outset * t, vec3(sat)) right after the curve
    t_id = words[site["div"] + 2]
    sat = None
    if saturation != 1.0:
        outset = next((r for r, (i, op) in defs.items() if op == OP_MATRIX_TIMES_VECTOR and words[i + 4] == t_id), None)
        for r, (i, op) in defs.items():
            if isinstance(op, tuple) and op[1][1] == outset and defs.get(op[1][2], (0, 0))[1] == OP_COMPOSITE_CONSTRUCT:
                ic2 = defs[op[1][2]][0]
                v = words[ic2 + 3]
                if words[ic2 + 3:ic2 + 6] == [v, v, v]:
                    sat = dict(construct=ic2, value=v)
                    break
        if sat is None:
            raise ValueError("saturation mix not found")
    b = words[3]
    ids = iter(range(b, b + 16))
    id_n, id_nm1, id_zero, id_nvec, id_zvec, id_amax, id_an, id_mp, id_s0, id_gain, id_s = (next(ids) for _ in range(11))
    f32, vec3 = site["f32"], site["vec3"]
    gain_n = 2.0 ** (-exposure * n)              # s / g^n
    decl = [(4 << 16) | OP_CONSTANT, f32, id_n, _f32(n),
            (4 << 16) | OP_CONSTANT, f32, id_nm1, _f32(n - 1.0),
            (4 << 16) | OP_CONSTANT, f32, id_zero, _f32(0.0),
            (4 << 16) | OP_CONSTANT, f32, id_gain, _f32(gain_n),
            (6 << 16) | OP_CONSTANT_COMPOSITE, vec3, id_nvec, id_n, id_n, id_n,
            (6 << 16) | OP_CONSTANT_COMPOSITE, vec3, id_zvec, id_zero, id_zero, id_zero]
    code = [(7 << 16) | OP_EXT_INST, vec3, id_amax, glsl, GLSL_FMAX, site["num"], id_zvec,
            (7 << 16) | OP_EXT_INST, vec3, id_an, glsl, GLSL_POW, id_amax, id_nvec,
            (7 << 16) | OP_EXT_INST, f32, id_mp, glsl, GLSL_POW, site["mid"], id_nm1,
            (5 << 16) | OP_FMUL, f32, id_s0, id_mp, site["k"],
            (5 << 16) | OP_FMUL, f32, id_s, id_s0, id_gain]
    ic, ia, idv = site["construct"], site["add"], site["div"]
    edits = []                                   # (word index, new instructions) inserted later
    words[ic + 3:ic + 6] = [id_s, id_s, id_s]   # C = vec3(s)
    words[ia + 3] = id_an                        # An + C
    words[idv + 3] = id_an                       # An / (An + C)
    edits.append((ic, code))                     # code goes right before C is built
    used = 11
    if sat is not None:
        id_satk, id_sat = b + used, b + used + 1
        used += 2
        decl += [(4 << 16) | OP_CONSTANT, f32, id_satk, _f32(saturation)]
        ic2 = sat["construct"]
        words[ic2 + 3:ic2 + 6] = [id_sat, id_sat, id_sat]
        edits.append((ic2, [(5 << 16) | OP_FMUL, f32, id_sat, sat["value"], id_satk]))
    for at, ins in sorted(edits, reverse=True):  # back to front keeps indices valid
        words = words[:at] + ins + words[at:]
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = b + used
    return struct.pack(f"<{len(words)}I", *words), 1


def sky_boost(spv, gain, saturation):
    """Scale the sky dome's output radiance (before tone mapping).

    XP (every dome variant):  V = sky_sample * scale            (vec3)
                              out = clamp(vec4(V.x, V.y, V.z, 0), -65504, 65504)
    patched:                  L = dot(V, rec709)
                              V' = max((L + (V - L) * saturation) * gain, 0)
    """
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    floats, glsl, first_fn, out_var = set(), None, None, None
    defs = {}
    stores = []
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_NAME and struct.pack(f"<{wc - 2}I", *a[1:]).split(b"\0")[0] == b"data_out0":
            out_var = a[0]
        elif op == OP_EXT_INST_IMPORT and struct.pack(f"<{wc - 2}I", *a[1:]).startswith(b"GLSL.std.450"):
            glsl = a[0]
        elif op == OP_TYPE_FLOAT and a[1] == 32:
            floats.add(a[0])
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        elif op == OP_STORE:
            stores.append(i)
        if op in (OP_COMPOSITE_CONSTRUCT, OP_COMPOSITE_EXTRACT, OP_EXT_INST, OP_VECTOR_TIMES_SCALAR, OP_FMUL):
            defs[a[1]] = (i, op)
    for st in stores:
        if words[st + 1] != out_var or defs.get(words[st + 2], (0, 0))[1] != OP_EXT_INST:
            continue
        ic = defs[words[st + 2]][0]                      # FClamp(vec4(...), lo, hi)
        if words[ic + 3] != glsl or words[ic + 4] != GLSL_FCLAMP:
            continue
        cc = words[ic + 5]
        if defs.get(cc, (0, 0))[1] != OP_COMPOSITE_CONSTRUCT:
            continue
        icc = defs[cc][0]
        comps = words[icc + 3:icc + 6]
        ex = [defs.get(x) for x in comps]
        if not all(e and e[1] == OP_COMPOSITE_EXTRACT for e in ex):
            continue
        srcs = {words[e[0] + 3] for e in ex}
        lanes = [words[e[0] + 4] for e in ex]
        if len(srcs) != 1 or lanes != [0, 1, 2]:
            continue
        v = srcs.pop()
        iv = defs[v][0]
        vec3, f32 = words[iv + 1], next(iter(floats))
        break
    else:
        return None, 0
    b = words[3]
    (id_lr, id_lg, id_lb, id_luma, id_sat, id_gain, id_zero, id_zvec,
     id_l, id_lv, id_d, id_ds, id_m, id_g, id_out) = range(b, b + 15)
    decl = [(4 << 16) | OP_CONSTANT, f32, id_lr, _f32(0.2126),
            (4 << 16) | OP_CONSTANT, f32, id_lg, _f32(0.7152),
            (4 << 16) | OP_CONSTANT, f32, id_lb, _f32(0.0722),
            (4 << 16) | OP_CONSTANT, f32, id_sat, _f32(saturation),
            (4 << 16) | OP_CONSTANT, f32, id_gain, _f32(gain),
            (4 << 16) | OP_CONSTANT, f32, id_zero, _f32(0.0),
            (6 << 16) | OP_CONSTANT_COMPOSITE, vec3, id_luma, id_lr, id_lg, id_lb,
            (6 << 16) | OP_CONSTANT_COMPOSITE, vec3, id_zvec, id_zero, id_zero, id_zero]
    code = [(5 << 16) | OP_DOT, f32, id_l, v, id_luma,
            (6 << 16) | OP_COMPOSITE_CONSTRUCT, vec3, id_lv, id_l, id_l, id_l,
            (5 << 16) | OP_FSUB, vec3, id_d, v, id_lv,
            (5 << 16) | OP_VECTOR_TIMES_SCALAR, vec3, id_ds, id_d, id_sat,
            (5 << 16) | OP_FADD, vec3, id_m, id_lv, id_ds,
            (5 << 16) | OP_VECTOR_TIMES_SCALAR, vec3, id_g, id_m, id_gain,
            (7 << 16) | OP_EXT_INST, vec3, id_out, glsl, GLSL_FMAX, id_g, id_zvec]
    for e in ex:
        words[e[0] + 3] = id_out                  # extracts now read the boosted sky
    at = iv + (words[iv] >> 16)                   # right after V is defined
    words = words[:at] + code + words[at:]
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = b + 15
    return struct.pack(f"<{len(words)}I", *words), 1


def haze_scale(spv, amount):
    """Scale the aerial-perspective pass output.

    XP (every variant):  out = vec4(inscatter, 1 - luma(transmittance))
                         blended as scene * (1 - out.a) + out.rgb
    patched:             out = out * amount
    Equivalent to XP's own fade factor in the table-driven variants: both the
    added in-scatter and the dimming of the scene shrink together.
    """
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    floats, first_fn, out_var = set(), None, None
    defs, stores = {}, []
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_NAME and struct.pack(f"<{wc - 2}I", *a[1:]).split(b"\0")[0] == b"data_out0":
            out_var = a[0]
        elif op == OP_TYPE_FLOAT and a[1] == 32:
            floats.add(a[0])
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        elif op == OP_STORE:
            stores.append(i)
        if op == OP_COMPOSITE_CONSTRUCT:
            defs[a[1]] = (i, op)
    sites = [st for st in stores if words[st + 1] == out_var and words[st + 2] in defs]
    if not sites:
        return None, 0
    b = words[3]
    id_amount = b
    vec4 = words[defs[words[sites[0] + 2]][0] + 1]
    f32 = next(iter(floats))
    decl = [(4 << 16) | OP_CONSTANT, f32, id_amount, _f32(amount)]
    nxt = b + 1
    for st in sorted(sites, reverse=True):          # back to front keeps indices valid
        value = words[st + 2]
        words[st + 2] = nxt
        words = words[:st] + [(5 << 16) | OP_VECTOR_TIMES_SCALAR, vec4, nxt, value, id_amount] + words[st:]
        nxt += 1
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = nxt
    return struct.pack(f"<{len(words)}I", *words), len(sites)


def _replace_uses(words, mapping, start):
    """Replace operand references (not result/type ids) from word index `start` on."""
    for i, op, wc in _instructions(words):
        if i < start:
            continue
        a0 = 3 if op in _TYPED_RESULT_OPS else 1
        if op == OP_EXT_INST:
            a0 = 5                                   # skip set id + instruction literal
        for k in range(i + a0, i + wc):
            if words[k] in mapping:
                words[k] = mapping[words[k]]


_TYPED_RESULT_OPS = {OP_EXT_INST, OP_LOAD, OP_ACCESS_CHAIN, OP_COMPOSITE_CONSTRUCT, OP_COMPOSITE_EXTRACT,
                     OP_CONVERT_U_TO_F, OP_BITCAST, OP_IEQUAL, OP_FADD, OP_FSUB, OP_FMUL, OP_FDIV,
                     OP_VECTOR_TIMES_SCALAR, OP_DOT, OP_MATRIX_TIMES_VECTOR,
                     79, 86, 87, 88, 95, 98, 100, 103, 104, 109, 110, 111, 126, 127, 128, 130, 132, 134,
                     135, 137, 139, 141, 143, 144, 146, 154, 155, 156, 157, 164, 165, 166, 167, 168, 169,
                     171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 184, 186, 188, 190, 194, 195,
                     196, 197, 198, 199, 200, 245, 234}


def centre_weighted_meter(spv, radius, offset_y):
    """Force XP's centre-weighted histogram mode (lumi_histo variant 1).

    XP:       mode, centre, radius come from u_histo_data (members 5, 6, 7);
              p = gid.xy / min(w, h); inside = |p - centre| <= radius;
              mode 1 counts every pixel, inside ones ceil(1/radius) times.
    patched:  mode = 1, radius = constant,
              centre = (w, h) / (2 min(w, h)) + (0, offset_y)   (screen centre)
    """
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    names, member_names, floats, ints = {}, {}, set(), {}
    consts, first_fn, histo_struct = {}, None, None
    loads, chains, defs = {}, {}, {}
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_NAME:
            names[a[0]] = struct.pack(f"<{wc - 2}I", *a[1:]).split(b"\0")[0]
        elif op == OP_MEMBER_NAME:
            member_names[(a[0], a[1])] = struct.pack(f"<{wc - 3}I", *a[2:]).split(b"\0")[0]
        elif op == OP_TYPE_FLOAT and a[1] == 32:
            floats.add(a[0])
        elif op == OP_TYPE_INT:
            ints[a[0]] = a[2]
        elif op == OP_CONSTANT:
            consts[a[1]] = (a[0], a[2])
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        elif op == OP_ACCESS_CHAIN and wc == 5:
            chains[a[1]] = a[3]                       # pointer id -> member index constant
        elif op == OP_LOAD:
            loads[a[1]] = (i, a[2], a[0])            # result -> (index, pointer, type)
        if op in _TYPED_RESULT_OPS:
            defs[a[1]] = (i, op, a[0])
    member_of = {}
    for (st, idx), nm in member_names.items():
        member_of[nm] = idx
    if not {b"u_histo_mode", b"u_spot_center", b"u_spot_radius"} <= set(member_of):
        return None, 0
    want = {member_of[b"u_histo_mode"]: "mode", member_of[b"u_spot_center"]: "centre",
            member_of[b"u_spot_radius"]: "radius"}
    found = {}
    for res, (i, ptr, typ) in loads.items():
        idx_const = chains.get(ptr)
        if idx_const in consts and consts[idx_const][1] in want:
            found[want[consts[idx_const][1]]] = (res, i, typ)
    if set(found) != {"mode", "centre", "radius"}:
        return None, 0
    mode_id, _, mode_type = found["mode"]
    centre_id, _, vec2 = found["centre"]
    radius_id, _, f32 = found["radius"]
    # the image size: two Bitcasts of loaded u_image_width / u_image_height
    w_member, h_member = member_of[b"u_image_width"], member_of[b"u_image_height"]
    size = {}
    for res, (i, op, typ) in defs.items():
        if op == OP_BITCAST:
            src = words[i + 3]
            if src in loads:
                c = chains.get(loads[src][1])
                if c in consts and consts[c][1] in (w_member, h_member):
                    size[consts[c][1]] = (res, i)
    if set(size) != {w_member, h_member}:
        raise ValueError("image size loads not found")
    # first use of the centre: insert the computed centre right before it
    use_at = None
    for i, op, wc in _instructions(words):
        if i > first_fn and centre_id in words[i + 3:i + wc] and op in _TYPED_RESULT_OPS:
            use_at = i
            break
    # min(w, h) as float: the FDiv denominator next to the centre use is vec2(minf, minf)
    minf = None
    for res, (i, op, typ) in defs.items():
        if op == OP_COMPOSITE_CONSTRUCT and typ == vec2 and words[i + 3] == words[i + 4] and i < use_at:
            minf = words[i + 3]
    if minf is None or use_at is None:
        raise ValueError("centre computation site not found")
    b = words[3]
    (id_one, id_radius, id_two, id_offy, id_wf, id_hf, id_m2, id_cx, id_cy0, id_cy, id_c) = range(b, b + 11)
    decl = [(4 << 16) | OP_CONSTANT, mode_type, id_one, 1,
            (4 << 16) | OP_CONSTANT, f32, id_radius, _f32(radius),
            (4 << 16) | OP_CONSTANT, f32, id_two, _f32(2.0),
            (4 << 16) | OP_CONSTANT, f32, id_offy, _f32(offset_y)]
    code = [(4 << 16) | OP_CONVERT_U_TO_F, f32, id_wf, size[w_member][0],
            (4 << 16) | OP_CONVERT_U_TO_F, f32, id_hf, size[h_member][0],
            (5 << 16) | OP_FMUL, f32, id_m2, minf, id_two,
            (5 << 16) | OP_FDIV, f32, id_cx, id_wf, id_m2,
            (5 << 16) | OP_FDIV, f32, id_cy0, id_hf, id_m2,
            (5 << 16) | OP_FADD, f32, id_cy, id_cy0, id_offy,
            (5 << 16) | OP_COMPOSITE_CONSTRUCT, vec2, id_c, id_cx, id_cy]
    _replace_uses(words, {mode_id: id_one, centre_id: id_c, radius_id: id_radius}, first_fn)
    words = words[:use_at] + code + words[use_at:]
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = b + 11
    return struct.pack(f"<{len(words)}I", *words), 1


def adaptation_speed(spv, k_darker, k_brighter):
    """Speed up XP's eye adaptation (lumi_histo variant 2).

    XP:       adapted = mix(prev, new, u_blend_speed[0])  if prev > new (scene darker)
              adapted = mix(prev, new, u_blend_speed[1])  otherwise    (scene brighter)
    patched:  rate' = 1 - clamp(1 - rate, 0, 1)^k
    """
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    member_of, consts, glsl, first_fn = {}, {}, None, None
    chains, loads, fmix = {}, {}, []
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_MEMBER_NAME:
            member_of[struct.pack(f"<{wc - 3}I", *a[2:]).split(b"\0")[0]] = a[1]
        elif op == OP_EXT_INST_IMPORT and struct.pack(f"<{wc - 2}I", *a[1:]).startswith(b"GLSL.std.450"):
            glsl = a[0]
        elif op == OP_CONSTANT:
            consts[a[1]] = (a[0], a[2])
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        elif op == OP_ACCESS_CHAIN and wc == 6:
            chains[a[1]] = (a[3], a[4])               # pointer -> (member const, component const)
        elif op == OP_LOAD:
            loads[a[1]] = (a[2], a[0])                # result -> (pointer, type)
        elif op == OP_EXT_INST and a[2] == glsl and a[3] == GLSL_FMIX:
            fmix.append(i)
    if b"u_blend_speed" not in member_of:
        return None, 0
    speed_member = member_of[b"u_blend_speed"]
    sites = []
    for i in fmix:
        rate = words[i + 7]
        if rate not in loads or loads[rate][0] not in chains:
            continue
        m, c = chains[loads[rate][0]]
        if m in consts and consts[m][1] == speed_member and c in consts:
            sites.append((i, rate, loads[rate][1], consts[c][1]))
    if len(sites) != 2 or {s_[3] for s_ in sites} != {0, 1}:
        return None, 0
    f32 = sites[0][2]
    b = words[3]
    id_one, id_zero, id_kd, id_kb = b, b + 1, b + 2, b + 3
    nxt = b + 4
    decl = [(4 << 16) | OP_CONSTANT, f32, id_one, _f32(1.0),
            (4 << 16) | OP_CONSTANT, f32, id_zero, _f32(0.0),
            (4 << 16) | OP_CONSTANT, f32, id_kd, _f32(k_darker),
            (4 << 16) | OP_CONSTANT, f32, id_kb, _f32(k_brighter)]
    for i, rate, _t, comp in sorted(sites, reverse=True):     # back to front keeps indices valid
        k = id_kd if comp == 0 else id_kb
        om, cl, pw, nr = nxt, nxt + 1, nxt + 2, nxt + 3
        nxt += 4
        code = [(5 << 16) | OP_FSUB, f32, om, id_one, rate,
                (8 << 16) | OP_EXT_INST, f32, cl, glsl, GLSL_FCLAMP, om, id_zero, id_one,
                (7 << 16) | OP_EXT_INST, f32, pw, glsl, GLSL_POW, cl, k,
                (5 << 16) | OP_FSUB, f32, nr, id_one, pw]
        words[i + 7] = nr
        words = words[:i] + code + words[i:]
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = nxt
    return struct.pack(f"<{len(words)}I", *words), len(sites)


def fusion_strength(spv, strength):
    """Scale exposure-fusion sigma2 in the weight pass: w = exp(-0.5 d^2 * sigma2 * strength)."""
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    if words[0] != 0x07230203:
        raise ValueError("not SPIR-V")
    member_of, consts, first_fn = {}, {}, None
    chains, loads, vts = {}, {}, []
    for i, op, wc in _instructions(words):
        a = words[i + 1:i + wc]
        if op == OP_MEMBER_NAME:
            member_of[struct.pack(f"<{wc - 3}I", *a[2:]).split(b"\0")[0]] = a[1]
        elif op == OP_CONSTANT:
            consts[a[1]] = (a[0], a[2])
        elif op == OP_FUNCTION and first_fn is None:
            first_fn = i
        elif op == OP_ACCESS_CHAIN and wc == 5:
            chains[a[1]] = a[3]
        elif op == OP_LOAD:
            loads[a[1]] = (a[2], a[0])
        elif op == OP_VECTOR_TIMES_SCALAR:
            vts.append(i)
    if b"u_sigma2" not in member_of:
        return None, 0
    sig = member_of[b"u_sigma2"]
    sites = []
    for i in vts:
        sc = words[i + 4]
        if sc in loads and chains.get(loads[sc][0]) in consts and consts[chains[loads[sc][0]]][1] == sig:
            sites.append((i, sc, loads[sc][1]))
    if not sites:
        return None, 0
    f32 = sites[0][2]
    b = words[3]
    id_k, nxt = b, b + 1
    decl = [(4 << 16) | OP_CONSTANT, f32, id_k, _f32(strength)]
    for i, sc, _t in sorted(sites, reverse=True):
        words[i + 4] = nxt
        words = words[:i] + [(5 << 16) | OP_FMUL, f32, nxt, sc, id_k] + words[i:]
        nxt += 1
    words = words[:first_fn] + decl + words[first_fn:]
    words[3] = nxt
    return struct.pack(f"<{len(words)}I", *words), len(sites)


def _build_archive(src, transform, verbose=True):
    """Return (archive bytes, sites patched) for `src` with `transform` applied to each entry."""
    with zipfile.ZipFile(src) as z:
        entries = [(info, z.read(info)) for info in z.infolist()]
    buf = io.BytesIO()
    total = 0
    with zipfile.ZipFile(buf, "w") as out:
        for info, data in entries:
            new = transform(info.filename, data)
            if new is not None:
                data, n = new
                total += n
                if verbose:
                    print(f"  {Path(src).name}/{info.filename}: {n} site(s) patched")
            out.writestr(info, data, compress_type=info.compress_type)
    return buf.getvalue(), total


def _state_path():
    return DATA_DIR / "patch_state.json"


def _load_state():
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_state(state):
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=1))


def _classify(name):
    """'stock', 'ours' (a patch this tool wrote) or 'foreign' (changed by someone else, e.g. an update)."""
    orig, cur = _orig_dir() / name, SPV_DIR / name
    if not orig.exists() or _md5(orig) == _md5(cur):
        return "stock"
    rec = _load_state().get(name)
    return "ours" if rec and rec["md5"] == _md5(cur) else "foreign"


def _adopt_new_stock(name):
    """The live file is Laminar's new stock: archive the old pristine copy, make this the original."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    old = DATA_DIR / "backups" / f"shaders.original.replaced_{stamp}"
    old.mkdir(parents=True, exist_ok=True)
    shutil.move(str(_orig_dir() / name), old / name)
    shutil.copy2(SPV_DIR / name, _orig_dir() / name)
    state = _load_state()
    state.pop(name, None)
    _save_state(state)
    print(f"{name}: changed by X-Plane (update or file verification) - adopted as the new original; "
          f"previous original kept in {old}")


def xplane_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq X-Plane.exe"], capture_output=True, text=True).stdout
    return "X-Plane.exe" in out


def cmd_status(_args):
    state = _load_state()
    for name in ARCHIVES:
        kind = _classify(name)
        if kind == "stock":
            print(f"{name}: ORIGINAL")
        elif kind == "ours":
            print(f"{name}: PATCHED ({state[name]['patch']})")
        else:
            print(f"{name}: CHANGED BY X-PLANE? (matches neither the original nor a recorded patch)")


def _marker_transform(marker, transform_one):
    def transform(_fname, data):
        if marker not in data:
            return None
        new, count = transform_one(data)
        return (new, count) if new else None
    return transform


def _patch(name, marker, transform_one, label):
    """Patch one archive (all others untouched); refuses if it is already patched."""
    if xplane_running():
        sys.exit("X-Plane is running - quit it first (shader archives are loaded at startup)")
    kind = _classify(name)
    if kind == "ours":
        key = "/".join(k for k, v in RESTORE_NAMES.items() if v == name)
        sys.exit(f"{name} is already patched ({_load_state()[name]['patch']}) - run restore {key} first")
    backup(f"before_{label}")
    if kind == "foreign":
        _adopt_new_stock(name)
    data, total = _build_archive(SPV_DIR / name, _marker_transform(marker, transform_one))
    if total == 0:
        sys.exit(f"{name}: patch '{label}' found nothing to patch - the shader changed; file left untouched")
    (SPV_DIR / name).write_bytes(data)
    state = _load_state()
    state[name] = {"md5": _md5(SPV_DIR / name), "patch": label}
    _save_state(state)
    return total


def cmd_test(args):
    total = _patch("hdr.xsa", TONEMAP_MARKER, lambda d: clamp_red(d, args.red_max), "test")
    print(f"patched {total} clamp(s); red output now capped at {args.red_max}")


def cmd_curve(args):
    label = f"curve_c{args.contrast:g}_e{args.exposure:g}_s{args.saturation:g}"
    total = _patch("hdr.xsa", TONEMAP_MARKER,
                   lambda d: contrast_curve(d, args.contrast, args.exposure, args.saturation), label)
    print(f"patched {total} tone curve(s); contrast {args.contrast:g}, exposure {args.exposure:+g} stops, "
          f"saturation x{args.saturation:g}")


def cmd_sky(args):
    total = _patch("dome.xsa", SKY_MARKER, lambda d: sky_boost(d, args.gain, args.saturation),
                   f"sky_g{args.gain:g}_s{args.saturation:g}")
    print(f"patched {total} sky dome variant(s); gain x{args.gain:g}, saturation x{args.saturation:g}")


def cmd_haze(args):
    total = _patch("scatter_render_atmosphere.xsa", HAZE_MARKER, lambda d: haze_scale(d, args.amount),
                   f"haze_{args.amount:g}")
    print(f"patched {total} aerial-perspective output(s); haze amount x{args.amount:g}")


def cmd_meter(args):
    total = _patch("lumi_histo.xsa", METER_MARKER,
                   lambda d: centre_weighted_meter(d, args.radius, args.offset_y),
                   f"meter_r{args.radius:g}_y{args.offset_y:g}")
    print(f"patched {total} histogram variant(s); centre-weighted, radius {args.radius:g}, "
          f"weight x{math.ceil(1 / args.radius)}, centre offset {args.offset_y:+g}")


def cmd_adapt(args):
    total = _patch("lumi_histo.xsa", ADAPT_MARKER,
                   lambda d: adaptation_speed(d, args.darker, args.brighter),
                   f"adapt_d{args.darker:g}_b{args.brighter:g}")
    print(f"patched {total} adaptation blend(s); {args.darker:g}x faster when darker, "
          f"{args.brighter:g}x faster when brighter")


def cmd_fusion(args):
    total = _patch("exposure_fusion.xsa", FUSION_MARKER, lambda d: fusion_strength(d, args.strength),
                   f"fusion_s{args.strength:g}")
    print(f"patched {total} exposure-fusion weight pass(es); strength x{args.strength:g}")


def cmd_restore(args):
    if xplane_running():
        sys.exit("X-Plane is running - quit it first")
    names = ARCHIVES if args.which == "all" else (RESTORE_NAMES[args.which],)
    orig = _orig_dir()
    kinds = {n: _classify(n) for n in names}
    for n in [n for n, k in kinds.items() if k == "foreign"]:
        print(f"{n}: changed by X-Plane, not by this tool - left as is (it is the new stock)")
    todo = [n for n, k in kinds.items() if k == "ours"]
    if not todo:
        print("nothing to restore - already original")
        return
    backup("before_restore")
    state = _load_state()
    for name in todo:
        shutil.copy2(orig / name, SPV_DIR / name)
        if _md5(orig / name) != _md5(SPV_DIR / name):
            sys.exit(f"restore verification failed for {name}")
        state.pop(name, None)
        print(f"{name} restored to original")
    _save_state(state)


# profile key -> (archive, marker, transform factory, label factory)
_PROFILE_PATCHES = {
    "curve": ("hdr.xsa", TONEMAP_MARKER,
              lambda c: (lambda d: contrast_curve(d, c.get("contrast", 1.0), c.get("exposure", 0.0),
                                                  c.get("saturation", 1.0))),
              lambda c: f"curve_c{c.get('contrast', 1.0):g}_e{c.get('exposure', 0.0):g}_s{c.get('saturation', 1.0):g}"),
    "sky": ("dome.xsa", SKY_MARKER,
            lambda c: (lambda d: sky_boost(d, c.get("gain", 1.0), c.get("saturation", 1.0))),
            lambda c: f"sky_g{c.get('gain', 1.0):g}_s{c.get('saturation', 1.0):g}"),
    "haze": ("scatter_render_atmosphere.xsa", HAZE_MARKER,
             lambda c: (lambda d: haze_scale(d, c["amount"])),
             lambda c: f"haze_{c['amount']:g}"),
    "adapt": ("lumi_histo.xsa", ADAPT_MARKER,
              lambda c: (lambda d: adaptation_speed(d, c.get("darker", 1.0), c.get("brighter", 1.0))),
              lambda c: f"adapt_d{c.get('darker', 1.0):g}_b{c.get('brighter', 1.0):g}"),
    "fusion": ("exposure_fusion.xsa", FUSION_MARKER,
               lambda c: (lambda d: fusion_strength(d, c["strength"])),
               lambda c: f"fusion_s{c['strength']:g}"),
}


def cmd_profile(args):
    """Bring every archive in the profile to its patched state; leave the others alone."""
    if xplane_running():
        sys.exit("X-Plane is running - quit it first (shader archives are loaded at startup)")
    profile = json.loads(Path(args.profile).read_text())
    unknown = set(profile) - set(_PROFILE_PATCHES) - {"_comment"}
    if unknown:
        sys.exit(f"unknown profile entries: {sorted(unknown)}")
    for key, cfg in profile.items():
        if key == "_comment":
            continue
        name, marker, make, make_label = _PROFILE_PATCHES[key]
        label = make_label(cfg)
        kind = _classify(name)
        orig = _orig_dir() / name
        if kind in ("ours", "foreign") and orig.exists():
            # already exactly this patch?  rebuild it from the original and compare
            want, _n = _build_archive(orig, _marker_transform(marker, make(cfg)), verbose=False)
            if hashlib.md5(want).hexdigest() == _md5(SPV_DIR / name):
                state = _load_state()
                state[name] = {"md5": _md5(SPV_DIR / name), "patch": label}
                _save_state(state)
                print(f"{name}: {label} already applied")
                continue
        if kind == "ours":
            cmd_restore(argparse.Namespace(which=next(k for k, v in RESTORE_NAMES.items() if v == name)))
        total = _patch(name, marker, make(cfg), label)
        print(f"{name}: {label} applied ({total} site(s))")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("test")
    p.add_argument("--red-max", type=float, default=0.5)
    p.set_defaults(func=cmd_test)
    p = sub.add_parser("curve")
    p.add_argument("--contrast", type=float, default=1.3)
    p.add_argument("--exposure", type=float, default=0.0, help="stops, applied before the curve")
    p.add_argument("--saturation", type=float, default=1.0, help="multiplies XP's saturation factor")
    p.set_defaults(func=cmd_curve)
    p = sub.add_parser("sky")
    p.add_argument("--gain", type=float, default=1.6, help="sky radiance multiplier (1.6 = +0.68 stop)")
    p.add_argument("--saturation", type=float, default=1.0)
    p.set_defaults(func=cmd_sky)
    p = sub.add_parser("haze")
    p.add_argument("--amount", type=float, default=0.7, help="1 = stock haze, 0 = none")
    p.set_defaults(func=cmd_haze)
    p = sub.add_parser("meter")
    p.add_argument("--radius", type=float, default=0.3, help="fraction of screen height")
    p.add_argument("--offset-y", type=float, default=0.0, help="centre shift, fraction of height, + = lower")
    p.set_defaults(func=cmd_meter)
    p = sub.add_parser("adapt")
    p.add_argument("--darker", type=float, default=2.0, help="speed factor when the scene gets darker")
    p.add_argument("--brighter", type=float, default=2.0, help="speed factor when the scene gets brighter")
    p.set_defaults(func=cmd_adapt)
    p = sub.add_parser("fusion")
    p.add_argument("--strength", type=float, default=1.5, help="multiplies sigma2; 1 = stock")
    p.set_defaults(func=cmd_fusion)
    p = sub.add_parser("profile")
    p.add_argument("profile", nargs="?", default=str(PROFILE))
    p.set_defaults(func=cmd_profile)
    p = sub.add_parser("restore")
    p.add_argument("which", nargs="?", default="all", choices=("all",) + tuple(RESTORE_NAMES))
    p.set_defaults(func=cmd_restore)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
