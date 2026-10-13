"""X-Plane 12 art-control tuning: recover defaults, apply presets, restore.

X-Plane 12 registers ~8k "art controls" (engine tuning values) in
X-Plane.exe.  Resources/settings.txt maps UI settings to art-control values
via SETTING_1 lines, so any art control can be overridden there.  This tool:

  defaults   decode the default of every float art control from X-Plane.exe
             (static-initialiser call sites: lea rdx,[name]; xmm3 = default)
  status     show which preset (if any) is applied
  apply      write a preset block into settings.txt (backs the file up first)
  restore    remove the preset block (backs the file up first), or copy a
             backup back with --from

Tested in-sim on 12.4.2: settings.txt only reaches values X-Plane reads once
(e.g. shadow/csm/far_limit_exterior on an existing Laminar line).  Values the
engine recomputes every frame (colour management, ev100 bias, scattering...)
ignore it, whether added as new lines or appended to Laminar's - so the
presets/ files do NOT work; use shader_patch.py for those.

Presets are text files: one "art/control value" pair per line, # comments.
The block is piggybacked on the always-present renopt_gload setting, with one
SETTING_1 line per setting value so the overrides apply whatever it is set to.

Every write first copies settings.txt to the backup directory; the very first
backup is also kept as settings.original.txt and never overwritten.
"""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

XP_ROOT = Path(r"D:\SteamLibrary\steamapps\common\X-Plane 12")
DATA_DIR = Path(r"E:\Ortho4XP_Review\xp12_lighting")
PRESET_DIR = Path(__file__).resolve().parent / "presets"

BEGIN = "# >>> o4xp-lighting preset: "
END = "# <<< o4xp-lighting"
HOST_SETTING = "renopt_gload"
HOST_VALUES = (0, 1)
_NAME_RE = re.compile(rb"\x00([a-z][a-z0-9_]{1,30}/[a-z0-9_/]{2,60})\x00")


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

class _Image:
    def __init__(self, path):
        import pefile
        pe = pefile.PE(str(path), fast_load=True)
        self.data = pe.__data__
        self.base = pe.OPTIONAL_HEADER.ImageBase
        self.secs = {s.Name.rstrip(b"\0").decode(): (s.VirtualAddress, s.PointerToRawData, s.SizeOfRawData)
                     for s in pe.sections}

    def off2rva(self, off):
        for va, raw, size in self.secs.values():
            if raw <= off < raw + size:
                return va + off - raw
        return None

    def rva2off(self, rva):
        for va, raw, size in self.secs.values():
            if va <= rva < va + size:
                return raw + rva - va
        return None


_XMM_MODRM = {0x05: 0, 0x0D: 1, 0x15: 2, 0x1D: 3, 0x25: 4, 0x2D: 5, 0x35: 6, 0x3D: 7}


def _track_xmm3(img, text, tva, start, end):
    """Follow the few instruction forms MSVC uses to load the float argument."""
    xmm = {}
    i = start
    while i < end:
        b = text[i:i + 6]
        if b[:3] == b"\xf3\x0f\x10" and b[3] in _XMM_MODRM:          # movss xmmN,[rip+d]
            disp = struct.unpack_from("<i", text, i + 4)[0]
            off = img.rva2off(tva + i + 8 + disp)
            if off is not None:     # None = misaligned match inside another instruction
                xmm[_XMM_MODRM[b[3]]] = struct.unpack_from("<f", img.data, off)[0]
                i += 8
                continue
        if b[:4] == b"\xf3\x44\x0f\x10" and b[4] in _XMM_MODRM:      # movss xmm8+N,[rip+d]
            disp = struct.unpack_from("<i", text, i + 5)[0]
            off = img.rva2off(tva + i + 9 + disp)
            if off is not None:
                xmm[8 + _XMM_MODRM[b[4]]] = struct.unpack_from("<f", img.data, off)[0]
                i += 9
                continue
        if b[:2] == b"\x0f\x57" and b[2] >> 6 == 3 and (b[2] >> 3) & 7 == b[2] & 7:   # xorps xmmN,xmmN
            xmm[b[2] & 7] = 0.0
            i += 3
            continue
        if b[:2] == b"\x0f\x28" and b[2] >> 6 == 3:                   # movaps xmmD,xmmS
            xmm[(b[2] >> 3) & 7] = xmm.get(b[2] & 7)
            i += 3
            continue
        if b[:3] == b"\x41\x0f\x28" and b[3] >> 6 == 3:               # movaps xmmD,xmm8+S
            xmm[(b[3] >> 3) & 7] = xmm.get(8 + (b[3] & 7))
            i += 4
            continue
        i += 1
    return xmm.get(3)


def extract_defaults(exe):
    """Return {name: default float or None} for every art control with a direct call site."""
    import numpy as np
    img = _Image(exe)
    tva, traw, tsize = img.secs[".text"]
    text = img.data[traw:traw + tsize]
    names = {}
    for m in _NAME_RE.finditer(img.data):
        names[img.off2rva(m.start() + 1)] = m.group(1).decode()
    want = np.array(sorted(names), dtype=np.int64)
    arr = np.frombuffer(text, dtype=np.uint8)
    refs = defaultdict(list)
    for align in range(4):
        n = (tsize - align) // 4
        disp = np.frombuffer(arr[align:align + n * 4].tobytes(), dtype="<i4").astype(np.int64)
        pos = align + np.arange(n) * 4
        tgt = tva + pos + 4 + disp
        hit = np.isin(tgt, want)
        for p, t in zip(pos[hit], tgt[hit]):
            refs[int(t)].append(int(p))
    sites = defaultdict(list)       # name -> [(call_target, xmm3)]
    for rva, plist in refs.items():
        for p in plist:
            if text[p - 3] not in (0x48, 0x4C) or text[p - 2] != 0x8D or text[p - 1] != 0x15:
                continue            # only lea rdx,[rip+name]: name is the 2nd argument
            j = p + 4
            while j < p + 48 and text[j] != 0xE8:
                j += 1
            if text[j] != 0xE8:
                continue
            call = tva + j + 5 + struct.unpack_from("<i", text, j + 1)[0]
            k = p - 4
            while k > p - 64 and text[k] != 0xCC and not (text[k] == 0xE8 and k + 5 <= p - 4):
                k -= 1
            start = k + 1 if text[k] == 0xCC else k + 5
            sites[names[rva]].append((call, _track_xmm3(img, text, tva, start, j)))
    # the float art-control constructor is by far the most common call target
    ctor = Counter(c for v in sites.values() for c, _ in v).most_common(1)[0][0]
    out = {}
    for name in sorted(set(names.values())):
        vals = [x for c, x in sites.get(name, []) if c == ctor]
        out[name] = vals[0] if vals else None
    return out, hex(ctor)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_defaults(refresh=False):
    exe = XP_ROOT / "X-Plane.exe"
    cache = DATA_DIR / "defaults.json"
    digest = _md5(exe)
    if cache.exists() and not refresh:
        doc = json.loads(cache.read_text())
        if doc.get("exe_md5") == digest:
            return doc["defaults"]
        print("X-Plane.exe changed since defaults were extracted - re-extracting")
    defaults, ctor = extract_defaults(exe)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"exe_md5": digest, "ctor": ctor, "defaults": defaults}, indent=1))
    return defaults


# --------------------------------------------------------------------------
# settings.txt
# --------------------------------------------------------------------------

def settings_path():
    return XP_ROOT / "Resources" / "settings.txt"


def read_settings():
    # newline="" keeps the file's own line endings so restore is byte-exact
    with open(settings_path(), encoding="utf-8", newline="") as f:
        return f.read()


def write_settings(text):
    with open(settings_path(), "w", encoding="utf-8", newline="") as f:
        f.write(text)


def backup_settings(reason):
    src = settings_path()
    bdir = DATA_DIR / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    original = bdir / "settings.original.txt"
    if not original.exists():
        shutil.copy2(src, original)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = bdir / f"settings_{stamp}_{reason}.txt"
    n = 2
    while dst.exists():
        dst = bdir / f"settings_{stamp}_{reason}_{n}.txt"
        n += 1
    shutil.copy2(src, dst)
    if _md5(dst) != _md5(src):
        sys.exit(f"backup verification failed for {dst}")
    print(f"backup: {dst}")
    return dst


def read_preset(path):
    controls = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, value = line.split()
        controls[name] = float(value)
    return controls


def strip_block(text):
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith(BEGIN):
            inside = True
            continue
        if inside and line.startswith(END):
            inside = False
            continue
        if not inside:
            out.append(line)
    if inside:
        sys.exit("settings.txt has an unterminated preset block - restore from a backup with --from")
    return "".join(out)


def applied_preset(text):
    for line in text.splitlines():
        if line.startswith(BEGIN):
            return line[len(BEGIN):].strip()
    return None


def make_block(name, controls):
    pairs = "\t".join(f"{k} {v:g}" for k, v in controls.items())
    lines = [f"{BEGIN}{name}\n"]
    lines += [f"SETTING_1 {HOST_SETTING} {v} {pairs}\n" for v in HOST_VALUES]
    lines.append(f"{END}\n")
    return "".join(lines)


def insert_block(text, block):
    """Insert after the host setting's last SETTING_1 line so it stays inside its section."""
    if "\r\n" in text:
        block = block.replace("\n", "\r\n")
    lines = text.splitlines(keepends=True)
    last = max((i for i, l in enumerate(lines) if l.startswith(f"SETTING_1 {HOST_SETTING} ")), default=None)
    if last is None:
        sys.exit(f"{HOST_SETTING} not found in settings.txt - the file format changed")
    return "".join(lines[:last + 1]) + block + "".join(lines[last + 1:])


def xplane_running():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq X-Plane.exe"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return "X-Plane.exe" in out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_defaults(args):
    defaults = load_defaults(refresh=args.refresh)
    prefixes = tuple(args.prefix) if args.prefix else None
    for name, value in defaults.items():
        if prefixes and not name.startswith(prefixes):
            continue
        if value is None and not args.all:
            continue
        print(f"{name:50s} {'?' if value is None else f'{value:g}'}")


def cmd_status(_args):
    text = read_settings()
    print(f"applied preset: {applied_preset(text) or '(none)'}")


def cmd_apply(args):
    path = Path(args.preset)
    if not path.exists():
        path = PRESET_DIR / f"{args.preset}.txt"
    if not path.exists():
        sys.exit(f"no such preset: {args.preset}")
    controls = read_preset(path)
    defaults = load_defaults()
    unknown = [k for k in controls if k not in defaults]
    if unknown and not args.force:
        sys.exit(f"not art controls in this X-Plane build: {unknown} (use --force to write anyway)")
    for k, v in controls.items():
        d = defaults.get(k)
        print(f"  {k:45s} {'?' if d is None else f'{d:g}':>10s} -> {v:g}")
    # Laminar's own lines may set the same control later in the file; which one
    # wins is not known, so flag it rather than guess.
    others = strip_block(read_settings())
    clashes = [k for k in controls if re.search(rf"\s{re.escape(k)}\s", others)]
    if clashes:
        print(f"warning: also set by other settings.txt lines (may override the preset): {clashes}")
    if xplane_running():
        print("note: X-Plane is running; settings.txt is read at startup, restart it to see the change")
    backup_settings(f"before_{path.stem}")
    text = strip_block(read_settings())
    write_settings(insert_block(text, make_block(path.stem, controls)))
    print(f"applied preset {path.stem}")


def cmd_restore(args):
    if args.source:
        src = Path(args.source)
        if not src.exists():
            sys.exit(f"no such backup: {src}")
        backup_settings("before_restore")
        shutil.copy2(src, settings_path())
        print(f"restored settings.txt from {src}")
        return
    text = read_settings()
    if applied_preset(text) is None:
        print("no preset applied - nothing to do")
        return
    backup_settings("before_restore")
    write_settings(strip_block(text))
    print("preset block removed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("defaults")
    p.add_argument("prefix", nargs="*", help="only names starting with these prefixes")
    p.add_argument("--all", action="store_true", help="include controls whose default was not decoded")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_defaults)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("apply")
    p.add_argument("preset", help="preset name (presets/<name>.txt) or path")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_apply)
    p = sub.add_parser("restore")
    p.add_argument("--from", dest="source", help="copy this backup file back instead of stripping the block")
    p.set_defaults(func=cmd_restore)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
