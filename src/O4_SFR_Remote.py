"""
O4_SFR_Remote.py — Optional remote-GPU offload for SFR model inference.

Runs ONLY the SegFormer and YOLO forward passes on another machine on the
local network (reached via SSH); everything else — stitching, caching,
polygonisation, placement, DSF writing — stays in the local process.

This is deliberately NOT a config setting: the choice is made per run
(session-only GUI checkbox / O4_SFR_REMOTE=1 env for headless runs) and is
honoured only while the host is reachable, with automatic local fallback.

Transport: a persistent worker process (src/scripts/sfr_remote_worker.py) is
started on the remote host through `ssh` and spoken to over its stdin/stdout
with length-prefixed pickled messages. No open ports, no extra daemons — if
the SSH key works, the offload works. The worker exits on stdin EOF, so the
GUI Stop button (which kills the local process tree, including ssh.exe)
also stops the remote side.

Remote layout (created on demand):
  ~/.ortho4xp_sfr/code/         synced copy of the repo src/ tree
  ~/.ortho4xp_sfr/venv/         python venv with torch(+CUDA)/transformers/ultralytics
  ~/.ortho4xp_sfr/checkpoints/  YOLO .pt files uploaded by sha256
"""

import atexit
import hashlib
import io
import os
import re
import socket
import struct
import subprocess
import sys
import tarfile
import threading
import time
import zlib

import numpy as np

_REMOTE_ROOT = "~/.ortho4xp_sfr"
_REMOTE_CODE = f"{_REMOTE_ROOT}/code"
_REMOTE_VENV_PY = f"{_REMOTE_ROOT}/venv/bin/python"
_REMOTE_WORKER = f"{_REMOTE_CODE}/src/scripts/sfr_remote_worker.py"
_REMOTE_IMPORT_CHECK = (
    "import torch, cv2, numpy, PIL, shapely, transformers, ultralytics"
)

# Compress array payloads above this size (hedges Wi-Fi / 100 Mb links while
# costing little on the DDS-sized images that dominate transfer time).
_COMPRESS_THRESHOLD = 1 << 20

_PROBE_TIMEOUT_S = 5


def _no_window():
    if sys.platform.startswith("win"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _ssh_base(connect_timeout=8):
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(connect_timeout)}",
    ]


def default_host():
    """Remote host name (an ~/.ssh/config alias). Env-overridable, not a
    tile/global config setting by design."""
    return os.environ.get("O4_SFR_REMOTE_HOST", "bravo-nobara").strip()


def _ui():
    try:
        import O4_UI_Utils as UI
        return UI
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Wire format helpers (shared with sfr_remote_worker.py, which imports them)
# ─────────────────────────────────────────────────────────────────────────────

def pack_array(arr):
    """Encode a numpy array for the wire (optionally zlib-compressed)."""
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    raw = arr.tobytes()
    compress = len(raw) > _COMPRESS_THRESHOLD
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "z": compress,
        "data": zlib.compress(raw, 1) if compress else raw,
    }


def unpack_array(d):
    if d is None:
        return None
    raw = zlib.decompress(d["data"]) if d["z"] else d["data"]
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def send_msg(stream, obj):
    import pickle
    payload = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack(">Q", len(payload)))
    stream.write(payload)
    stream.flush()


def recv_msg(stream):
    import pickle
    header = stream.read(8)
    if not header or len(header) < 8:
        return None
    (n,) = struct.unpack(">Q", header)
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(min(remaining, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return pickle.loads(b"".join(chunks))


# ─────────────────────────────────────────────────────────────────────────────
# LAN host discovery (for the GUI host picker)
# ─────────────────────────────────────────────────────────────────────────────

def parse_ssh_config_hosts():
    """Return concrete Host entries from ~/.ssh/config.

    [{'alias', 'hostname', 'user'}, …] — wildcard patterns are skipped. These
    are the best picker candidates: they already carry user + key settings.
    """
    path = os.path.expanduser("~/.ssh/config")
    hosts = []
    current = None
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts[0].lower(), parts[1].strip()
                if key == "host":
                    current = None
                    alias = value.split()[0]
                    if not any(c in alias for c in "*?!"):
                        current = {"alias": alias, "hostname": "", "user": ""}
                        hosts.append(current)
                elif current is not None:
                    if key == "hostname":
                        current["hostname"] = value
                    elif key == "user":
                        current["user"] = value
    except OSError:
        return []
    return hosts


def _local_ipv4_addresses():
    ips = set()
    try:
        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe_sock.connect(("8.8.8.8", 53))  # no traffic sent; routing lookup only
        ips.add(probe_sock.getsockname()[0])
        probe_sock.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    return {ip for ip in ips if not ip.startswith("127.")}


def discover_ssh_hosts(port=22, timeout=0.35, progress_cb=None):
    """Scan the local /24 subnet(s) for machines answering on the SSH port.

    Returns [{'ip', 'name', 'banner'}, …] sorted by address. `name` is the
    reverse-resolved hostname when available and `banner` the SSH server's
    identification string (e.g. 'SSH-2.0-OpenSSH_9.9') — both help identify
    the box. `progress_cb(entry)` is called from worker threads as hosts are
    found. This machine's own addresses are excluded. Takes a few seconds.
    """
    from concurrent.futures import ThreadPoolExecutor

    own = _local_ipv4_addresses()
    # Prefer real LAN ranges over VPN/virtual-switch subnets and drop
    # link-local autoconfig entirely; only the top three /24s get scanned.
    def _subnet_priority(net):
        if net.startswith("192.168."):
            return 0
        if net.startswith("10."):
            return 1
        return 2

    subnets = sorted(
        {
            ip.rsplit(".", 1)[0]
            for ip in own
            if not ip.startswith("169.254.")
        },
        key=lambda net: (_subnet_priority(net), net),
    )[:3]
    targets = [
        f"{net}.{i}" for net in subnets for i in range(1, 255)
        if f"{net}.{i}" not in own
    ]
    results = []
    lock = threading.Lock()

    def check(ip):
        try:
            with socket.create_connection((ip, port), timeout=timeout) as conn:
                conn.settimeout(1.0)
                try:
                    banner = conn.recv(128).decode(errors="replace")
                    banner = banner.splitlines()[0].strip() if banner else ""
                except Exception:
                    banner = ""
        except Exception:
            return
        try:
            name = socket.gethostbyaddr(ip)[0]
        except Exception:
            name = ""
        entry = {"ip": ip, "name": name, "banner": banner}
        with lock:
            results.append(entry)
        if progress_cb:
            try:
                progress_cb(entry)
            except Exception:
                pass

    with ThreadPoolExecutor(max_workers=64) as pool:
        list(pool.map(check, targets))
    results.sort(key=lambda e: tuple(int(x) for x in e["ip"].split(".")))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Host availability + one-time remote environment preparation (main process)
# ─────────────────────────────────────────────────────────────────────────────

def probe(host, timeout=_PROBE_TIMEOUT_S):
    """Return True when the host accepts a key-based SSH login right now."""
    if not host:
        return False
    try:
        ret = subprocess.call(
            _ssh_base(timeout) + [host, "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 10,
            **_no_window(),
        )
        return ret == 0
    except Exception:
        return False


def _run_ssh_streamed(host, command, label="[SFR Remote]"):
    """Run a remote command, streaming its output to the local log.

    Registered with the GUI subprocess registry so Stop / window close kills
    it (long pip installs would otherwise be unstoppable).
    """
    proc = subprocess.Popen(
        _ssh_base() + [host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        **_no_window(),
    )
    UI = _ui()
    if UI:
        UI.register_subprocess(proc)

        def _watchdog():
            while proc.poll() is None:
                if getattr(UI, "red_flag", False):
                    UI.kill_subprocess(proc)
                    break
                time.sleep(0.5)

        threading.Thread(target=_watchdog, daemon=True).start()
    try:
        for line in proc.stdout:
            print(f"{label} {line.rstrip()}", flush=True)
        proc.wait()
    finally:
        if UI:
            UI.unregister_subprocess(proc)
    return proc.returncode


def _local_root_dir():
    """Repo/bundle root containing src/ (mirrors O4_SFR_Pipeline logic)."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "sfr_scripts")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sync_code(host):
    """Push the src/ python tree to the remote as a tar stream over ssh."""
    root = _local_root_dir()
    src = os.path.join(root, "src")
    buf = io.BytesIO()
    n_files = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [
                d for d in dirnames if d not in ("__pycache__", "Unused")
            ]
            for name in filenames:
                if not name.endswith((".py", ".json")):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                tar.add(full, arcname=rel)
                n_files += 1
    data = buf.getvalue()
    proc = subprocess.run(
        _ssh_base()
        + [host, f"mkdir -p {_REMOTE_CODE} && tar xzf - -C {_REMOTE_CODE}"],
        input=data,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
        **_no_window(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"code sync failed (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    print(
        f"[SFR Remote] Synced {n_files} source files "
        f"({len(data) // 1024} KB) to {host}:{_REMOTE_CODE}",
        flush=True,
    )


def _env_ok(host):
    try:
        ret = subprocess.call(
            _ssh_base()
            + [host, f"{_REMOTE_VENV_PY} -c '{_REMOTE_IMPORT_CHECK}'"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
            **_no_window(),
        )
        return ret == 0
    except Exception:
        return False


# Like O4_SFR_Pipeline._CUDA_WHEEL_MAP but extended upward: modern Python on
# the remote (3.14 on Nobara) only has torch wheels on the cu126+ indexes.
# Deliberately capped at cu128 even for CUDA-13 drivers — it fully supports
# Turing (2060 Super) and is the most exercised wheel track.
_CUDA_WHEEL_MAP = [
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]


def _remote_torch_index(host):
    """Pick the torch wheel index for the remote GPU via its nvidia-smi."""
    try:
        out = subprocess.check_output(
            _ssh_base() + [host, "nvidia-smi"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
            **_no_window(),
        )
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            driver_cuda = (int(m.group(1)), int(m.group(2)))
            for min_ver, tag in _CUDA_WHEEL_MAP:
                if driver_cuda >= min_ver:
                    return f"https://download.pytorch.org/whl/{tag}", tag
    except Exception:
        pass
    return "https://download.pytorch.org/whl/cpu", "cpu"


def _setup_env(host):
    """Create the remote venv and install inference dependencies (one-time)."""
    index_url, tag = _remote_torch_index(host)
    if tag == "cpu":
        print(
            f"[SFR Remote] WARNING: no NVIDIA GPU detected on {host} — "
            "installing CPU torch (offload will be slow).",
            flush=True,
        )
    else:
        print(f"[SFR Remote] Installing torch ({tag}) on {host} …", flush=True)
    steps = [
        f"python3 -m venv {_REMOTE_ROOT}/venv",
        f"{_REMOTE_VENV_PY} -m pip install --upgrade pip",
        f"{_REMOTE_VENV_PY} -m pip install torch torchvision "
        f"--index-url {index_url}",
        f"{_REMOTE_VENV_PY} -m pip install numpy pillow opencv-python shapely "
        f"'transformers>=4.30.0' 'huggingface-hub>=0.20.0' 'ultralytics>=8.0.0'",
    ]
    command = f"mkdir -p {_REMOTE_ROOT} && " + " && ".join(steps)
    ret = _run_ssh_streamed(host, command)
    if ret != 0:
        print(
            f"[SFR Remote] Remote environment setup failed (exit {ret}).",
            flush=True,
        )
        return False
    return True


# Hosts whose venv already passed the import check this session.
_ready_hosts = set()
_ready_lock = threading.Lock()


def prepare_remote(host):
    """Probe the host and make sure code + venv are ready.

    Returns True when the host can serve inference, False otherwise (caller
    falls back to local inference). Code is re-synced on every call (cheap);
    the venv import check runs once per session per host.
    """
    if not probe(host):
        return False
    try:
        _sync_code(host)
    except Exception as exc:
        print(f"[SFR Remote] Code sync to {host} failed: {exc}", flush=True)
        return False
    with _ready_lock:
        ready = host in _ready_hosts
    if ready:
        return True
    if not _env_ok(host):
        print(
            f"[SFR Remote] Python environment on {host} not ready — running "
            "one-time setup (several GB of downloads; subsequent runs start "
            "immediately) …",
            flush=True,
        )
        if not _setup_env(host):
            return False
        if not _env_ok(host):
            print(
                "[SFR Remote] Environment still not importable after setup — "
                "falling back to local inference.",
                flush=True,
            )
            return False
    with _ready_lock:
        _ready_hosts.add(host)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Result shims — the minimal ultralytics Results surface our consumers touch
# ─────────────────────────────────────────────────────────────────────────────

class _ShimTensor:
    """Duck-types the `.detach().cpu().numpy()` chain used by the OBB readers."""

    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class _ShimOBB:
    __slots__ = ("xyxyxyxy", "xywhr", "conf", "cls")

    def __init__(self, xyxyxyxy, xywhr, conf, cls):
        self.xyxyxyxy = None if xyxyxyxy is None else _ShimTensor(xyxyxyxy)
        self.xywhr = None if xywhr is None else _ShimTensor(xywhr)
        self.conf = None if conf is None else _ShimTensor(conf)
        self.cls = None if cls is None else _ShimTensor(cls)


class _ShimResult:
    __slots__ = ("obb",)

    def __init__(self, obb):
        self.obb = obb


class RemoteYOLO:
    """Stand-in for an `ultralytics.YOLO` model whose predict() runs remotely.

    Exposes `.names` (used by _yolo_model_class_count) and `.predict(...)`
    with the keyword surface the SFR crop loops use. Deliberately does NOT
    define `.fuse` so `hasattr(model, 'fuse')` callers skip fusing.
    """

    def __init__(self, client, sha, names, label):
        self._client = client
        self._sha = sha
        self.names = names
        self._label = label

    def predict(self, source=None, imgsz=640, conf=0.25, iou=0.7, max_det=300,
                device=None, verbose=False, stream=False, batch=1, **kwargs):
        # `device` is intentionally ignored — the worker picks its own GPU.
        single = not isinstance(source, (list, tuple))
        crops = [source] if single else list(source)
        resp = self._client.call({
            "op": "yolo_predict",
            "sha": self._sha,
            "crops": [pack_array(c) for c in crops],
            "imgsz": int(imgsz),
            "conf": float(conf),
            "iou": float(iou),
            "max_det": int(max_det),
            "batch": max(1, int(batch or 1)),
        })
        results = []
        for r in resp["results"]:
            if r is None:
                results.append(_ShimResult(None))
                continue
            results.append(_ShimResult(_ShimOBB(
                unpack_array(r["xyxyxyxy"]),
                unpack_array(r["xywhr"]),
                unpack_array(r["conf"]),
                unpack_array(r["cls"]),
            )))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Client (lives in the .venv overlay subprocess)
# ─────────────────────────────────────────────────────────────────────────────

class RemoteInferenceClient:
    def __init__(self, host):
        self.host = host
        self._proc = None
        self._lock = threading.Lock()
        self._yolo_proxies = {}

    def start(self):
        self._proc = subprocess.Popen(
            _ssh_base() + [self.host, f"{_REMOTE_VENV_PY} -u {_REMOTE_WORKER}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_no_window(),
        )
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        atexit.register(self.close)
        info = self.call({"op": "ping"})
        gpu = info.get("device_name") or "CPU"
        print(
            f"[SFR Remote] Connected to {self.host}: {gpu} "
            f"(torch {info.get('torch', '?')}, "
            f"CUDA {'available' if info.get('cuda') else 'NOT available'})",
            flush=True,
        )

    def _pump_stderr(self):
        try:
            for raw in self._proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    print(f"[SFR Remote] {line}", flush=True)
        except Exception:
            pass

    def call(self, request):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise RuntimeError(
                    f"remote inference worker on {self.host} is not running"
                )
            send_msg(self._proc.stdin, request)
            response = recv_msg(self._proc.stdout)
        if response is None:
            raise RuntimeError(
                f"lost connection to remote inference worker on {self.host}"
            )
        if not response.get("ok"):
            raise RuntimeError(
                response.get("error", "remote inference failed")
            )
        return response

    def close(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    send_msg(proc.stdin, {"op": "shutdown"})
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    # ── SegFormer ────────────────────────────────────────────────────────────

    def segformer_infer(self, kind, img_rgb):
        """Run one SegFormer pass remotely. Returns the (H, W) int8 class map."""
        import O4_SFR_Inference as SEG
        settings = {
            "patch_size": SEG.segformer_patch_size,
            "overlap": SEG.segformer_overlap,
            "batch_size": SEG.segformer_batch_size,
            "confidence_threshold": SEG.segformer_confidence_threshold,
            "use_amp": SEG.segformer_use_amp,
            "channels_last": SEG.segformer_channels_last,
        }
        resp = self.call({
            "op": "segformer",
            "kind": kind,
            "img": pack_array(img_rgb),
            "settings": settings,
        })
        return unpack_array(resp["map"])

    # ── YOLO ─────────────────────────────────────────────────────────────────

    def remote_yolo(self, checkpoint_path):
        """Return a RemoteYOLO proxy, uploading the checkpoint if needed."""
        path = os.path.abspath(str(checkpoint_path))
        sha = _file_sha256(path)
        proxy = self._yolo_proxies.get(sha)
        if proxy is not None:
            return proxy
        name = os.path.basename(path)
        resp = self.call({"op": "yolo_ensure", "sha": sha, "name": name})
        if not resp.get("have"):
            size_mb = os.path.getsize(path) / 1e6
            print(
                f"[SFR Remote] Uploading YOLO checkpoint {name} "
                f"({size_mb:.1f} MB, one-time per version) …",
                flush=True,
            )
            with open(path, "rb") as handle:
                data = handle.read()
            resp = self.call(
                {"op": "yolo_put", "sha": sha, "name": name, "data": data}
            )
        proxy = RemoteYOLO(self, sha, resp["names"], name)
        self._yolo_proxies[sha] = proxy
        return proxy


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Activation (called inside the overlay subprocess)
# ─────────────────────────────────────────────────────────────────────────────

_client = None


def active_client():
    """The connected RemoteInferenceClient, or None when running locally."""
    return _client


def activate(host):
    """Connect to the remote worker and route SegFormer loads through it.

    On any failure this prints a warning and leaves inference local — the
    overlay run must never die just because the remote box went away.
    """
    global _client
    if not host:
        return False
    client = RemoteInferenceClient(host)
    try:
        client.start()
    except Exception as exc:
        print(
            f"[SFR Remote] Could not connect to {host} ({exc}) — "
            "running inference locally.",
            flush=True,
        )
        try:
            client.close()
        except Exception:
            pass
        return False
    _client = client
    import O4_SFR_Inference as SEG
    SEG.remote_client = client
    return True
