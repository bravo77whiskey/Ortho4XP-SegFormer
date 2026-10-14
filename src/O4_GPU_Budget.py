"""Keep a fixed amount of VRAM free while several processes share the GPU.

During a batch build several .venv processes use the GPU at once: the
building overlay and its texture helpers, the vegetation overlay and the
colour correction of the next tile, next to whatever else holds VRAM (a
browser, X-Plane).  When the card fills up the Windows driver does not fail
the allocation, it moves memory to system RAM and every GPU process crawls.
So each GPU step (one texture's SegFormer, YOLO, HeightNet or colour pass)
is admitted here first:

* It books the memory it is expected to need: the most that step has needed
  before, measured and shared between processes in a small ledger in the
  temp dir (a default until it has been measured once).
* It is admitted when the card's free memory, minus what other admitted
  steps have booked, still leaves the reserve (O4_VRAM_RESERVE_MB, default
  1024; 0 turns all of this off).  Otherwise it waits; processes that are
  between steps hand their cached memory back while someone waits.
* While admitted, the process's PyTorch allocator is capped at what it holds
  plus what it booked, so a low estimate ends in a CUDA out-of-memory error
  (which the callers handle, and :func:`call` retries with a bigger booking)
  instead of eating into the reserve.

Imports nothing from Ortho4XP and torch only lazily, so the standalone
colour-correction worker can use it too.  Any failure of the ledger itself
turns the guard off for the process rather than failing a build.
"""
import contextlib
import json
import os
import sys
import tempfile
import threading
import time

MB = 1024 * 1024
DEFAULT_RESERVE_MB = 1024
# Used until a step has been measured on this machine: peak reserved growth
# on a dense Shenzhen ZL16 texture, RTX 4080, 2026-10-14 (SegFormer 2.6 GB,
# YOLO b8 2.0 GB, stock YOLO b8 2.6 GB, HeightNet 1.9 GB) plus a margin.
# The colour fit was not measured; it works on a whole group, so it is generous.
DEFAULT_NEED_MB = {
    "segformer": 2816,
    "yolo": 2304,
    "stock-yolo": 2816,
    "heightnet": 2048,
    "colour fit": 6144,
    "colour": 2048,
    "load": 768,
}
FALLBACK_NEED_MB = 1536
SLACK = 1.10                # on top of the estimate: fragmentation, new kernels
SLACK_MB = 128
POLL_S = 0.25
ANNOUNCE_S = 2.0            # say we are waiting once it lasts this long
HOPELESS_S = 15.0           # nothing else running for this long: memory will not come
LOW_WATER_MB = 1024         # after a step, hand the cache back when free gets this close
WATCH_S = 1.0
STALE_S = 12 * 3600         # a booking this old belongs to a hung process
WAIT_STALE_S = 10.0
GIVE_BACK_S = 2.0           # time the watchers get to return their caches
OOM_GROWTH = 1.5

_lock = threading.RLock()   # this process's bookings and the ledger file
_local = threading.local()
_active = {}                # token -> grant (bytes), this process
_local_taint = set()        # tokens that overlapped another step of this process
_cap_base = 0
_process_limit = None       # optional per-process cap (bytes), e.g. SegFormer's setting
_state = {"torch": None, "checked": False, "off": False, "watcher": False, "paths": None}
_tokens = iter(range(1, 1 << 62))
_floors = {}                # label -> booking an OOM showed is needed, this process


def reserve_bytes():
    raw = os.environ.get("O4_VRAM_RESERVE_MB", "").strip()
    try:
        mb = float(raw) if raw else DEFAULT_RESERVE_MB
    except ValueError:
        mb = DEFAULT_RESERVE_MB
    return int(max(0.0, mb) * MB)


def _cuda():
    """torch when the budget applies to this process, else None."""
    if _state["off"]:
        return None
    if not _state["checked"]:
        _state["checked"] = True
        try:
            import torch
            if torch.cuda.is_available() and reserve_bytes() > 0:
                _state["torch"] = torch
        except Exception:
            pass
    return _state["torch"]


def _turn_off(exc):
    if not _state["off"]:
        _state["off"] = True
        print(f"[GPU] VRAM reserve disabled for this process: {exc}", flush=True)
        try:
            _apply_cap(_state["torch"], None)
        except Exception:
            pass


def _is_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text


################################################################################
# Free memory of the whole card
#
# Under Windows WDDM torch.cuda.mem_get_info() only sees the calling
# process's own allocations (every process is told it has the whole card),
# so the card-wide figure comes from NVML, the library nvidia-smi reads.

def _nvml_reader(torch):
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    if sys.platform.startswith("win"):
        lib = None
        for path in ("nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                              "NVIDIA Corporation", "NVSMI", "nvml.dll")):
            try:
                lib = ctypes.WinDLL(path)
                break
            except OSError:
                continue
        if lib is None:
            return None
    else:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2() != 0:
        return None
    handle = ctypes.c_void_p()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    found = False
    if hasattr(props, "pci_bus_id"):
        bus = "%08x:%02x:%02x.0" % (getattr(props, "pci_domain_id", 0), props.pci_bus_id,
                                     props.pci_device_id)
        found = lib.nvmlDeviceGetHandleByPciBusId_v2(bus.encode(), ctypes.byref(handle)) == 0
    if not found and lib.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(torch.cuda.current_device()), ctypes.byref(handle)) != 0:
        return None
    mem = Memory()

    def free():
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            raise OSError("nvmlDeviceGetMemoryInfo failed")
        return int(mem.free)

    free()
    return free


def card_free_bytes():
    """Free VRAM of the whole card (None without CUDA)."""
    torch = _cuda()
    if torch is None:
        return None
    if "nvml" not in _state:
        try:
            _state["nvml"] = _nvml_reader(torch)
        except Exception:
            _state["nvml"] = None
        if _state["nvml"] is None and sys.platform.startswith("win"):
            print("[GPU] NVML not found: VRAM reserve only sees this process's own use",
                  flush=True)
    if _state["nvml"] is not None:
        try:
            return _state["nvml"]()
        except Exception:
            _state["nvml"] = None
    return int(torch.cuda.mem_get_info()[0])


################################################################################
# Ledger shared by every process on the machine

def _paths():
    if _state["paths"] is None:
        d = os.path.join(tempfile.gettempdir(), "o4xp_gpu")
        os.makedirs(d, exist_ok=True)
        dev = _state["torch"].cuda.current_device()
        _state["paths"] = (os.path.join(d, f"gpu{dev}.json"), os.path.join(d, f"gpu{dev}.lock"))
    return _state["paths"]


@contextlib.contextmanager
def _ledger():
    """Exclusive access to the ledger; yields the dict, written back on exit."""
    path, lock_path = _paths()
    fh = open(lock_path, "a+b")
    try:
        deadline = time.monotonic() + 30.0
        if sys.platform.startswith("win"):
            import msvcrt
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("ledger lock timed out")
                    time.sleep(0.01)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            try:
                with open(path) as f:
                    led = json.load(f)
            except (OSError, ValueError):
                led = {}
            for key in ("steps", "waiting", "need", "procs"):
                if not isinstance(led.get(key), dict):
                    led[key] = {}
            _prune(led)
            yield led
            with open(path, "w") as f:
                json.dump(led, f)
        finally:
            if sys.platform.startswith("win"):
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        fh.close()


def _alive(pid):
    if pid == os.getpid():
        return True
    if sys.platform.startswith("win"):
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _prune(led):
    now = time.time()
    alive = {}

    def ok(pid, t):
        if now - float(t) > STALE_S:
            return False
        if pid not in alive:
            alive[pid] = _alive(pid)
        return alive[pid]

    led["steps"] = {k: v for k, v in led["steps"].items() if ok(int(v["pid"]), v["t"])}
    led["procs"] = {k: v for k, v in led["procs"].items()
                    if isinstance(v, dict) and ok(int(k), v["t"])}
    # a waiting step refreshes "seen" every poll; one that stopped is gone
    led["waiting"] = {k: v for k, v in led["waiting"].items()
                      if isinstance(v, dict) and now - float(v.get("seen", 0)) < WAIT_STALE_S
                      and ok(_pid_of(k), v["t"])}


def _pid_of(key):
    return int(str(key).split(":")[0])


def _publish(led, torch):
    """Record what this process holds, so waiters know whether to expect memory back."""
    led["procs"][str(os.getpid())] = {"alloc": int(torch.cuda.memory_allocated()),
                                      "reserved": int(torch.cuda.memory_reserved()),
                                      "t": time.time()}


def _spare_elsewhere(led):
    """Cached memory other processes could hand back."""
    return sum(max(0, int(v["reserved"]) - int(v["alloc"]))
               for k, v in led["procs"].items() if int(k) != os.getpid())


################################################################################
# Allocator cap of this process

def set_process_limit(limit_bytes):
    """Optional fixed cap for this process (None = none); steps never exceed it."""
    global _process_limit
    import torch
    with _lock:
        _process_limit = int(limit_bytes) if limit_bytes else None
        if torch.cuda.is_available():
            _refresh_cap(torch)


def _apply_cap(torch, cap_bytes):
    if torch is None:
        return
    limits = [b for b in (cap_bytes, _process_limit) if b]
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    fraction = 1.0 if not limits else max(1e-3, min(1.0, min(limits) / float(total)))
    torch.cuda.set_per_process_memory_fraction(fraction)


def _refresh_cap(torch):
    _apply_cap(torch, (_cap_base + sum(_active.values())) if _active else None)


################################################################################
# Steps

def _estimate(led, label, need_mb):
    """Booking for a step: given, else measured (exact runs only), else the default."""
    if need_mb:
        got = int(need_mb * MB)
    elif led["need"].get(label):
        got = int(led["need"][label])
    else:
        got = int(DEFAULT_NEED_MB.get(label, DEFAULT_NEED_MB.get(label.split(" ")[0],
                                                                FALLBACK_NEED_MB)) * MB)
    return max(got, _floors.get(label, 0))


def _admit(torch, label, need_mb, floor_bytes=0):
    """Wait until the step fits; returns (token, estimate, measuring).

    First come, first served: a step goes ahead only if it still leaves room
    for the steps that have been waiting longer, so a process running steps
    back to back cannot starve the others.
    """
    global _cap_base
    pid = os.getpid()
    me = f"{pid}:{threading.get_ident()}"
    since = None
    reserve = reserve_bytes()
    t0 = time.monotonic()
    announced = emptied = False
    alone_since = None
    while True:
        with _lock:
            own_busy = bool(_active)
            reserved = torch.cuda.memory_reserved()
            # cache an admitted step of this process may still grow into is
            # not ours to hand out again
            spare = 0 if own_busy else max(0, reserved - torch.cuda.memory_allocated())
            with _ledger() as led:
                _publish(led, torch)
                est = max(_estimate(led, label, need_mb), int(floor_bytes))
                want = max(0, int(est * SLACK) + SLACK_MB * MB - spare)
                free = card_free_bytes()
                booked = sum(int(v["bytes"]) for v in led["steps"].values())
                ahead = sum(int(v["want"]) for k, v in led["waiting"].items()
                            if k != me and (since is None or v["t"] < since))
                head = free - booked - reserve
                others = any(True for v in led["steps"].values())
                now = time.monotonic()
                if others:
                    alone_since = None
                elif alone_since is None:
                    alone_since = now
                # nothing running and no cache elsewhere to hand back (after
                # the watchers had a moment to see us): waiting cannot help
                stuck = (alone_since is not None and since is not None
                         and time.time() - since >= GIVE_BACK_S
                         and _spare_elsewhere(led) < 64 * MB)
                hopeless = stuck or (alone_since is not None and now - alone_since >= HOPELESS_S)
                if want <= head - ahead or (hopeless and not ahead):
                    grant = want if want <= head else max(0, head)
                    token = next(_tokens)
                    led["steps"][f"{pid}:{token}"] = {
                        "pid": pid, "label": label, "bytes": int(grant), "t": time.time()}
                    led["waiting"].pop(me, None)
                    if not own_busy:
                        _cap_base = reserved
                    for other in _active:
                        _local_taint.add(other)
                    _active[token] = int(grant)
                    _refresh_cap(torch)
                    measuring = not own_busy
                    if measuring:
                        torch.cuda.reset_peak_memory_stats()
                    if want > head:
                        print(f"[GPU] {label}: only {max(0, free - reserve) // MB} MB free beyond the "
                              f"{reserve // MB} MB reserve, about {want // MB} MB wanted; running "
                              "within what is left (close other GPU programs if this repeats)",
                              flush=True)
                    elif announced:
                        print(f"[GPU] {label}: admitted after {now - t0:.0f} s", flush=True)
                    return token, est, measuring
                if since is None:
                    since = time.time()
                led["waiting"][me] = {"t": since, "seen": time.time(), "want": int(want)}
        if not emptied:
            emptied = True
            torch.cuda.empty_cache()
            continue
        if not announced and time.monotonic() - t0 >= ANNOUNCE_S:
            announced = True
            print(f"[GPU] {label}: waiting for VRAM (about {want // MB} MB wanted, "
                  f"{free // MB} MB free, {booked // MB} MB booked by running steps, "
                  f"keeping {reserve // MB} MB spare)", flush=True)
        _start_watcher(torch)
        time.sleep(POLL_S)



def _release(torch, token, measuring, alloc0):
    with _lock:
        _active.pop(token, None)
        tainted = token in _local_taint
        _local_taint.discard(token)
        used = None
        if measuring and not tainted:
            # the cap applies to reserved memory, which fragmentation and
            # short-lived workspaces push above the allocated peak
            used = max(0, torch.cuda.max_memory_reserved() - alloc0)
        label = None
        with _ledger() as led:
            entry = led["steps"].pop(f"{os.getpid()}:{token}", None)
            label = entry["label"] if entry else None
            if label and used is not None and used > int(led["need"].get(label, 0)):
                led["need"][label] = int(used)
            waiters = any(_pid_of(k) != os.getpid() for k in led["waiting"])
        _refresh_cap(torch)
        if not _active:
            free = card_free_bytes()
            if waiters or free < reserve_bytes() + LOW_WATER_MB * MB:
                torch.cuda.empty_cache()
        with _ledger() as led:
            _publish(led, torch)
    _start_watcher(torch)


def _raise_estimate(label, at_least):
    # kept in this process only: the shared ledger holds measurements, and a
    # bump after an OOM is a guess that would otherwise inflate every process
    with _lock:
        _floors[label] = max(_floors.get(label, 0), int(at_least))


@contextlib.contextmanager
def step(label, need_mb=None, _floor=0):
    """Run the block as one admitted GPU step (nested steps in a thread are free)."""
    torch = _cuda()
    depth = getattr(_local, "depth", 0)
    if torch is None or depth:
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth = depth
        return
    try:
        token, est, measuring = _admit(torch, label, need_mb, _floor)
        alloc0 = torch.cuda.memory_allocated()
    except Exception as exc:
        _turn_off(exc)
        _local.depth = 1
        try:
            yield
        finally:
            _local.depth = 0
        return
    _local.depth = 1
    try:
        yield
    finally:
        _local.depth = 0
        try:
            _release(torch, token, measuring, alloc0)
        except Exception as exc:
            _turn_off(exc)
        _local.last_estimate = est


def call(label, fn, *args, need_mb=None, retries=2, **kwargs):
    """``fn(*args, **kwargs)`` as one step; a CUDA OOM is retried with a bigger booking."""
    floor = 0
    for attempt in range(retries + 1):
        try:
            with step(label, need_mb, floor):
                return fn(*args, **kwargs)
        except RuntimeError as exc:
            torch = _state["torch"]
            if (attempt == retries or torch is None or _state["off"]
                    or getattr(_local, "depth", 0) or not _is_oom(exc)):
                raise
            floor = int(getattr(_local, "last_estimate", FALLBACK_NEED_MB * MB) * OOM_GROWTH)
            _raise_estimate(label, floor)
            print(f"[GPU] {label}: out of VRAM within its booking ({str(exc).splitlines()[0][:120]}); "
                  f"retrying with {floor // MB} MB", flush=True)
            torch.cuda.empty_cache()


################################################################################
# Between steps: hand cached memory back to processes that are waiting

def _start_watcher(torch):
    with _lock:
        if _state["watcher"] or _state["off"]:
            return
        _state["watcher"] = True

    def watch():
        pid = os.getpid()
        while not _state["off"]:
            time.sleep(WATCH_S)
            try:
                with _lock:
                    if _active:
                        continue
                    spare = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                    with _ledger() as led:
                        if spare >= 64 * MB and any(_pid_of(k) != pid for k in led["waiting"]):
                            torch.cuda.empty_cache()
                        _publish(led, torch)
            except Exception:
                pass

    threading.Thread(target=watch, name="o4-vram-watch", daemon=True).start()
