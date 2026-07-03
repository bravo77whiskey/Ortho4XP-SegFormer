#!/usr/bin/env python3
"""
sfr_remote_worker.py — remote GPU inference worker for Ortho4XP SFR offload.

Started on the remote host by O4_SFR_Remote.RemoteInferenceClient as:

    ~/.ortho4xp_sfr/venv/bin/python -u ~/.ortho4xp_sfr/code/src/scripts/sfr_remote_worker.py

Speaks length-prefixed pickled messages on stdin/stdout (helpers shared with
O4_SFR_Remote, which is part of the synced code tree). All human-readable
output goes to stderr, which the client relays into the Ortho4XP log with a
"[SFR Remote]" prefix. Exits on stdin EOF, so killing the local ssh process
(GUI Stop button) tears the worker down too.

Serves exactly two model families:
  - SegFormer semantic segmentation — runs the real O4_SFR_Inference.run_inference
    on this machine's GPU, so patch/blend behaviour is identical to a local run.
  - YOLO OBB predict — checkpoints are uploaded once (by sha256) and cached
    under ~/.ortho4xp_sfr/checkpoints/.
"""

import os
import sys
import traceback

# Keep the wire channel clean: capture binary stdout for the protocol before
# redirecting all prints (ours and the model libraries') to stderr.
_WIRE_OUT = sys.stdout.buffer
_WIRE_IN = sys.stdin.buffer
sys.stdout = sys.stderr

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("YOLO_VERBOSE", "false")

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
for _path in (_ROOT_DIR, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from O4_SFR_Remote import pack_array, unpack_array, send_msg, recv_msg  # noqa: E402

_CKPT_DIR = os.path.expanduser("~/.ortho4xp_sfr/checkpoints")

_yolo_models = {}          # sha256 -> ultralytics.YOLO
_segformer_ready = set()   # kinds already loaded ('veg' / 'bld')


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _torch():
    import torch
    return torch


def _handle_ping(_request):
    torch = _torch()
    cuda = torch.cuda.is_available()
    return {
        "ok": True,
        "torch": torch.__version__,
        "cuda": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
    }


def _apply_segformer_settings(settings):
    import O4_SFR_Inference as SEG
    SEG.segformer_patch_size = int(settings.get("patch_size", 512))
    SEG.segformer_overlap = int(settings.get("overlap", 64))
    SEG.segformer_batch_size = int(settings.get("batch_size", 0))
    SEG.segformer_confidence_threshold = float(
        settings.get("confidence_threshold", 0.5)
    )
    SEG.segformer_use_amp = bool(settings.get("use_amp", False))
    SEG.segformer_channels_last = bool(settings.get("channels_last", False))


def _handle_segformer(request):
    import O4_SFR_Inference as SEG
    _apply_segformer_settings(request.get("settings") or {})
    kind = request["kind"]
    if kind == "bld":
        model, processor, device = SEG.load_building_model(None)
    else:
        model, processor, device = SEG.load_vegetation_model(None)
    if kind not in _segformer_ready:
        _segformer_ready.add(kind)
        log(f"SegFormer '{kind}' model ready on {device}")
    img = unpack_array(request["img"])
    class_map = SEG.run_inference(model, device, img, processor)
    return {"ok": True, "map": pack_array(class_map)}


def _ckpt_path(sha):
    return os.path.join(_CKPT_DIR, f"{sha}.pt")


def _load_yolo(sha):
    model = _yolo_models.get(sha)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(_ckpt_path(sha))
        _yolo_models[sha] = model
    return model


def _yolo_names(model):
    names = getattr(model, "names", None)
    if names is None and getattr(model, "model", None) is not None:
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return dict(names)
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def _handle_yolo_ensure(request):
    sha = request["sha"]
    if not os.path.exists(_ckpt_path(sha)):
        return {"ok": True, "have": False}
    model = _load_yolo(sha)
    return {"ok": True, "have": True, "names": _yolo_names(model)}


def _handle_yolo_put(request):
    sha = request["sha"]
    os.makedirs(_CKPT_DIR, exist_ok=True)
    path = _ckpt_path(sha)
    tmp = path + ".part"
    with open(tmp, "wb") as handle:
        handle.write(request["data"])
    os.replace(tmp, path)
    log(f"Stored YOLO checkpoint {request.get('name', sha)} "
        f"({os.path.getsize(path) // (1 << 20)} MB)")
    model = _load_yolo(sha)
    return {"ok": True, "have": True, "names": _yolo_names(model)}


def _extract_obb(result):
    obb = getattr(result, "obb", None)
    if obb is None:
        return None

    def as_np(t):
        return None if t is None else t.detach().cpu().numpy()

    return {
        "xyxyxyxy": pack_array(as_np(getattr(obb, "xyxyxyxy", None))),
        "xywhr": pack_array(as_np(getattr(obb, "xywhr", None))),
        "conf": pack_array(as_np(getattr(obb, "conf", None))),
        "cls": pack_array(as_np(getattr(obb, "cls", None))),
    }


def _handle_yolo_predict(request):
    torch = _torch()
    model = _load_yolo(request["sha"])
    crops = [unpack_array(c) for c in request["crops"]]
    batch = max(1, int(request.get("batch", 1)))
    kwargs = dict(
        imgsz=int(request["imgsz"]),
        conf=float(request["conf"]),
        iou=float(request["iou"]),
        max_det=int(request["max_det"]),
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        stream=True,
    )
    source = crops[0] if len(crops) == 1 else crops
    if len(crops) > 1:
        kwargs["batch"] = batch
    out = []
    with torch.inference_mode():
        results = model.predict(source=source, **kwargs)
        for result in results:
            out.append(_extract_obb(result))
        del results
    return {"ok": True, "results": out}


_HANDLERS = {
    "ping": _handle_ping,
    "segformer": _handle_segformer,
    "yolo_ensure": _handle_yolo_ensure,
    "yolo_put": _handle_yolo_put,
    "yolo_predict": _handle_yolo_predict,
}


def main():
    log(f"worker started (python {sys.version.split()[0]}, pid {os.getpid()})")
    while True:
        try:
            request = recv_msg(_WIRE_IN)
        except Exception:
            break
        if request is None or request.get("op") == "shutdown":
            break
        handler = _HANDLERS.get(request.get("op"))
        if handler is None:
            response = {"ok": False, "error": f"unknown op {request.get('op')!r}"}
        else:
            try:
                response = handler(request)
            except Exception as exc:
                # Preserve CUDA OOM text so the client's batch-size fallback
                # (SEGFORMER.is_cuda_oom) still triggers on the local side.
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                log(traceback.format_exc())
                try:
                    _torch().cuda.empty_cache()
                except Exception:
                    pass
        try:
            send_msg(_WIRE_OUT, response)
        except Exception:
            break
    log("worker exiting")


if __name__ == "__main__":
    main()
