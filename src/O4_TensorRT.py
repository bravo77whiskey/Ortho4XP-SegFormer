"""TensorRT fp16 engines for the landcover SegFormer and HeightNet.

Both models run faster through TensorRT at fp16 with practically the same
output: 99.9 % of SegFormer pixels identical, HeightNet within 0.02 m, building
overlay stages about a third shorter (docs/model_size_study_2026-11.md).

An engine is tied to the GPU, the TensorRT version and the weights, so it is
built here on first use (ONNX export, then a TensorRT build, a few minutes) and
cached under ``<overlay cache>/_tensorrt`` (O4_TENSORRT_DIR, set by
O4_Overlay_Pipeline). One process builds an engine at a time; the others keep
their torch model meanwhile. Without CUDA, the tensorrt/onnx packages, or with
the overlay_tensorrt setting off (O4_TENSORRT=0), ``engine`` returns None and
the caller keeps its torch model.

Engines take a fixed batch; ``Runner`` splits larger batches and pads smaller
ones. Scratch memory is allocated through torch on every call, so the GPU
budget (O4_GPU_Budget) sees it and nothing is held between textures.
"""

import hashlib
import os
import tempfile
import threading
import time

import O4_GPU_Budget as GPUB

ENV_ON = "O4_TENSORRT"
ENV_DIR = "O4_TENSORRT_DIR"
BUILD_VERSION = 1           # bump when the export or builder settings change
WORKSPACE_MB = 4096
BUILD_NEED_MB = 6144        # booked while building: workspace + weights + tactics
LOCK_STALE_S = 1800         # a build lock older than this belongs to a dead process

_trt_module = None
_failed = set()


def enabled():
    return os.environ.get(ENV_ON, "1").strip().lower() not in ("0", "false", "no", "off")


def engine_dir():
    path = os.environ.get(ENV_DIR, "").strip()
    if path:
        return path
    import O4_Persistent_Cache as PCACHE
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(PCACHE.cache_root(root), "_tensorrt")


def _trt():
    global _trt_module
    if _trt_module is None:
        try:
            import tensorrt
            _trt_module = tensorrt
        except Exception:
            _trt_module = False
    return _trt_module or None


def _engine_path(trt, torch, name, source_id):
    props = torch.cuda.get_device_properties(0)
    key = "|".join((name, str(source_id), trt.__version__, props.name,
                    f"{props.major}.{props.minor}", str(BUILD_VERSION)))
    return os.path.join(engine_dir(), f"{name}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.engine")


def _take_lock(lock_path):
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) < LOCK_STALE_S:
                    return False
                os.remove(lock_path)
            except OSError:
                return False
    return False


def _build(trt, path, export_onnx, half_io):
    logger = trt.Logger(trt.Logger.WARNING)
    work = tempfile.mkdtemp(prefix="onnx_", dir=os.path.dirname(path))
    onnx_path = os.path.join(work, "model.onnx")
    try:
        export_onnx(onnx_path)
        builder = trt.Builder(logger)
        net = builder.create_network(0)
        parser = trt.OnnxParser(net, logger)
        # parse_from_file finds weights the exporter wrote beside a big model
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB << 20)
        cfg.set_flag(trt.BuilderFlag.FP16)
        if half_io:
            for i in range(net.num_inputs):
                net.get_input(i).dtype = trt.float16
            for i in range(net.num_outputs):
                net.get_output(i).dtype = trt.float16
        data = builder.build_serialized_network(net, cfg)
        if data is None:
            raise RuntimeError("TensorRT build failed")
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        for root, dirs, files in os.walk(work, topdown=False):
            for n in files:
                os.remove(os.path.join(root, n))
            for n in dirs:
                os.rmdir(os.path.join(root, n))
        os.rmdir(work)


def engine(name, source_id, export_onnx, half_io=False, log=print):
    """``Runner`` for engine ``name``, built on first use, or None to keep torch.

    ``source_id`` names the weights and input shapes; ``export_onnx(path)``
    writes the model to ONNX at fixed shapes (fp32; ``half_io`` makes the
    engine's inputs and outputs fp16).
    """
    if not enabled() or name in _failed:
        return None
    trt = _trt()
    if trt is None:
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    try:
        path = _engine_path(trt, torch, name, source_id)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            lock = path + ".lock"
            if not _take_lock(lock):
                log(f"[TensorRT] {name}: another process is building the engine; using torch for now")
                return None
            try:
                if not os.path.exists(path):
                    log(f"[TensorRT] {name}: building the engine for this GPU (one-time, a few minutes) ...")
                    t0 = time.time()
                    with GPUB.step("tensorrt build", need_mb=BUILD_NEED_MB):
                        _build(trt, path, export_onnx, half_io)
                    log(f"[TensorRT] {name}: engine built in {time.time() - t0:.0f} s")
            finally:
                try:
                    os.remove(lock)
                except OSError:
                    pass
        return Runner(trt, torch, path)
    except Exception as exc:
        _failed.add(name)
        log(f"[TensorRT] {name}: {type(exc).__name__}: {str(exc)[:300]}; using torch")
        return None


class Runner:
    """Calls a fixed-batch engine like the torch model: inputs in engine order
    (CUDA tensors, cast to the engine's dtypes), returns the output tensor(s)."""

    _DTYPES = {"FLOAT": "float32", "HALF": "float16", "INT32": "int32", "INT64": "int64"}

    def __init__(self, trt, torch, path):
        self.torch = torch
        self.lock = threading.Lock()  # one set of io buffers
        with open(path, "rb") as f:
            # ERROR: the default-stream performance note would print every process
            self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot load {os.path.basename(path)}")
        self.ctx = self.engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        self.scratch_bytes = int(self.engine.device_memory_size_v2)
        self.inputs, self.outputs, self.bufs = [], [], {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = getattr(torch, self._DTYPES[self.engine.get_tensor_dtype(name).name])
            self.bufs[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)),
                                          dtype=dtype, device="cuda")
            self.ctx.set_tensor_address(name, self.bufs[name].data_ptr())
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs).append(name)
        self.batch = self.bufs[self.inputs[0]].shape[0]

    def input_shape(self, i=0):
        return tuple(self.bufs[self.inputs[i]].shape)

    def __call__(self, *inputs):
        n = inputs[0].shape[0]
        parts = [self._run([x[s:s + self.batch] for x in inputs])
                 for s in range(0, n, self.batch)]
        outs = [p[0] if len(parts) == 1 else self.torch.cat(p) for p in zip(*parts)]
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _run(self, inputs):
        with self.lock:
            return self._run_locked(inputs)

    def _run_locked(self, inputs):
        n = inputs[0].shape[0]
        for name, x in zip(self.inputs, inputs):
            buf = self.bufs[name]
            buf[:n].copy_(x)
            if n < self.batch:
                buf[n:].zero_()
        torch = self.torch
        scratch = torch.empty(self.scratch_bytes, dtype=torch.uint8, device="cuda")
        self.ctx.set_device_memory(scratch.data_ptr(), self.scratch_bytes)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        # the caching allocator reuses the scratch only after this stream's work
        del scratch
        return [self.bufs[name][:n].clone() for name in self.outputs]
