"""Tree canopy height from the ortho itself, for airport-zone textures (ZL18+).

At ZL18/19 the landcover SegFormer reads patchy airfield meadow as tree. The
Meta/WRI canopy-height network (HighResCanopyHeight, Apache-2.0; network code
vendored in O4_Canopy_Net) predicts height in metres per pixel, so grass is
~0 m and hedges, tree rows and woods stand out. Bing's ZL18+ imagery is aerial
photography, so each patch is first colour-matched to satellite imagery by
Meta's small normalisation network, as their own aerial inference does.

Inference runs at ZL18 scale (0.37 m/px at 52 deg): ZL19 textures are shrunk
2x (~1.8 s each on an RTX 4080), ZL18 textures run as they are (~6.4 s). At
ZL17 scale it is 3.5x faster but misses most garden and street trees. ZL16/17
textures are not handled here — at their own scale the network flips whole
patches, and running them at ZL18 scale would cost 25+ s per texture.

The weights (3.1 GB + 9 MB) are downloaded once from Meta's public bucket into
``<overlay cache>/_canopy_height`` (O4_CANOPY_MODEL_DIR overrides). Without
CUDA, or when the weights are unavailable, ``CanopyModel.load`` returns None
and the caller keeps its SegFormer result.
"""

import os
import pickle
import time
import types

import numpy as np

import O4_GPU_Budget as GPUB

BUCKET = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/models/saved_checkpoints/"
HEIGHT_CKPT = "SSLhuge_satellite.pth"
NORM_CKPT = "aerial_normalization_quantiles_predictor.ckpt"
INFER_ZL = 18          # textures are shrunk to this zoom level's scale
PATCH = 256
STRIDE = 192
BATCH = 16
NEED_MB = 4096         # fp32 ViT-H weights + one bf16 batch
# Heights (m) whose exceedance probability is kept per pixel, for a
# confidence threshold on "at least this tall".
LEVELS_M = (1.0, 1.5, 2.0, 3.0, 5.0)
# Satellite-image normalisation the network was trained with.
MEAN = (0.420, 0.411, 0.296)
STD = (0.213, 0.156, 0.143)
HTTP_TIMEOUT = 30
STREAM_STALL_S = 30
STREAM_RESUMES = 8


def _log_default(msg):
    print(msg, flush=True)


def default_model_dir(tile_cache_dir):
    """``<cache root>/_canopy_height`` from a per-tile ``<cache root>/<grid>/<tile>`` dir."""
    env = os.environ.get("O4_CANOPY_MODEL_DIR", "").strip()
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(tile_cache_dir)))
    return os.path.join(root, "_canopy_height")


# ── Download ────────────────────────────────────────────────────────────────

def _remote_size(session, url):
    r = session.head(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _download(url, dst, log):
    """Ranged, resumable download to ``dst`` (via ``dst.part``)."""
    import requests

    session = requests.Session()
    size = _remote_size(session, url)
    part = dst + ".part"
    pos = os.path.getsize(part) if os.path.exists(part) else 0
    fruitless = 0
    t0 = time.time()
    last = 0.0
    with open(part, "ab") as out:
        while pos < size:
            before = pos
            problem = None
            try:
                with session.get(url, headers={"Range": f"bytes={pos}-"}, stream=True,
                                 timeout=(HTTP_TIMEOUT, STREAM_STALL_S)) as r:
                    r.raise_for_status()
                    if pos and r.status_code != 206:
                        raise IOError(f"server ignored the byte range (HTTP {r.status_code})")
                    for chunk in r.iter_content(4 << 20):
                        out.write(chunk)
                        pos += len(chunk)
                        if time.time() - last > 30:
                            last = time.time()
                            log(f"  [Canopy] {os.path.basename(dst)}: "
                                f"{pos / 1e9:.2f} of {size / 1e9:.2f} GB")
            except (requests.RequestException, ConnectionError, TimeoutError) as exc:
                problem = exc.__class__.__name__
            if pos >= size:
                break
            fruitless = 0 if pos > before else fruitless + 1
            if fruitless > STREAM_RESUMES:
                raise IOError(f"download of {os.path.basename(dst)} stalled at "
                              f"{pos / 1e9:.2f} of {size / 1e9:.2f} GB")
            log(f"  [Canopy] download stopped at {pos / 1e9:.2f} GB "
                f"({problem or 'connection closed early'}); resuming")
            if fruitless:
                time.sleep(min(30, 2 ** fruitless))
    if os.path.getsize(part) != size:
        raise IOError(f"incomplete download of {os.path.basename(dst)}")
    os.replace(part, dst)
    log(f"  [Canopy] downloaded {os.path.basename(dst)} ({size / 1e9:.2f} GB) "
        f"in {time.time() - t0:.0f}s")


def ensure_weights(model_dir, log=_log_default):
    """Paths of both checkpoints, downloading whichever is missing."""
    os.makedirs(model_dir, exist_ok=True)
    paths = []
    for name in (HEIGHT_CKPT, NORM_CKPT):
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            log(f"  [Canopy] downloading {name} from Meta's public bucket "
                f"(one-off; the height model is 3.1 GB) into {model_dir}")
            _download(BUCKET + name, path, log)
        paths.append(path)
    return paths


# ── Loading ─────────────────────────────────────────────────────────────────

def _load_height_state(path, torch):
    """State dict of the height checkpoint, unpickled with ``weights_only``.

    Besides tensors it holds numpy scalars from training (metrics); those are
    the only extra types let through.
    """
    allowed = [(np.core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype]
    allowed += [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    with torch.serialization.safe_globals(allowed):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    return {k.replace("chm_module_.", ""): v for k, v in state.items()}


class _Stub(dict):
    """Stand-in for the Lightning and training-repo objects the normaliser pickled."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _NormUnpickler(pickle.Unpickler):
    """Tensors and their containers only; training objects become stubs; nothing else.

    The checkpoint also pickled its optimiser, scheduler, loss function and
    model class; none of them is needed to read the weights.
    """

    _EXACT = {("collections", "OrderedDict"), ("functools", "partial")}
    _STUBBED = ("pytorch_lightning", "lightning", "pl_modules", "models.", "torch.optim")

    def find_class(self, module, name):
        if module.startswith(self._STUBBED):
            return _Stub
        if ((module, name) in self._EXACT
                or (module == "torch._utils" and name.startswith("_rebuild_"))
                or (module == "torch" and name.endswith("Storage"))):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in {NORM_CKPT}")


def _load_norm_state(path, torch):
    pm = types.ModuleType("o4_canopy_pickle")
    pm.Unpickler = _NormUnpickler
    pm.load = pickle.load
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pm)
    return {k.replace("backbone.", ""): v for k, v in ckpt["state_dict"].items()}


class _HeightNet:
    """SSL ViT-huge backbone + DPT head, as in Meta's inference.SSLAE(huge=True)."""

    def __init__(self, torch):
        import O4_Canopy_Net as NET

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = NET.SSLVisionTransformer(
                    embed_dim=1280, num_heads=20, out_indices=(9, 16, 22, 29),
                    depth=32, pretrained=None)
                self.decode_head = NET.DPTHead(
                    classify=True, in_channels=(1280,) * 4, embed_dims=1280,
                    post_process_channels=[160, 320, 640, 1280])

            def forward(self, x):
                """(B, 1 + len(LEVELS_M), H, W): mean height, then P(height >= level).

                Meta's DPTHead.forward (classify=True) up to its bin spread, kept
                instead of collapsed to the mean; heights x10 as in their SSLModule.
                """
                head = self.decode_head
                feats = head.reassemble_blocks(list(self.backbone(x)))
                feats = [head.convs[i](f) for i, f in enumerate(feats)]
                out = head.fusion_blocks[0](feats[-1])
                for i in range(1, len(head.fusion_blocks)):
                    out = head.fusion_blocks[i](out, feats[-(i + 1)])
                out = head.project(out)
                logit = torch.relu(head.conv_depth(out)) + 0.1
                prob = logit / logit.sum(dim=1, keepdim=True)
                bins_m = 10 * torch.linspace(head.min_depth, head.max_depth, head.n_bins,
                                             device=prob.device, dtype=prob.dtype)
                mean = torch.einsum("ikmn,k->imn", prob, bins_m)
                exceed = [prob[:, bins_m >= level].sum(dim=1) for level in LEVELS_M]
                return torch.stack([mean] + exceed, dim=1)

        self.module = Net().eval()


class CanopyModel:
    """Loaded height + aerial-normalisation networks on the GPU."""

    def __init__(self, torch, device, net, rnet):
        self.torch = torch
        self.device = device
        self.net = net
        self.rnet = rnet
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @classmethod
    def load(cls, model_dir, log=_log_default):
        """The model on CUDA, or None (no CUDA, weights unavailable, load failure)."""
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            log("  [Canopy] no CUDA device; airport-zone trees stay with SegFormer")
            return None
        try:
            height_path, norm_path = ensure_weights(model_dir, log)
            t0 = time.time()
            net = _HeightNet(torch).module
            missing, unexpected = net.load_state_dict(_load_height_state(height_path, torch),
                                                      strict=False)
            if missing:
                raise RuntimeError(f"{len(missing)} weights missing from {HEIGHT_CKPT}")
            import O4_Canopy_Net as NET
            rnet = NET.RNet(n_classes=6).eval()
            rnet.load_state_dict(_load_norm_state(norm_path, torch))
            device = torch.device("cuda")
            model = cls(torch, device, net.to(device), rnet.to(device))
            log(f"  [Canopy] height model ready in {time.time() - t0:.0f}s")
            return model
        except Exception as exc:
            log(f"  [Canopy] unavailable ({exc}); airport-zone trees stay with SegFormer")
            return None

    def maps(self, img_rgb, zl):
        """Canopy maps for an RGB texture at INFER_ZL scale, float16 (1 + len(LEVELS_M), h, w).

        Channel 0 is the mean height in metres; channel 1 + i is the probability
        that the canopy is at least LEVELS_M[i] tall.
        """
        return GPUB.call("canopy", self._maps, img_rgb, zl, need_mb=NEED_MB)

    def _maps(self, img_rgb, zl):
        import cv2

        torch = self.torch
        F = torch.nn.functional
        h, w = img_rgb.shape[:2]
        factor = 2 ** max(0, int(zl) - INFER_ZL)
        sh, sw = max(PATCH, h // factor), max(PATCH, w // factor)
        small = cv2.resize(img_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float() / 255.0
        # Reflect-pad so the texture edge sits inside a patch, not on its rim.
        pad = min(PATCH // 2, sh - 1, sw - 1)
        x = F.pad(x[None], (pad, pad, pad, pad), mode="reflect")[0]
        H, W = x.shape[1:]
        acc = torch.zeros((1 + len(LEVELS_M), H, W))
        wt = torch.zeros((H, W))
        ramp = torch.hann_window(PATCH, periodic=False)
        win = torch.outer(ramp, ramp) + 1e-3
        rows = sorted(set(range(0, H - PATCH + 1, STRIDE)) | {H - PATCH})
        cols = sorted(set(range(0, W - PATCH + 1, STRIDE)) | {W - PATCH})
        coords = [(r, c) for r in rows for c in cols]
        with torch.no_grad():
            for i in range(0, len(coords), BATCH):
                cs = coords[i:i + BATCH]
                batch = torch.stack([x[:, r:r + PATCH, c:c + PATCH] for r, c in cs]).to(self.device)
                # Aerial -> satellite colours: predicted 5/95 % quantiles per patch.
                q = self.rnet(batch)
                flat = batch.flatten(2)
                p5 = torch.quantile(flat, 0.05, dim=2)
                p95 = torch.quantile(flat, 0.95, dim=2)
                gain = (q[:, 3:] - q[:, :3]) / (p95 - p5).clamp(min=1e-3)
                batch = (batch - p5[..., None, None]) * gain[..., None, None] + q[:, :3, None, None]
                batch = (batch - self.mean) / self.std
                # bf16: fp16 overflows inside the ViT-huge and returns NaN.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = self.net(batch).float().cpu()
                for (r, c), p in zip(cs, pred):
                    acc[:, r:r + PATCH, c:c + PATCH] += p * win
                    wt[r:r + PATCH, c:c + PATCH] += win
        out = (acc / wt)[:, pad:pad + sh, pad:pad + sw].numpy()
        return out.astype(np.float16)


def tree_mask(maps, shape, min_height_m, min_confidence=None):
    """Tree pixels of a ``shape`` texture from ``CanopyModel.maps`` output.

    Without ``min_confidence``: the mean height reaches ``min_height_m``.
    With it: the probability that the canopy reaches ``min_height_m`` (which
    must be one of LEVELS_M) is at least ``min_confidence``.
    """
    import cv2

    if min_confidence is None:
        layer = maps[0]
        cut = float(min_height_m)
    else:
        layer = maps[1 + LEVELS_M.index(float(min_height_m))]
        cut = float(min_confidence)
    h, w = shape
    up = cv2.resize(layer.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return up >= cut
