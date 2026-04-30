import queue
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_DSF_Utils as DSF
import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG


def _dds_bytes(width=4096, height=4096, fourcc="DXT1", mipmap_count=1):
    block_bytes = {"DXT1": 8, "DXT5": 16}[fourcc]
    payload_size = DSF._dds_payload_size(width, height, block_bytes, mipmap_count)
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x0002100F)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 20, payload_size)
    struct.pack_into("<I", header, 28, mipmap_count)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = fourcc.encode("ascii")
    struct.pack_into("<I", header, 108, 0x1000)
    return bytes(header) + (b"\0" * payload_size)


def _save_valid_jpeg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4096, 4096), (32, 64, 96)).save(path, "JPEG")


def _provider(code="TST"):
    return {
        "code": code,
        "imagery_dir": "normal",
        "color_filters": "none",
    }


def _tile():
    return SimpleNamespace(lat=12, lon=34, mask_zl=18)


def test_missing_cached_dds_returns_invalid(tmp_path):
    assert DSF.cached_dds_validation_reason(str(tmp_path / "missing.dds")) == "missing"


def test_truncated_cached_dds_returns_invalid(tmp_path):
    path = tmp_path / "bad.dds"
    path.write_bytes(b"DDS ")

    assert "truncated" in DSF.cached_dds_validation_reason(str(path))


def test_valid_dxt1_and_dxt5_cached_dds_return_valid(tmp_path):
    dxt1_path = tmp_path / "valid_dxt1.dds"
    dxt5_path = tmp_path / "valid_dxt5.dds"
    dxt1_path.write_bytes(_dds_bytes(fourcc="DXT1", mipmap_count=1))
    dxt5_path.write_bytes(_dds_bytes(fourcc="DXT5", mipmap_count=1))

    assert DSF.cached_dds_validation_reason(str(dxt1_path)) is None
    assert DSF.cached_dds_validation_reason(str(dxt5_path)) is None


def test_corrupted_jpeg_returns_invalid(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not a jpeg")

    assert "invalid JPEG" in IMG.cached_jpeg_validation_reason(str(path))


def test_valid_4096_jpeg_returns_valid(tmp_path):
    path = tmp_path / "valid.jpg"
    _save_valid_jpeg(path)

    assert IMG.cached_jpeg_validation_reason(str(path)) is None


def test_build_jpeg_ortho_redownloads_invalid_cached_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(FNAMES, "Imagery_dir", str(tmp_path / "Orthophotos"))
    monkeypatch.setattr(IMG, "providers_dict", {"TST": _provider("TST")})
    monkeypatch.setattr(IMG, "local_combined_providers_dict", {})

    calls = []

    def fake_download(file_dir, file_name, *attrs):
        calls.append((file_dir, file_name, attrs))
        _save_valid_jpeg(Path(file_dir) / file_name)
        return 1

    monkeypatch.setattr(IMG, "download_jpeg_ortho", fake_download)
    file_name = FNAMES.jpeg_file_name_from_attributes(1, 2, 18, "TST")
    file_dir = FNAMES.jpeg_file_dir_from_attributes(12, 34, 18, IMG.providers_dict["TST"])
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    (Path(file_dir) / file_name).write_bytes(b"bad jpeg")

    assert IMG.build_jpeg_ortho(_tile(), 1, 2, 18, "TST") == 1
    assert len(calls) == 1


def test_build_jpeg_ortho_skips_download_for_valid_cached_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(FNAMES, "Imagery_dir", str(tmp_path / "Orthophotos"))
    monkeypatch.setattr(IMG, "providers_dict", {"TST": _provider("TST")})
    monkeypatch.setattr(IMG, "local_combined_providers_dict", {})

    def fail_download(*args):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(IMG, "download_jpeg_ortho", fail_download)
    file_name = FNAMES.jpeg_file_name_from_attributes(1, 2, 18, "TST")
    file_dir = FNAMES.jpeg_file_dir_from_attributes(12, 34, 18, IMG.providers_dict["TST"])
    _save_valid_jpeg(Path(file_dir) / file_name)

    assert IMG.build_jpeg_ortho(_tile(), 1, 2, 18, "TST") == 1


def test_combined_provider_redownloads_invalid_component_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(FNAMES, "Imagery_dir", str(tmp_path / "Orthophotos"))
    monkeypatch.setattr(IMG, "providers_dict", {"TST": _provider("TST")})
    monkeypatch.setattr(
        IMG,
        "local_combined_providers_dict",
        {"COMB": [{"priority": "high", "layer_code": "TST", "extent_code": "global"}]},
    )

    calls = []

    def fake_download(file_dir, file_name, *attrs):
        calls.append((file_dir, file_name, attrs))
        _save_valid_jpeg(Path(file_dir) / file_name)
        return 1

    monkeypatch.setattr(IMG, "download_jpeg_ortho", fake_download)
    file_name = FNAMES.jpeg_file_name_from_attributes(1, 2, 18, "TST")
    file_dir = FNAMES.jpeg_file_dir_from_attributes(12, 34, 18, IMG.providers_dict["TST"])
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    (Path(file_dir) / file_name).write_bytes(b"bad jpeg")

    assert IMG.build_jpeg_ortho(_tile(), 1, 2, 18, "COMB") == 1
    assert len(calls) == 1


def test_invalid_cached_dds_is_queued_for_rebuild(tmp_path):
    path = tmp_path / "corrupt.dds"
    path.write_bytes(b"bad dds")
    download_queue = queue.Queue()
    attrs = (1, 2, 18, "TST")

    rebuilt = DSF._queue_texture_if_needed(
        download_queue,
        attrs,
        str(path),
        "2_1_TST18.dds",
        expected_dxt5=False,
    )

    assert rebuilt is True
    assert download_queue.get_nowait() == attrs
