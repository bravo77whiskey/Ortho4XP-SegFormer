"""O4_BLD_AGRI=block: greenhouse areas keep buildings off them."""

import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import O4_Building_Overlay as BLD  # noqa: E402


def _pass(routes=False, strict_fit=False):
    tp = object.__new__(BLD._TexturePass)
    tp.run = SimpleNamespace(strict_fit=strict_fit)
    tp.m_per_px = 0.5
    tp.file_counts = {}
    tp.greenhouse_block_mask = None
    tp.static_occ_mask = np.zeros((200, 200), np.uint8)
    tp.static_occ_mask[:5, :] = 1  # tile edge
    tp.static_occ_integral = cv2.integral(tp.static_occ_mask, sdepth=cv2.CV_32S)
    tp.fit_scratch = np.zeros_like(tp.static_occ_mask)
    if routes:
        road = np.zeros_like(tp.static_occ_mask)
        road[150:155, :] = 1
        tp.placement_occ_mask = tp.static_occ_mask | road
        tp.placement_occ_integral = cv2.integral(tp.placement_occ_mask, sdepth=cv2.CV_32S)
    else:
        tp.placement_occ_mask = tp.static_occ_mask
        tp.placement_occ_integral = tp.static_occ_integral
    return tp


def _box(x0, y0, x1, y1):
    return {'points': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _poly(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)


@pytest.mark.parametrize("routes", [False, True])
def test_greenhouse_centres_are_blocked(routes):
    tp = _pass(routes=routes)
    static_before = tp.static_occ_mask.copy()
    tp._block_greenhouse_area([_box(40, 40, 100, 80)])
    assert tp.placement_occ_mask[60, 70]
    assert tp.placement_occ_mask[0, 0]  # earlier blockers kept
    np.testing.assert_array_equal(
        tp.placement_occ_integral, cv2.integral(tp.placement_occ_mask, sdepth=cv2.CV_32S))
    # the zero-tolerance scenery mask is left alone
    np.testing.assert_array_equal(tp.static_occ_mask, static_before)
    if routes:
        assert tp.placement_occ_mask[152, 10]
    assert tp.file_counts['greenhouse_blocked_px'] > 0


def test_overlap_tolerance():
    tp = _pass()
    tp._block_greenhouse_area([_box(40, 40, 100, 80)])
    assert tp._on_greenhouse(_poly(60, 50, 80, 70))       # inside
    assert not tp._on_greenhouse(_poly(95, 50, 135, 70))  # ~12 % grazes
    assert tp._on_greenhouse(_poly(85, 50, 125, 70))      # ~40 % on it
    assert not tp._on_greenhouse(_poly(120, 100, 150, 130))


def test_no_greenhouses_blocks_nothing():
    tp = _pass()
    assert not tp._on_greenhouse(_poly(60, 50, 80, 70))
    tp._block_greenhouse_area([])
    assert tp.greenhouse_block_mask is None


def test_second_pass_accumulates():
    tp = _pass()
    tp._block_greenhouse_area([_box(40, 40, 100, 80)])
    tp._block_greenhouse_area([_box(120, 100, 180, 140)])
    assert tp.greenhouse_block_mask[60, 70] and tp.greenhouse_block_mask[120, 150]
    assert tp.placement_occ_mask[60, 70] and tp.placement_occ_mask[120, 150]


def test_block_is_default_mode(monkeypatch):
    monkeypatch.delenv("O4_BLD_AGRI", raising=False)
    assert BLD._agri_mode() == "block"
    for raw, mode in (("greenhouse", "greenhouse"), ("drop", "drop"), ("0", "off")):
        monkeypatch.setenv("O4_BLD_AGRI", raw)
        assert BLD._agri_mode() == mode
