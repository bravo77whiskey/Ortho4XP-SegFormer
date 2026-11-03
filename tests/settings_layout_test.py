"""The settings window must show every setting exactly once."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import O4_Cfg_Vars as CFGVARS


def _items(sections):
    return [
        item
        for _title, groups in sections
        for _group, items in groups
        for item in items
    ]


def test_every_tile_setting_appears_once():
    shown = _items(CFGVARS.settings_sections)
    assert len(shown) == len(set(shown))
    wanted = (
        CFGVARS.list_vector_vars + CFGVARS.list_mesh_vars + CFGVARS.list_mask_vars
        + CFGVARS.list_dsf_vars + CFGVARS.list_other_vars
        + CFGVARS.list_bld_vars + CFGVARS.list_veg_vars
    )
    assert sorted(shown) == sorted(wanted)


def test_every_app_setting_appears_once():
    shown = _items(CFGVARS.app_settings_sections)
    assert len(shown) == len(set(shown))
    assert sorted(shown) == sorted(CFGVARS.list_app_vars)


def test_shown_settings_are_defined():
    for item in _items(CFGVARS.settings_sections):
        assert item in CFGVARS.cfg_tile_vars
    for item in _items(CFGVARS.app_settings_sections):
        assert item in CFGVARS.cfg_app_vars
