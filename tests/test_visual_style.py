# -*- coding: utf-8 -*-
"""Smoke tests for the trimmed visual_style module."""

import re

import src.visual_style as visual_style

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_cluster_colors_are_four_valid_hex():
    assert len(visual_style.CLUSTER_COLORS) == 4
    assert all(_HEX.match(c) for c in visual_style.CLUSTER_COLORS)


def test_kept_tokens_exist():
    assert _HEX.match(visual_style.BLUE)
    assert isinstance(visual_style.SEQUENTIAL_CMAP, str)


def test_dead_symbols_are_gone():
    for name in (
        "apply_publication_style",
        "feature_color",
        "save_figure",
        "PUBLICATION_RC",
        "FEATURE_FAMILY_COLORS",
        "FEATURE_FAMILIES",
        "DIVERGING_CMAP",
    ):
        assert not hasattr(visual_style, name), f"{name} should have been removed"