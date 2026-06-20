"""seismology.py karakterizasyon + sınır testleri."""

from __future__ import annotations

import numpy as np
import pytest

from eyequake.analysis.seismology import (
    b_value_aki,
    frequency_magnitude_distribution,
    magnitude_of_completeness,
)


def test_fmd_cumulative_monotonic_non_increasing():
    mags = np.array([4.0, 4.1, 4.1, 4.2, 5.0, 4.5, 4.3])
    fmd = frequency_magnitude_distribution(mags, bin_width=0.1)
    assert np.all(np.diff(fmd.cumulative) <= 0)  # M≥ kümülatif azalır


def test_fmd_incremental_sums_to_n():
    mags = np.array([4.0, 4.0, 4.5, 5.0])
    fmd = frequency_magnitude_distribution(mags)
    assert fmd.incremental.sum() == len(mags)
    assert fmd.cumulative.max() == len(mags)


def test_mc_is_float_within_range():
    mags = np.concatenate([np.full(10, 3.0), np.full(5, 3.5), np.full(2, 4.0)])
    mc = magnitude_of_completeness(mags, bin_width=0.5)
    assert isinstance(mc, float)
    assert 3.0 <= mc <= 4.0


def test_b_value_positive_and_near_one_for_synthetic():
    rng = np.random.default_rng(0)
    mags = 3.0 + rng.exponential(0.43, size=3000)  # GR b≈1
    bv = b_value_aki(mags, mc=3.0)
    assert bv.b > 0
    assert bv.n_above_mc > 0
    assert 0.7 < bv.b < 1.2
    assert bv.sigma > 0


def test_b_value_raises_when_too_few():
    with pytest.raises(ValueError):
        b_value_aki(np.array([4.0]), mc=4.0)
