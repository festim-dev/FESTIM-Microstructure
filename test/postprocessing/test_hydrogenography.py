"""Synthetic hydrogenography: dot integration, classification and counting."""

import numpy as np
import pytest

from festim_microstructure.postprocessing import hydrogenography as hg

UM = 1e-6


def test_centres_stay_inside_extent():
    pattern = hg.IndicatorPattern(offset=3 * UM)
    centres = pattern.centres((0.0, 100 * UM))
    assert np.all(centres - pattern.diameter / 2 >= 0)
    assert np.all(centres + pattern.diameter / 2 <= 100 * UM)
    assert np.allclose(np.diff(centres), pattern.pitch)
    assert centres.size == 10


def test_uniform_flux_gives_flux_times_exposure():
    pattern = hg.IndicatorPattern()
    edges = np.linspace(0, 200 * UM, 401)
    flux = np.full(400, 3e14)
    for collect in ("footprint", "apex"):
        _, signals = hg.dot_signals(
            pattern, edges, flux, exposure=10.0, collect=collect
        )
        assert np.allclose(signals, 3e15)


def test_piecewise_integration_is_exact_across_a_step():
    # Flux 1 left of x = 12 um, 3 right of it; the dot at 12.5 um spans
    # [10, 15] um, so it collects (2 * 1 + 3 * 3) / 5 = 2.2.
    pattern = hg.IndicatorPattern(offset=2.5 * UM)
    edges = np.array([0.0, 12.0, 40.0]) * UM
    centres, signals = hg.dot_signals(pattern, edges, [1.0, 3.0], exposure=1.0)
    i = np.argmin(np.abs(centres - 12.5 * UM))
    assert signals[i] == pytest.approx(2.2)


def test_mouth_is_credited_to_the_dot_containing_it_only():
    pattern = hg.IndicatorPattern()
    edges = np.array([0.0, 100 * UM])
    q = 5e8  # particles per metre per second
    centres, signals = hg.dot_signals(
        pattern, edges, [0.0], mouths=[(41 * UM, q)], exposure=1.0
    )
    hit = np.flatnonzero(signals)
    assert hit.size == 1
    assert centres[hit[0]] == pytest.approx(40 * UM)
    assert signals[hit[0]] == pytest.approx(q / pattern.diameter)
    # A mouth in the gap between dots is seen by none of them.
    _, missed = hg.dot_signals(
        pattern, edges, [0.0], mouths=[(45 * UM, q)], exposure=1.0
    )
    assert not missed.any()


def test_apex_collection_ignores_flux_in_the_rim():
    pattern = hg.IndicatorPattern()
    # Flux only within the outer 1 um of the dot centred at 20 um.
    edges = np.array([0.0, 17.5, 18.5, 100.0]) * UM
    _, footprint = hg.dot_signals(pattern, edges, [0, 1.0, 0], exposure=1.0)
    _, apex = hg.dot_signals(pattern, edges, [0, 1.0, 0], exposure=1.0, collect="apex")
    assert footprint.max() > 0
    assert not apex.any()


def test_on_boundary_uses_the_full_footprint():
    pattern = hg.IndicatorPattern()
    centres = np.array([10, 20, 30]) * UM
    assert hg.on_boundary(pattern, centres, [22.4 * UM]).tolist() == [
        False,
        True,
        False,
    ]
    assert not hg.on_boundary(pattern, centres, []).any()


def test_classification_modes():
    signals = np.array([0.0, 0.05, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert hg.classify(signals, "range", 0.1).sum() == 2
    assert hg.classify(signals, "absolute", 0.01).sum() == 3
    assert hg.classify(signals, "percentile", 0.5).sum() == 3
    assert not hg.classify(np.ones(4), "range").any()
    with pytest.raises(ValueError):
        hg.classify(signals, "median")


def test_count_normalises_to_dots_on_boundaries():
    discoloured = np.array([True, True, False, True, True])
    on_gb = np.array([True, False, True, False, False])
    counts = hg.count(discoloured, on_gb)
    assert counts["dots_on_gb"] == 2
    assert counts["pct_gb"] == pytest.approx(50.0)
    assert counts["pct_grain"] == pytest.approx(150.0)
    assert counts["pct_grain_err"] == pytest.approx(100 * np.sqrt(3) / 2)


def test_invalid_patterns_are_rejected():
    with pytest.raises(ValueError):
        hg.IndicatorPattern(diameter=12e-6, pitch=10e-6)
    with pytest.raises(ValueError):
        hg.IndicatorPattern(rim=2.5e-6)


def test_response_is_zero_below_detection_and_saturates():
    g = hg.response([0.5e20, 1e20, 1.55e21, 5e21], detection=1e20, saturation=3e21)
    assert g.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError):
        hg.response([1.0], detection=2.0, saturation=1.0)


def test_normalise_maps_an_image_onto_the_unit_interval():
    assert hg.normalise([2.0, 4.0, 3.0]).tolist() == [0.0, 1.0, 0.5]
    assert not hg.normalise([1.0, 1.0]).any()
