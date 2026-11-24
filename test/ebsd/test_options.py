"""Validation of EBSD pipeline and mesh options."""

import pytest

from festim_microstructure.ebsd import EbsdOptions, Settings, UpxoMeshOptions


@pytest.mark.parametrize(
    "field,value",
    [
        ("smoothing", "moving_average"),
        ("smooth_iter", -1),
        ("n_threads", 0),
        ("mesh_size_gb", float("nan")),
        ("smooth_mu", 0),
        ("thin_grain_px", -1),
        ("thin_grain_px", float("nan")),
    ],
)
def test_reject_unsupported_or_invalid_options(field, value):
    with pytest.raises(ValueError):
        UpxoMeshOptions(**{field: value})


def test_pipeline_options_reject_invalid_worker_timeout():
    with pytest.raises(ValueError, match="worker timeout"):
        EbsdOptions(Settings(ctf="map.ctf"), timeout=0)
