"""Validation of EBSD pipeline and mesh options."""

from dataclasses import FrozenInstanceError, replace

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


def test_pipeline_options_validate_nested_types_and_defaults():
    settings = Settings(ctf="map.ctf")
    assert EbsdOptions(settings).mesh == UpxoMeshOptions()
    assert EbsdOptions(settings, mesh=None).mesh == UpxoMeshOptions()
    with pytest.raises(TypeError, match="import_settings must be Settings"):
        EbsdOptions("map.ctf")
    with pytest.raises(TypeError, match="mesh must be UpxoMeshOptions"):
        EbsdOptions(settings, mesh={})


def test_options_are_frozen_and_replacement_validates():
    settings = Settings(ctf="map.ctf")
    mesh = UpxoMeshOptions()
    options = EbsdOptions(settings, mesh=mesh)
    for value, name in ((settings, "fill"), (mesh, "smoothing"), (options, "unit")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, None)
    with pytest.raises(ValueError, match="unit must be finite and positive"):
        replace(options, unit=0)


def test_import_settings_copy_crop_and_euler_inputs():
    crop = [0, 4, 0, 3]
    euler = [30, 45, 60]
    settings = Settings(ctf="map.ctf", crop=crop, euler_correction=euler)
    crop[0] = euler[0] = 10
    assert settings.crop == (0.0, 4.0, 0.0, 3.0)
    assert settings.euler_correction == (30.0, 45.0, 60.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("crop", [0, 4]),
        ("crop", [0, 4, 0, float("inf")]),
        ("euler_correction", [0, 4]),
        ("euler_correction", [0, 4, float("nan")]),
    ],
)
def test_import_settings_validate_crop_and_euler_shape_and_finiteness(field, value):
    with pytest.raises(ValueError, match=field):
        Settings(ctf="map.ctf", **{field: value})
