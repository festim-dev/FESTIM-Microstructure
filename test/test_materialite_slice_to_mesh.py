"""Verify crop identity and coordinates independently of the real example data."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from festim_microstructure.formats.ebsd import EbsdData, read_ebsd, write_ebsd

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "materialite_slice_to_mesh.py"
)
spec = importlib.util.spec_from_file_location(
    "materialite_slice_to_mesh_example", EXAMPLE
)
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)


@pytest.fixture
def source():
    labels = np.ones((6, 7), dtype=np.int32)
    labels[1:, 2:5] = 2
    quats = np.array([[1.0, 0, 0, 0], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
    return {
        "labels": labels,
        "grain_quats": quats,
        "pixel_quats": quats[labels - 1],
        "indexed": np.ones_like(labels, dtype=bool),
        "ever_melted": np.ones_like(labels, dtype=bool),
        "deposition_mask": np.ones_like(labels, dtype=bool),
        "temperature_K": np.full(labels.shape, 300.0),
        "spin_xz": np.array([0, 17, 23])[labels],
        "source_spin_by_grain": np.array([0, 17, 23]),
        "vox": np.array([5.0, 5.0]),
        "spacing_m": np.array([5e-6, 5e-6]),
        "mesh_unit": np.array(1e-6),
        "section_basis": example.SECTION_BASIS.copy(),
        "voxel_corner_origin_m": np.array([-2.5e-6, 157.5e-6]),
        "y_coordinate_m": np.array(250e-6),
        "simulation_complete": np.array(True),
        "completed_steps": np.array(10),
        "requested_steps": np.array(10),
    }


def test_crop_splits_a_parent_without_losing_orientation(source):
    data, provenance = example.prepare_crop(source, (1, 6, 0, 7))
    assert data.ncells == 3
    assert [row["parent_grain_id"] for row in provenance["mapping"]] == [1, 1, 2]
    assert [row["materialite_spin"] for row in provenance["mapping"]] == [17, 17, 23]
    np.testing.assert_array_equal(data.grain_quats, source["grain_quats"][[0, 0, 1]])
    np.testing.assert_array_equal(data.pixel_quats, source["pixel_quats"][1:6])
    np.testing.assert_array_equal(np.unique(data.labels), [1, 2, 3])
    for grain in (1, 2, 3):
        assert ndimage.label(data.labels == grain)[1] == 1
    assert data.indexed.all() and data.labels.min() > 0


def test_crop_preserves_physical_offset_and_does_not_rotate_again(source):
    data, provenance = example.prepare_crop(source, (1, 6, 1, 6))
    np.testing.assert_allclose(provenance["source_corner_xz_m"], [2.5e-6, 162.5e-6])
    np.testing.assert_allclose(provenance["extent_m"], [25e-6, 25e-6])
    assert provenance["mesh_local_corner_m"] == [0, 0]
    assert provenance["source_y_coordinate_m"] == pytest.approx(250e-6)
    np.testing.assert_array_equal(data.vox, [5, 5])
    for row in provenance["mapping"]:
        np.testing.assert_array_equal(
            data.grain_quats[row["grain_id"] - 1],
            source["grain_quats"][row["parent_grain_id"] - 1],
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("indexed", False),
        ("ever_melted", False),
        ("deposition_mask", False),
        ("temperature_K", 2000.0),
        ("temperature_K", np.nan),
        ("spin_xz", 0),
        ("labels", 0),
    ],
)
def test_unprocessed_or_invalid_pixels_are_rejected(source, key, value):
    source[key][2, 2] = value
    with pytest.raises(ValueError, match="processed solid"):
        example.prepare_crop(source, (1, 6, 0, 7))


def test_incomplete_source_is_rejected(source):
    source["simulation_complete"] = np.array(False)
    with pytest.raises(ValueError, match="incomplete"):
        example.prepare_crop(source, (1, 6, 0, 7))
    source["simulation_complete"] = np.array(True)
    source["completed_steps"] = np.array(9)
    with pytest.raises(ValueError, match="every requested step"):
        example.prepare_crop(source, (1, 6, 0, 7))


def test_bad_crop_window_is_rejected(source):
    with pytest.raises(ValueError, match="outside"):
        example.prepare_crop(source, (1, 9, 0, 7))


def test_crop_native_archive_round_trip(source, tmp_path):
    data, _ = example.prepare_crop(source, (1, 6, 0, 7))
    back = read_ebsd(write_ebsd(tmp_path / "crop.npz", data))
    for key in (
        "labels",
        "grain_quats",
        "pixel_quats",
        "source_labels",
        "indexed",
        "vox",
    ):
        np.testing.assert_array_equal(getattr(back, key), getattr(data, key))


def test_diagonal_contacts_become_separate_pieces(source):
    labels = np.array([[1, 2], [2, 1]], dtype=np.int32)
    source["labels"] = labels
    source["pixel_quats"] = source["grain_quats"][labels - 1]
    source["spin_xz"] = source["source_spin_by_grain"][labels]
    for key in ("indexed", "ever_melted", "deposition_mask"):
        source[key] = np.ones_like(labels, dtype=bool)
    source["temperature_K"] = np.full(labels.shape, 300.0)
    data, provenance = example.prepare_crop(source, (0, 2, 0, 2))
    assert data.ncells == 4
    assert [r["parent_grain_id"] for r in provenance["mapping"]] == [1, 1, 2, 2]
    np.testing.assert_array_equal(np.sort(data.labels.ravel()), [1, 2, 3, 4])


def test_shape_quality_and_degenerate_triangles():
    points = np.array([[0, 0], [1e-6, 0], [0, 1e-6]])
    triangles = np.array([[0, 1, 2]])
    quality = example.triangle_quality(points, triangles)
    assert quality["total_area_m2"] == pytest.approx(0.5e-12)
    assert quality["minimum_triangle_quality"] == pytest.approx(np.sqrt(3) / 2)
    with pytest.raises(ValueError, match="degenerate"):
        example.triangle_quality(np.zeros_like(points), triangles)


def test_long_raster_retains_one_pixel_grain_area(source, tmp_path):
    """A span-scaled snap grid moves this grain by more than the area limit."""
    labels = np.ones((102, 13), dtype=np.int32)
    labels[33, 10] = 2
    data = EbsdData(labels, source["grain_quats"], (5.0, 5.0))
    archive = write_ebsd(tmp_path / "long.npz", data)
    base = example.fm.meshing.upxo.mesh_ebsd(
        archive,
        example.fm.UpxoMeshOptions(smoothing="none"),
        workdir=tmp_path,
    )
    report = json.loads(Path(f"{base}-validation.json").read_text())
    assert report["accepted"]
    assert report["mesh"]["max_mesh_polygon_area_relative_error"] < 1e-10
    assert report["geometry"]["max_boundary_displacement_px"] == 0
    np.testing.assert_array_equal(read_ebsd(f"{base}-ebsd.npz").labels, labels)
