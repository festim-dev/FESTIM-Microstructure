"""Geometry preservation and compatibility of the UPXO backend."""

import json
from pathlib import Path

import numpy as np
import pytest

from festim_microstructure.ebsd import (
    UpxoMeshOptions,
    mesh_grains,
    upxo,
    worker,
)
from festim_microstructure.ebsd.archive import EbsdData, read_ebsd, write_ebsd
from festim_microstructure.ebsd.orientation import euler_bunge_to_quat


def closed_loop_labels(case):
    """A host grain with closed interior boundaries."""
    labels = np.ones((24, 24), dtype=np.int32)
    if case == "disconnected":
        labels[4:8, 4:8] = labels[15:19, 15:19] = 2
    elif case == "void":
        labels[8:16, 8:16] = 0
    else:
        labels[8:16, 8:16] = 2
        if case == "nested_islands":
            labels[9:15, 9:15] = 3
    return labels


def test_geometry_gate_rejects_lost_grain_contact():
    pytest.importorskip("shapely")
    from shapely.geometry import box

    reference = {1: box(0, 0, 1, 1), 2: box(1, 0, 2, 1)}
    moved = {1: reference[1], 2: box(1.01, 0, 2, 1)}
    report = upxo.geometry_metrics(reference, moved)
    assert not report["accepted"]
    assert report["lost_neighbor_pairs"] == [(1, 2)]
    assert report["gap_fraction"] > 0


def test_geometry_gate_rejects_filled_parent_hole():
    shapely = pytest.importorskip("shapely")
    pytest.importorskip("rasterio")
    original = upxo.raster_polygons(closed_loop_labels("island"))
    assert upxo.geometry_metrics(original, original)["accepted"]
    altered = dict(original)
    altered[1] = shapely.Polygon(original[1].exterior)
    report = upxo.geometry_metrics(original, altered)
    assert not report["accepted"]
    assert report["overlap_fraction"] > 0
    assert report["reference_topology"]["1"]["holes"] == 1
    assert report["actual_topology"]["1"]["holes"] == 0


def test_pixel_seeds_sample_their_own_pixel_including_first_row_and_column():
    from scipy.ndimage import map_coordinates

    labels = np.arange(35, dtype=np.int32).reshape(5, 7)
    seeds = upxo.pixel_seeds(labels.shape)
    sampled = map_coordinates(labels, seeds[:, ::-1].T, order=0, mode="nearest")
    np.testing.assert_array_equal(sampled, labels.ravel())


@pytest.mark.parametrize("smoothing", ["none", "taubin"])
def test_real_worker_retains_island_and_ownership(tmp_path, smoothing):
    if worker.resolve_python(required=False) is None:
        pytest.skip("configure FM_EBSD_PYTHON for the UPXO integration test")
    from festim_microstructure.msh4 import read_msh4

    labels = np.ones((24, 24), dtype=np.int32)
    labels[8:16, 8:16] = 2
    source = tmp_path / "map.npz"
    grain_quats = euler_bunge_to_quat(
        np.array([37, 21]), np.array([52, 43]), np.array([131, 73])
    )
    write_ebsd(source, EbsdData(labels, grain_quats, (1.6, 0.8)))
    base = mesh_grains(source, UpxoMeshOptions(smoothing=smoothing), workdir=tmp_path)
    report = json.loads(Path(f"{base}-validation.json").read_text())
    assert report["accepted"]
    assert report["export"]["accepted"]
    assert report["export"]["triangle_physical_ids_preserved"]
    assert report["export"]["boundary_physical_ids_preserved"]
    assert report["geometry"]["actual_topology"] == {
        "1": {"parts": 1, "holes": 1},
        "2": {"parts": 1, "holes": 0},
    }
    xyz, lines, triangles = read_msh4(base.with_suffix(".msh4"))
    assert {g for g, _ in triangles} == {1, 2}
    assert {tag for tag, _ in lines} == {1, 2}
    np.testing.assert_allclose(
        np.max(list(xyz.values()), axis=0)[:2], [38.4e-6, 19.2e-6]
    )
    np.testing.assert_allclose(read_ebsd(f"{base}-ebsd.npz").grain_quats, grain_quats)


@pytest.mark.parametrize(
    ("case", "expected_pairs"),
    [
        ("island", [(1, 0), (1, 2)]),
        ("nested_islands", [(1, 0), (1, 2), (2, 3)]),
        ("disconnected", [(1, 0), (1, 2), (1, 2)]),
        ("void", [(1, 0), (1, 0)]),
    ],
)
def test_boundary_arcs_keep_disconnected_closed_loops(
    case, expected_pairs, pixel_triangles
):
    triangles, owners = pixel_triangles(closed_loop_labels(case))
    arcs, nonmanifold = upxo.boundary_arcs(triangles, owners)
    assert nonmanifold == 0
    assert sorted(pair for pair, _edges in arcs) == sorted(expected_pairs)
    for _pair, edges in arcs:
        nodes, counts = np.unique(np.asarray(edges), return_counts=True)
        assert len(nodes) > 0
        np.testing.assert_array_equal(counts, 2)


def test_boundary_arcs_split_at_triple_junction(pixel_triangles):
    labels = np.ones((24, 24), dtype=np.int32)
    labels[:12, 12:] = 2
    labels[12:] = 3
    triangles, owners = pixel_triangles(labels)
    arcs, nonmanifold = upxo.boundary_arcs(triangles, owners)
    interior = [(pair, edges) for pair, edges in arcs if 0 not in pair]
    assert nonmanifold == 0
    assert sorted(pair for pair, _edges in interior) == [(1, 2), (1, 3), (2, 3)]
    endpoints = []
    for _pair, edges in interior:
        nodes, counts = np.unique(np.asarray(edges), return_counts=True)
        endpoints.append(set(nodes[counts == 1]))
    assert len(set.intersection(*endpoints)) == 1


@pytest.mark.parametrize(
    "case,expected",
    [
        ("island", {"1": {"parts": 1, "holes": 1}, "2": {"parts": 1, "holes": 0}}),
        (
            "disconnected",
            {"1": {"parts": 1, "holes": 2}, "2": {"parts": 2, "holes": 0}},
        ),
        ("void", {"1": {"parts": 1, "holes": 1}}),
    ],
)
def test_triangle_topology_counts_islands_voids_and_disconnected_parts(
    case, expected, pixel_triangles
):
    triangles, owners = pixel_triangles(closed_loop_labels(case))
    assert upxo.mesh_topology(triangles, owners) == expected


@pytest.mark.parametrize("smoothing", ["none", "taubin"])
def test_real_worker_void_preservation(tmp_path, smoothing):
    """Exact polygons preserve voids; moving their boundary remains a rejection."""
    if worker.resolve_python(required=False) is None:
        pytest.skip("UPXO interpreter unavailable")
    labels = np.ones((24, 24), dtype=np.int32)
    labels[8:16, 8:16] = 0
    source = tmp_path / "void.npz"
    write_ebsd(source, EbsdData(labels, np.array([[1.0, 0, 0, 0]]), (1, 1)))
    options = UpxoMeshOptions(smoothing=smoothing)
    if smoothing == "taubin":
        with pytest.raises(RuntimeError, match="UPXO meshing failed"):
            mesh_grains(source, options, workdir=tmp_path)
        report = json.loads((tmp_path / "poly-failed-validation.json").read_text())
        assert not report["geometry"]["accepted"]
        assert not (tmp_path / "poly.msh4").exists()
    else:
        base = mesh_grains(source, options, workdir=tmp_path)
        report = json.loads(Path(f"{base}-validation.json").read_text())
        assert report["accepted"]
        assert report["mesh"]["actual_topology"] == {"1": {"parts": 1, "holes": 1}}


def test_real_worker_thin_grain_protection_preserves_labels(tmp_path):
    if worker.resolve_python(required=False) is None:
        pytest.skip("UPXO interpreter unavailable")
    labels = np.ones((24, 24), dtype=np.int32)
    labels[:, 11:12] = 2
    source = tmp_path / "thin.npz"
    write_ebsd(source, EbsdData(labels, np.array([[1.0, 0, 0, 0]] * 2), (1, 1)))
    base = mesh_grains(source, UpxoMeshOptions(thin_grain_px=1.5), workdir=tmp_path)
    report = json.loads(Path(f"{base}-validation.json").read_text())
    assert report["accepted"]
    np.testing.assert_array_equal(read_ebsd(f"{base}-ebsd.npz").labels, labels)
    assert report["geometry"]["actual_topology"] == {
        "1": {"parts": 2, "holes": 0},
        "2": {"parts": 1, "holes": 0},
    }


@pytest.mark.parametrize(
    "version,revision,defdap,error",
    [
        ("1.2.0", upxo.UPXO_REVISION, "0.93.6", f"UPXO {upxo.UPXO_VERSION}"),
        (upxo.UPXO_VERSION, "wrong", "0.93.6", "evaluated commit"),
        (upxo.UPXO_VERSION, upxo.UPXO_REVISION, "0.93.5", "DefDAP 0.93.6"),
        (upxo.UPXO_VERSION, upxo.UPXO_REVISION, "0.93.6", None),
    ],
)
def test_shared_worker_compatibility(monkeypatch, version, revision, defdap, error):
    from types import SimpleNamespace

    monkeypatch.setattr(
        upxo.importlib.metadata,
        "version",
        lambda name: version if name == "upxo" else defdap,
    )
    monkeypatch.setattr(
        upxo.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            read_text=lambda name: json.dumps({"vcs_info": {"commit_id": revision}})
        ),
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            upxo._check_compatibility(require_defdap=True)
    else:
        upxo._check_compatibility(require_defdap=True)


def test_environment_fingerprint_detects_same_version_reinstall(monkeypatch):
    from types import SimpleNamespace

    record = {"RECORD": "first build", "direct_url.json": None}
    monkeypatch.setattr(upxo, "_check_compatibility", lambda **kw: None)
    monkeypatch.setattr(upxo.sys, "version_info", (3, 13))
    monkeypatch.setattr(upxo.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(
        upxo.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(
                metadata={"Name": "upxo"},
                version=upxo.UPXO_VERSION,
                read_text=lambda name: record[name],
            )
        ],
    )
    first = upxo._environment_report()
    record["RECORD"] = "same version, different build"
    assert first != upxo._environment_report()
