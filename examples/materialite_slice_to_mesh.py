"""Crop a processed Materialite x-z section and inspect native EBSD meshes.

Run in the FESTIM-Microstructure environment, with no Materialite dependency:
    python examples/materialite_slice_to_mesh.py --source-slice /path/to/slice_xz.npz

The twenty-layer crops are meshed exactly and with controlled Taubin smoothing.
No hydrogen-transport problem is constructed or solved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import ndimage

import festim_microstructure as fm
from festim_microstructure.formats.ebsd import EbsdData, read_ebsd, write_ebsd
from festim_microstructure.formats.msh4 import read_mesh, read_msh4
from festim_microstructure.meshing.ebsd import (
    mesh_diagnostics,
    read_extent,
    write_network_png,
)
from festim_microstructure.plotting import use_agg

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
# Half-open array windows: row_start, row_stop, column_start, column_stop.
CROPS = {"balanced": (36, 64, 14, 53), "tall": (90, 192, 59, 72)}
SECTION_BASIS = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_crop(source, window):
    """Return native data and an auditable, possibly one-to-many ID mapping.

    Orientations already use (x,z,-y). Cropping is a translation, not a rotation.
    """
    if not bool(source["simulation_complete"]):
        raise ValueError("The source simulation is incomplete")
    if int(source["completed_steps"]) != int(source["requested_steps"]):
        raise ValueError("The source did not complete every requested step")
    np.testing.assert_array_equal(source["section_basis"], SECTION_BASIS)
    spacing = np.asarray(source["spacing_m"], dtype=float)
    np.testing.assert_allclose(source["vox"], spacing / 1e-6, rtol=1e-12, atol=0)
    if float(source["mesh_unit"]) != 1e-6:
        raise ValueError("The native source spacing must be in micrometres")
    labels = np.asarray(source["labels"])
    z0, z1, x0, x1 = window
    if not (0 <= z0 < z1 <= labels.shape[0] and 0 <= x0 < x1 <= labels.shape[1]):
        raise ValueError("The crop window is outside the source section")
    selected = (slice(z0, z1), slice(x0, x1))
    temperature = source["temperature_K"][selected]
    if not (
        np.all(source["indexed"][selected])
        and np.all(source["ever_melted"][selected])
        and np.all(source["deposition_mask"][selected])
        and np.all(source["spin_xz"][selected] > 0)
        and np.all(labels[selected] > 0)
        and np.isfinite(temperature).all()
        and np.all(temperature < 1928.0)
    ):
        raise ValueError("The crop must contain only processed solid deposition pixels")
    parent = labels[selected]
    cropped = np.zeros(parent.shape, dtype=np.int32)
    parent_by_new = [0]
    for parent_id in np.unique(parent):
        pieces, count = ndimage.label(
            parent == parent_id, structure=ndimage.generate_binary_structure(2, 1)
        )
        keep = pieces > 0
        cropped[keep] = pieces[keep] + len(parent_by_new) - 1
        parent_by_new.extend([int(parent_id)] * count)
    parent_by_new = np.asarray(parent_by_new, dtype=np.int32)
    inherited = np.asarray(source["grain_quats"])[parent_by_new[1:] - 1].copy()
    pixels = np.asarray(source["pixel_quats"])[selected].copy()
    data = EbsdData(
        labels=cropped,
        grain_quats=inherited,
        vox=tuple(source["vox"]),
        pixel_quats=pixels,
        indexed=np.ones(cropped.shape, dtype=bool),
        source_labels=cropped.copy(),
        crysym="cubic",
    )
    np.testing.assert_array_equal(parent_by_new[cropped], parent)
    np.testing.assert_array_equal(data.pixel_quats, source["pixel_quats"][selected])
    dot = np.abs(np.sum(data.grain_quats[cropped - 1] * pixels, axis=-1))
    np.testing.assert_allclose(dot, 1.0, atol=1e-12)
    spins = np.asarray(source["source_spin_by_grain"])[parent_by_new]
    np.testing.assert_array_equal(spins[cropped], source["spin_xz"][selected])
    offset = np.asarray(source["voxel_corner_origin_m"]) + np.array([x0, z0]) * spacing
    provenance = {
        "window_rows_z_columns_x": [z0, z1, x0, x1],
        "shape_rows_z_columns_x": list(cropped.shape),
        "section_frame": "(x,z,-y); existing orientations inherited without rotation",
        "section_basis": source["section_basis"].tolist(),
        "source_y_coordinate_m": float(source["y_coordinate_m"]),
        "source_corner_xz_m": offset.tolist(),
        "mesh_local_corner_m": [0.0, 0.0],
        "extent_m": (np.asarray(data.extent) * 1e-6).tolist(),
        "vox_um": list(data.vox),
        "mesh_unit": 1e-6,
        "n_parent_grains": int(np.unique(parent).size),
        "n_grain_pieces": data.ncells,
        "mapping": [
            {
                "grain_id": i,
                "parent_grain_id": int(p),
                "materialite_spin": int(spins[i]),
            }
            for i, p in enumerate(parent_by_new[1:], start=1)
        ],
    }
    return data, provenance


def mesh_arrays(path):
    """Read actual exported triangle/curve tags for checks and plotting."""
    vertices, segments, tagged_triangles = read_msh4(path)
    points = np.asarray([vertices[i][:2] for i in range(len(vertices))])
    triangles = np.asarray([nodes for _, nodes in tagged_triangles], dtype=np.int32)
    owners = np.asarray([tag for tag, _ in tagged_triangles], dtype=np.int32)
    lines = np.asarray([nodes for _, nodes in segments], dtype=np.int32)
    return points, triangles, owners, lines


def triangle_quality(points, triangles):
    """Area and scale-independent shape quality; 1 is equilateral."""
    p = points[triangles]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    if not len(area) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("The mesh contains no triangles or has degenerate triangles")
    denominator = np.sum(a * a, axis=1) + np.sum(b * b, axis=1)
    denominator += np.sum((a - b) ** 2, axis=1)
    quality = 4 * np.sqrt(3) * area / denominator
    return {
        "minimum_triangle_area_m2": float(area.min()),
        "minimum_triangle_quality": float(quality.min()),
        "median_triangle_quality": float(np.median(quality)),
        "total_area_m2": float(area.sum()),
    }


def mesh_panel(ax, data, arrays=None, zoom=False):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.tri import Triangulation

    cmap = ListedColormap(np.random.default_rng(42).random((data.ncells, 3)))
    width, height = data.extent
    if arrays is None:
        ax.imshow(
            data.labels,
            origin="lower",
            extent=(0, width, 0, height),
            cmap=cmap,
            vmin=1,
            vmax=data.ncells,
            interpolation="nearest",
        )
    else:
        points, triangles, owners, lines = arrays
        xy = points / 1e-6
        tri = Triangulation(xy[:, 0], xy[:, 1], triangles)
        ax.tripcolor(tri, facecolors=owners, cmap=cmap, vmin=1, vmax=data.ncells)
        ax.triplot(tri, color="0.35", linewidth=0.25)
        ax.add_collection(LineCollection(xy[lines], colors="black", linewidths=0.7))
    if zoom:
        ax.set_xlim(width / 2 - 25, width / 2 + 25)
        ax.set_ylim(height / 2 - 25, height / 2 + 25)
    else:
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm, crop-local)", fontsize=9)
    ax.set_ylabel("z / build direction (µm, crop-local)", fontsize=9)
    ax.tick_params(labelsize=8)


def plot_selection(source, output):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    plt = use_agg()
    labels = source["labels"]
    colors = np.vstack(
        ([0.85, 0.85, 0.85], np.random.default_rng(42).random((int(labels.max()), 3)))
    )
    spacing = source["spacing_m"] / 1e-6
    corner = source["voxel_corner_origin_m"] / 1e-6
    extent = [
        corner[0],
        corner[0] + labels.shape[1] * spacing[0],
        corner[1],
        corner[1] + labels.shape[0] * spacing[1],
    ]
    fig, ax = plt.subplots(figsize=(6, 10), constrained_layout=True)
    ax.imshow(
        labels,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(colors),
        interpolation="nearest",
        vmin=0,
        vmax=labels.max(),
    )
    for (name, (z0, z1, x0, x1)), color in zip(
        CROPS.items(), ("black", "red"), strict=True
    ):
        xy = corner + np.array([x0, z0]) * spacing
        ax.add_patch(
            Rectangle(
                xy,
                (x1 - x0) * spacing[0],
                (z1 - z0) * spacing[1],
                fill=False,
                edgecolor=color,
                linewidth=2.5,
            )
        )
        ax.text(
            xy[0],
            xy[1] + (z1 - z0) * spacing[1],
            name,
            color=color,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    ax.set(
        title="Fully processed crops in the twenty-layer section",
        xlabel="Original x (µm)",
        ylabel="Original z / build direction (µm)",
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mesh(data, arrays, path, title):
    plt = use_agg()
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
    for ax, zoom in zip(axes, (False, True), strict=True):
        mesh_panel(ax, data, arrays, zoom=zoom)
        ax.set_title(
            "Full mesh" if not zoom else "Central 50 x 50 µm zoom", fontsize=11
        )
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_comparison(data, cases, output, crop_name):
    plt = use_agg()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 10),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    selected = [None, cases.get("exact"), cases.get("smoothed")]
    for col, (name, case) in enumerate(
        zip(("Source crop", "Exact mesh", "Smoothed mesh"), selected, strict=True)
    ):
        accepted = (case is None and col == 0) or (
            case and case["status"] == "accepted"
        )
        arrays = mesh_arrays(case["mesh_path"]) if case and accepted else None
        for row in (0, 1):
            ax = axes[row, col]
            if accepted:
                mesh_panel(ax, data, arrays, zoom=bool(row))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No accepted mesh\nSee attempt reports/logs",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            title = "Central 50 x 50 µm zoom" if row else name
            if not row and case and case["mode"] == "smoothed_gentle":
                title += " (1 iteration)"
            ax.set_title(title, fontsize=11)
    fig.suptitle(
        f"{crop_name}: {data.ncells} grain pieces; source colours retained",
        fontsize=12,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def run_case(data, archive, directory, mode, deadline):
    started = time.monotonic()
    remaining = deadline - started
    if remaining <= 0:
        return {"mode": mode, "status": "skipped", "reason": "Meshing budget exhausted"}
    options = fm.UpxoMeshOptions(
        smoothing="none" if mode == "exact" else "taubin",
        smooth_iter=1 if mode == "smoothed_gentle" else 5,
        thin_grain_px=1.5 if mode == "smoothed_gentle" else 0.0,
        mesh_size_gb=0.75,
        mesh_size_bulk=4.5,
        n_threads=1,
        timeout=min(600.0, remaining),
    )
    report = {"mode": mode, "status": "running", "options": asdict(options)}
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / "poly"
    print(
        f"Meshing {directory.parent.name}/{mode}: {data.ncells} grain pieces",
        flush=True,
    )
    try:
        base = fm.meshing.upxo.mesh_ebsd(
            archive, options, workdir=directory, stem="poly", unit=1e-6, force=False
        )
        validation = json.loads(Path(f"{base}-validation.json").read_text())
        if not validation["accepted"]:
            raise RuntimeError("The existing mesher rejected this geometry")
        change = mesh_diagnostics(base, check_images=True)
        if mode == "exact":
            np.testing.assert_allclose(
                change.a_mesh, change.a_raster, rtol=1e-8, atol=1e-10
            )
        mesh, cells, facets = read_mesh(base, gdim=2)
        if mesh.comm.size != 1:
            raise ValueError("Run this inspection example in serial")
        np.testing.assert_array_equal(
            np.unique(cells.values), np.arange(1, data.ncells + 1)
        )
        np.testing.assert_allclose(mesh.geometry.x.min(axis=0)[:2], 0, atol=1e-14)
        np.testing.assert_allclose(
            mesh.geometry.x.max(axis=0)[:2],
            np.asarray(data.extent) * 1e-6,
            rtol=1e-10,
            atol=1e-14,
        )
        micro = fm.EbsdMicrostructure.from_mesh(
            base, mesh, cells, facets, read_extent(base), theta_min=10.0
        )
        micro.check_orientations()
        np.testing.assert_array_equal(micro.grain_quats, data.grain_quats)
        np.testing.assert_array_equal(read_ebsd(f"{base}-ebsd.npz").labels, data.labels)
        write_network_png(base, mesh, micro, f"{base}-ebsd.npz")
        arrays = mesh_arrays(base.with_suffix(".msh4"))
        quality = triangle_quality(arrays[0], arrays[1])
        report.update(
            status="accepted",
            mesh_path=str(base.with_suffix(".msh4")),
            nodes=len(arrays[0]),
            triangles=len(arrays[1]),
            n_grains=micro.n_grains,
            n_boundaries=len(micro.edges),
            n_interior_boundaries=int(micro.interior_mask.sum()),
            triple_junctions=micro.triple_junctions,
            maximum_grain_area_change_percent=float(np.abs(change.delta).max()),
            maximum_boundary_displacement_px=validation["geometry"][
                "max_boundary_displacement_px"
            ],
            dolfinx_loaded=True,
            orientations_preserved=True,
            boundary_tags_preserved=True,
            quality=quality,
        )
        plot_mesh(
            data,
            arrays,
            directory / "mesh.png",
            f"{directory.parent.name}: {mode}, {report['triangles']} triangles",
        )
    except (RuntimeError, ValueError, AssertionError) as exc:
        report.update(status="rejected", reason=str(exc))
        failed = Path(f"{base}-failed-validation.json")
        if failed.is_file():
            report["failed_validation"] = json.loads(failed.read_text())
    report["runtime_s"] = time.monotonic() - started
    save_json(directory / "attempt.json", report)
    print(f"  {mode}: {report['status']} ({report['runtime_s']:.1f} s)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-slice", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--budget-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("The computation budget must be finite and positive")
    args.source_slice = args.source_slice.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.source_slice, allow_pickle=False) as arrays:
        source = dict(arrays)
    started = time.monotonic()
    deadline = started + args.budget_seconds
    source_hash = hashlib.sha256(args.source_slice.read_bytes()).hexdigest()
    data_by_crop, archives, reports = {}, {}, {}
    for name, window in CROPS.items():
        data, provenance = prepare_crop(source, window)
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        archive = write_ebsd(directory / "crop.npz", data)
        back = read_ebsd(archive)
        for key in (
            "labels",
            "grain_quats",
            "vox",
            "pixel_quats",
            "indexed",
            "source_labels",
        ):
            np.testing.assert_array_equal(getattr(back, key), getattr(data, key))
        provenance.update(source=str(args.source_slice), source_sha256=source_hash)
        save_json(directory / "crop-provenance.json", provenance)
        data_by_crop[name], archives[name] = data, archive
        reports[name] = {"crop": provenance, "attempts": []}
    plot_selection(source, args.output / "crop_selection.png")
    for name in CROPS:
        attempt = run_case(
            data_by_crop[name],
            archives[name],
            args.output / name / "exact",
            "exact",
            deadline,
        )
        reports[name]["attempts"].append(attempt)
    for name in CROPS:
        attempt = run_case(
            data_by_crop[name],
            archives[name],
            args.output / name / "smoothed_default",
            "smoothed_default",
            deadline,
        )
        reports[name]["attempts"].append(attempt)
        if attempt["status"] == "rejected":
            attempt = run_case(
                data_by_crop[name],
                archives[name],
                args.output / name / "smoothed_gentle",
                "smoothed_gentle",
                deadline,
            )
            reports[name]["attempts"].append(attempt)
    for name, report in reports.items():
        exact = next(a for a in report["attempts"] if a["mode"] == "exact")
        smoothed = next(
            (
                a
                for a in report["attempts"]
                if a["mode"] != "exact" and a["status"] == "accepted"
            ),
            None,
        )
        plot_comparison(
            data_by_crop[name],
            {"exact": exact, "smoothed": smoothed},
            args.output / name / "comparison.png",
            name,
        )
    rows = []
    for name, report in reports.items():
        for attempt in report["attempts"]:
            rows.append(
                {
                    "crop": name,
                    **{
                        key: attempt.get(key)
                        for key in (
                            "mode",
                            "status",
                            "nodes",
                            "triangles",
                            "n_grains",
                            "runtime_s",
                            "maximum_grain_area_change_percent",
                            "maximum_boundary_displacement_px",
                        )
                    },
                }
            )
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        args.output / "summary.json",
        {
            "source_sha256": source_hash,
            "runtime_s": time.monotonic() - started,
            "budget_seconds": args.budget_seconds,
            "hydrogen_transport_run": False,
            "crops": reports,
        },
    )
    print(f"Mesh inspection results: {args.output}", flush=True)
    complete = all(
        a["status"] != "skipped" for r in reports.values() for a in r["attempts"]
    )
    exact_ok = all(
        next(a for a in r["attempts"] if a["mode"] == "exact")["status"] == "accepted"
        for r in reports.values()
    )
    return 0 if complete and exact_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
