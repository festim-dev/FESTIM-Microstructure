"""Direct subprocess entry point for UPXO EBSD import and meshing."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

UPXO_VERSION = "1.3.1"
DEFDAP_VERSION = "0.93.6"
UPXO_REVISION = "a53885ef0a7a06f6b3fb195747363ae0a19bb127"
_PROTOCOL = 4


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _environment_report(*, health=False):
    """Fingerprint installed distribution metadata and test EBSD runtime imports."""
    if sys.version_info < (3, 13):
        raise RuntimeError("UPXO requires Python 3.13 or newer")
    _check_compatibility(require_defdap=True)
    required = (
        "upxo",
        "defdap",
        "numpy",
        "scipy",
        "shapely",
        "gmsh",
        "rasterio",
        "meshio",
        "pyvista",
    )
    packages = {name: importlib.metadata.version(name) for name in required}
    distributions = []
    for dist in importlib.metadata.distributions():
        distributions.append(
            {
                "name": dist.metadata["Name"],
                "version": dist.version,
                "origin": dist.read_text("direct_url.json"),
                "record_sha256": hashlib.sha256(
                    (dist.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }
        )
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "packages": packages,
        "distributions": sorted(distributions, key=lambda d: (d["name"], d["version"])),
    }
    if health:
        apis = {
            "upxo.interfaces.defdap.ebsd_reader": ("EBSDReader",),
            "upxo.meshing.conformal_mesher2d": ("confMesh2dGMSH",),
            "upxo.meshing.gsmesh2d": ("_flatten_cells",),
            "upxo.pxtalops.gssmooth2d": ("smooth_gs_slice",),
            "rasterio.features": ("shapes",),
            "shapely.geometry": ("Polygon",),
            "meshio": ("write",),
            "gmsh": ("initialize", "finalize"),
        }
        for module, attributes in apis.items():
            imported = importlib.import_module(module)
            for attribute in attributes:
                getattr(imported, attribute)
        report["health"] = "ok"
    return report


def raster_polygons(labels):
    """Polygonize exact pixels, retaining holes and shared edge subdivisions.

    Rasterio may simplify a straight edge differently on opposite sides of
    a junction. Restore each integer grid vertex so UPXO receives identical
    segment endpoints on both sides. No pixel labels or topology are repaired.
    """
    from rasterio.features import shapes
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    def dense_ring(ring):
        result = []
        coords = np.asarray(ring.coords)
        for a, b in pairwise(coords):
            n = round(np.abs(b - a).max())
            result.extend(a + (b - a) * k / n for k in range(n))
        return result

    grouped = defaultdict(list)
    for geometry, grain in shapes(labels, mask=labels > 0, connectivity=4):
        poly = shape(geometry)
        grouped[int(grain)].append(
            Polygon(
                dense_ring(poly.exterior),
                [dense_ring(r) for r in poly.interiors],
            )
        )
    return {gid: unary_union(parts) for gid, parts in grouped.items()}


def polygon_parts(geometry):
    return [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)


def pixel_seeds(shape):
    """Seeds inside each pixel that UPXO's nearest-index lookup samples correctly."""
    yy, xx = np.indices(shape, dtype=float)
    # nextafter(0.5, 0) is too close: ndimage's nearest rounding still picks 1.
    return np.column_stack([xx.ravel(), yy.ravel()]) + 0.5 - 1e-6


def geometry_metrics(reference, cells, max_area_change=0.05, max_displacement=1.0):
    """Check the partition against the original pixels, not just the mesh input."""
    from shapely.ops import unary_union

    expected = unary_union(list(reference.values()))
    actual = unary_union(list(cells.values()))
    scale = expected.area
    missing = sorted(set(reference) - set(cells))
    extra = sorted(set(cells) - set(reference))
    area_errors, displaced = {}, {}
    for gid in set(reference) & set(cells):
        original, new = reference[gid], cells[gid]
        area_errors[gid] = abs(new.area - original.area) / original.area
        displaced[gid] = original.symmetric_difference(new).area

    def topology(polygons):
        return {
            str(gid): {
                "parts": len(polygon_parts(geom)),
                "holes": sum(len(part.interiors) for part in polygon_parts(geom)),
            }
            for gid, geom in polygons.items()
        }

    def neighbors(polygons):
        gids = sorted(polygons)
        return {
            (a, b)
            for i, a in enumerate(gids)
            for b in gids[i + 1 :]
            if polygons[a].boundary.intersection(polygons[b].boundary).length > 1e-7
        }

    expected_pairs, actual_pairs = neighbors(reference), neighbors(cells)
    gap = expected.difference(actual).area / scale
    excess = actual.difference(expected).area / scale
    overlap = max(0.0, sum(g.area for g in cells.values()) - actual.area) / scale
    topo_reference, topo_actual = topology(reference), topology(cells)
    max_area = max(area_errors.values(), default=0.0)
    max_boundary_displacement = max(
        (
            reference[g].boundary.hausdorff_distance(cells[g].boundary)
            for g in set(reference) & set(cells)
        ),
        default=0.0,
    )
    return {
        "missing_ids": missing,
        "extra_ids": extra,
        "invalid_ids": sorted(g for g, p in cells.items() if not p.is_valid),
        "gap_fraction": gap,
        "excess_fraction": excess,
        "overlap_fraction": overlap,
        "max_grain_area_relative_error": max_area,
        "mean_grain_area_relative_error": float(np.mean(list(area_errors.values()))),
        "label_displacement_fraction": sum(displaced.values()) / (2 * scale),
        "max_boundary_displacement_px": max_boundary_displacement,
        "reference_topology": topo_reference,
        "actual_topology": topo_actual,
        "neighbor_pairs": sorted(expected_pairs),
        "lost_neighbor_pairs": sorted(expected_pairs - actual_pairs),
        "added_neighbor_pairs": sorted(actual_pairs - expected_pairs),
        "accepted": bool(
            not missing
            and not extra
            and all(p.is_valid for p in cells.values())
            and max(gap, excess, overlap) < 1e-7
            and max_area <= max_area_change
            and max_boundary_displacement <= max_displacement + 1e-7
            and topo_reference == topo_actual
            and expected_pairs == actual_pairs
        ),
    }


def boundary_arcs(triangles, owners):
    """Group mesh boundary facets into arcs ending at network junctions.

    Closed loops need no junction. Separate loops between the same grain pair
    get separate IDs; bulk edges inside one grain receive no boundary tag.
    """
    adjacency = defaultdict(list)
    for triangle, grain in zip(triangles, owners, strict=True):
        for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
            adjacency[tuple(sorted((int(a), int(b))))].append(int(grain))
    nonmanifold = sum(len(grains) > 2 for grains in adjacency.values())
    boundary = {
        edge: tuple(sorted(grains)) if len(grains) == 2 else (grains[0], 0)
        for edge, grains in adjacency.items()
        if len(grains) == 1 or (len(grains) == 2 and grains[0] != grains[1])
    }
    incidence = defaultdict(list)
    for edge in boundary:
        for node in edge:
            incidence[node].append(edge)
    unseen = set(boundary)
    arcs = []
    while unseen:
        seed = min(unseen)
        stack, edges = [seed], []
        unseen.remove(seed)
        while stack:
            edge = stack.pop()
            edges.append(edge)
            for node in edge:
                if len(incidence[node]) == 2:
                    for other in incidence[node]:
                        if other in unseen and boundary[other] == boundary[seed]:
                            unseen.remove(other)
                            stack.append(other)
        arcs.append((boundary[seed], sorted(edges)))
    return arcs, nonmanifold


def mesh_topology(triangles, owners):
    """Count components and holes in each grain's actual triangular submesh.

    Components connect across shared facets. Euler counts use separate vertices
    for separate components, preserving corner-only contacts between parts.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(triangles)
    edges = np.sort(
        np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
        ),
        axis=1,
    )
    triangle_ids = np.tile(np.arange(n), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered_edges, ordered_ids = edges[order], triangle_ids[order]
    shared = np.flatnonzero(np.all(ordered_edges[1:] == ordered_edges[:-1], axis=1))
    a, b = ordered_ids[shared], ordered_ids[shared + 1]
    same = owners[a] == owners[b]
    graph = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n, n))
    n_parts, component = connected_components(graph, directed=False)
    vertices = np.unique(
        np.column_stack([np.repeat(component, 3), triangles.ravel()]), axis=0
    )
    component_edges = np.unique(
        np.column_stack([component[triangle_ids], edges]), axis=0
    )
    v = np.bincount(vertices[:, 0], minlength=n_parts)
    e = np.bincount(component_edges[:, 0], minlength=n_parts)
    f = np.bincount(component, minlength=n_parts)
    holes = 1 - (v - e + f)
    part_owner = np.zeros(n_parts, dtype=owners.dtype)
    part_owner[component] = owners
    return {
        str(g): {
            "parts": int((part_owner == g).sum()),
            "holes": int(holes[part_owner == g].sum()),
        }
        for g in np.unique(owners)
    }


def mesh_metrics(mesher, cells):
    triangles = mesher.elConn["triangle"]
    owners = mesher._elem_owners["triangle"]
    xy = mesher.nodes[triangles, :2]
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    area = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    arcs, nonmanifold = boundary_arcs(triangles, owners)
    expected_edges = {e for _, edges in arcs for e in edges}
    actual_edges = {tuple(sorted(map(int, e))) for e in mesher.GBlines}
    unused_curve_segments = len(actual_edges - expected_edges)
    missing_curve_segments = len(expected_edges - actual_edges)
    valid_nodes = np.flatnonzero(np.isfinite(mesher.nodes[:, 0]))
    xy_nodes = mesher.nodes[valid_nodes, :2]
    duplicate_nodes = len(xy_nodes) - len(np.unique(np.round(xy_nodes, 8), axis=0))
    max_area = max(
        abs(area[owners == gid].sum() - geom.area) / geom.area
        for gid, geom in cells.items()
    )
    topology = mesh_topology(triangles, owners)
    expected_topology = {
        str(g): {
            "parts": len(polygon_parts(p)),
            "holes": sum(len(part.interiors) for part in polygon_parts(p)),
        }
        for g, p in cells.items()
    }
    return {
        "nodes": len(valid_nodes),
        "triangles": len(triangles),
        "inverted_or_zero_triangles": int((area <= 0).sum()),
        "nonmanifold_facets": nonmanifold,
        "duplicate_coordinate_nodes": duplicate_nodes,
        "unused_curve_segments": unused_curve_segments,
        "missing_curve_segments": missing_curve_segments,
        "missing_grain_ids": sorted(set(cells) - set(map(int, owners))),
        "max_mesh_polygon_area_relative_error": float(max_area),
        "min_angle_deg": float(
            mesher.quality_report["min_angle_deg"]["triangle"]["min"]
        ),
        "boundary_arcs": len(arcs),
        "actual_topology": topology,
        "accepted": bool(
            np.all(area > 0)
            and not nonmanifold
            and not duplicate_nodes
            and not unused_curve_segments
            and not missing_curve_segments
            and set(map(int, owners)) == set(cells)
            and max_area < 1e-7
            and topology == expected_topology
        ),
    }


def export_metrics(path, points, triangles, owners, lines, edge_ids):
    """Check the exported physical tags against UPXO's exact mesh ownership.

    Match vertices by coordinates, then compare connectivity and physical IDs
    per element. Gmsh may reorder nodes/elements or reverse their orientation.
    Raster overlap plays no part in this check.
    """
    import meshio
    from scipy.spatial import cKDTree

    exported = meshio.read(path, file_format="gmsh")
    distance, node_map = cKDTree(points).query(exported.points)
    tolerance = np.max(np.abs(points)) * 1e-12
    coordinates_preserved = bool(
        len(exported.points) == len(points)
        and np.all(distance <= tolerance)
        and len(np.unique(node_map)) == len(points)
    )
    report = {"coordinates_preserved": coordinates_preserved}

    def canonical_order(connectivity):
        connectivity = np.sort(connectivity, axis=1)
        order = np.lexsort(connectivity.T[::-1])
        return connectivity[order], order

    physical = exported.cell_data_dict.get("gmsh:physical", {})
    for kind, expected, tags, prefix in (
        ("triangle", triangles, owners, "triangle"),
        ("line", lines, edge_ids, "boundary"),
    ):
        actual = exported.get_cells_type(kind)
        expected_nodes, expected_order = canonical_order(np.asarray(expected))
        actual_nodes, actual_order = canonical_order(node_map[actual])
        connectivity_preserved = np.array_equal(expected_nodes, actual_nodes)
        actual_tags = physical.get(kind)
        tags_preserved = bool(
            connectivity_preserved
            and actual_tags is not None
            and np.array_equal(
                np.asarray(tags)[expected_order], actual_tags[actual_order]
            )
        )
        report[f"{prefix}_connectivity_preserved"] = connectivity_preserved
        report[f"{prefix}_physical_ids_preserved"] = tags_preserved
    report["element_types_preserved"] = all(
        block.type in ("line", "triangle") for block in exported.cells
    )
    report["accepted"] = all(report.values())
    return report


def export_mesh(mesher, base, vox, shape, unit):
    """Export UPXO's mesh and verify its physical grain and boundary ownership."""
    import gmsh
    import meshio

    valid = np.flatnonzero(np.isfinite(mesher.nodes[:, 0]))
    index = np.full(len(mesher.nodes), -1, dtype=int)
    index[valid] = np.arange(len(valid))
    points = mesher.nodes[valid].copy()
    points[:, :2] *= np.asarray(vox) * unit
    triangles = index[mesher.elConn["triangle"]]
    owners = mesher._elem_owners["triangle"]
    arcs, _ = boundary_arcs(triangles, owners)
    lines, edge_ids = [], []
    for edge_id, (_pair, edges) in enumerate(arcs, start=1):
        lines.extend(edges)
        edge_ids.extend([edge_id] * len(edges))
    lines, edge_ids = np.asarray(lines), np.asarray(edge_ids)
    field_data = {f"face{g}": np.array([g, 2]) for g in np.unique(owners)}
    field_data.update({f"edge{e}": np.array([e, 1]) for e in np.unique(edge_ids)})
    intermediate = base.with_suffix(".msh")
    meshio.write(
        intermediate,
        meshio.Mesh(
            points,
            [("line", lines), ("triangle", triangles)],
            cell_data={
                "gmsh:physical": [edge_ids, owners],
                "gmsh:geometrical": [edge_ids, owners],
            },
            field_data=field_data,
        ),
        file_format="gmsh22",
        binary=False,
    )
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(intermediate))
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(base.with_suffix(".msh4")))
    finally:
        gmsh.finalize()
    intermediate.unlink()
    ny, nx = shape
    _save_json(
        Path(f"{base}-metadata.json"),
        {
            "version": 1,
            "extent_m": [nx * vox[0] * unit, ny * vox[1] * unit],
            "unit": unit,
            "n_grains": len(np.unique(owners)),
            "crystal_symmetry": "cubic",
        },
    )
    return export_metrics(
        base.with_suffix(".msh4"), points, triangles, owners, lines, edge_ids
    )


def _check_compatibility(*, require_defdap=False):
    """Enforce the same evaluated versions and source revision in both workers."""
    if importlib.metadata.version("upxo") != UPXO_VERSION:
        raise RuntimeError(f"This backend requires evaluated UPXO {UPXO_VERSION}")
    distribution = importlib.metadata.distribution("upxo")
    origin = distribution.read_text("direct_url.json")
    revision = (
        json.loads(origin).get("vcs_info", {}).get("commit_id") if origin else None
    )
    if revision and revision != UPXO_REVISION:
        raise RuntimeError("UPXO is not installed from the evaluated commit")
    if require_defdap and importlib.metadata.version("defdap") != DEFDAP_VERSION:
        raise RuntimeError(f"EBSD import requires DefDAP {DEFDAP_VERSION}")


def _worker(request):
    """Run under the pinned UPXO environment; never import solver libraries."""
    from upxo.meshing.conformal_mesher2d import confMesh2dGMSH
    from upxo.meshing.gsmesh2d import _flatten_cells

    request = Path(request)
    config = json.loads(request.read_text())
    options = config["options"]
    directory, stem = request.parent, config["stem"]
    data = np.load(directory / "input.npz")
    labels, vox = data["labels"], data["vox"]
    report = {
        "accepted": False,
        "upxo_version": importlib.metadata.version("upxo"),
        "options": options,
    }
    distribution = importlib.metadata.distribution("upxo")
    origin = distribution.read_text("direct_url.json")
    source = json.loads(origin) if origin else {}
    revision = source.get("vcs_info", {}).get("commit_id")
    report["upxo_commit"] = revision
    report["packages"] = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "shapely", "gmsh", "rasterio", "meshio")
    }
    validation = directory / f"{stem}-validation.json"
    try:
        _check_compatibility()
        reference = raster_polygons(labels)
        cells = reference
        if options["smoothing"] == "taubin" and options["smooth_iter"]:
            from upxo.pxtalops.gssmooth2d import smooth_gs_slice

            smooth = smooth_gs_slice(
                labels,
                seeds=pixel_seeds(labels.shape),
                area_threshold=0,
                smooth_iter=options["smooth_iter"],
                smooth_lambda=options["smooth_lambda"],
                smooth_mu=options["smooth_mu"],
                method="taubin",
                trim_bounds=(0, 0, labels.shape[1], labels.shape[0]),
                fix_diagonal=False,
                merge_enclosed=False,
                jitter_factor=0,
                thin_grain_px=options["thin_grain_px"],
                close_staircase=False,
            )
            inverse = {new: old for old, new in smooth["old_to_new_gid"].items()}
            cells = {
                inverse[gid]: poly
                for gid, poly in smooth["cells"].items()
                if inverse[gid] > 0
            }
        report["geometry"] = geometry_metrics(
            reference,
            cells,
            options["max_area_change"],
            options["max_boundary_displacement"],
        )
        _save_json(validation, report)
        if not report["geometry"]["accepted"]:
            raise RuntimeError(
                "UPXO reconstruction/smoothing changed the raster beyond the geometry "
                "limits. Inspect the validation report; reduce smoothing or use "
                "UpxoMeshOptions(smoothing='none')."
            )
        flat, gid_map = _flatten_cells(cells)
        mesher = confMesh2dGMSH()
        mesher.femesh_gmsh(
            flat,
            gid_map=gid_map,
            mesh_size_gb=options["mesh_size_gb"],
            mesh_size_bulk=options["mesh_size_bulk"],
            mesh_algo=6,
            mesh_order=1,
            recombine_to_quads=False,
            n_threads=options["n_threads"],
        )
        report["mesh"] = mesh_metrics(mesher, cells)
        arcs, _ = boundary_arcs(
            mesher.elConn["triangle"], mesher._elem_owners["triangle"]
        )
        actual_pairs = {pair for pair, _ in arcs if 0 not in pair}
        expected_pairs = {tuple(p) for p in report["geometry"]["neighbor_pairs"]}
        report["mesh"]["lost_neighbor_pairs"] = sorted(expected_pairs - actual_pairs)
        report["mesh"]["added_neighbor_pairs"] = sorted(actual_pairs - expected_pairs)
        _save_json(validation, report)
        if not report["mesh"]["accepted"] or actual_pairs != expected_pairs:
            raise RuntimeError(
                "UPXO mesh failed conformity, ownership or adjacency checks"
            )
        report["export"] = export_mesh(
            mesher, directory / stem, vox, labels.shape, config["unit"]
        )
        if not report["export"]["accepted"]:
            raise RuntimeError(
                "Exported mesh failed connectivity or physical ownership checks"
            )
        report["accepted"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _save_json(validation, report)


def _sign_aligned_quaternion_mean(equivalents):
    """Average DefDAP symmetry equivalents with antipodal sign alignment.

    ``equivalents`` has shape (n_symmetry, 4, n_pixels), as returned by
    DefDAP's ``Quat.calcSymEqvs``. Keep its sequential symmetry selection, but
    align each selected quaternion with the running sum before adding it.
    DefDAP 0.93.6 selects by absolute dot product without this sign correction,
    which can cancel equivalent orientations and displace the grain mean.
    """
    total = equivalents[0, :, 0].copy()
    for i in range(1, equivalents.shape[2]):
        candidates = equivalents[:, :, i]
        dots = candidates @ total
        best = np.argmax(np.abs(dots))
        total += candidates[best] if dots[best] >= 0 else -candidates[best]
    total /= np.linalg.norm(total)
    return total if total[0] >= 0 else -total


def _import_worker(request):
    """UPXO import and DefDAP symmetry-aware means, without local segmentation."""
    from defdap.quat import Quat
    from upxo.interfaces.defdap.ebsd_reader import EBSDReader

    path = Path(request)
    config = json.loads(path.read_text())
    _check_compatibility(require_defdap=True)
    reader = EBSDReader.load(path.parent / "filtered.ctf")
    try:
        reader.detect_grains(
            min_grain_size=config["min_pixels"], misori_tol=config["threshold"]
        )
    except Exception as exc:
        if str(exc) == "No grains detected.":
            raise ValueError("no grains survived DefDAP grain detection") from exc
        raise
    grains = reader._defdap_map.grainList
    if not grains:
        raise ValueError("no grains survived DefDAP grain detection")
    means = []
    for grain in grains:
        equivalents = Quat.calcSymEqvs(grain.quatList, grain.crystalSym)
        means.append(_sign_aligned_quaternion_mean(equivalents))
    labels = np.asarray(reader.lfi_ebsd, dtype=np.int32)
    if not np.array_equal(np.unique(labels[labels > 0]), np.arange(1, len(means) + 1)):
        raise ValueError("UPXO grain IDs differ from DefDAP orientation row order")
    np.savez(
        path.parent / "import.npz",
        source_labels=labels,
        pixel_quats=reader.quat_ebsd,
        grain_quats=np.asarray(means),
    )
    _save_json(
        path.parent / "packages.json",
        {
            p: importlib.metadata.version(p)
            for p in ("upxo", "defdap", "numpy", "scipy")
        },
    )


if __name__ == "__main__":
    modes = {
        "--worker": _worker,
        "--import": _import_worker,
        "--environment": lambda path: _save_json(Path(path), _environment_report()),
        "--health": lambda path: _save_json(
            Path(path), _environment_report(health=True)
        ),
    }
    if len(sys.argv) != 3 or sys.argv[1] not in modes:
        script = Path(__file__).name
        raise SystemExit(
            f"usage: {script} --worker/--import/--environment/--health PATH"
        )
    modes[sys.argv[1]](sys.argv[2])
