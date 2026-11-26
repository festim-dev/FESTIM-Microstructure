"""A synthetic 3D foil seen through the hexagonal dot array (Delic et al. 2026).

The other ``delic2026_*`` scripts work on a 2D cross-section and a line of
dots. This one builds a 3D foil: a random Voronoi tessellation of the surface,
relaxed towards equal-sized grains and extruded through the thickness, which
is the columnar structure the paper's fractographs show for the 25 and 50 um
foils. The microstructure is invented, not measured: the grain size is the
paper's (Table 1), the boundaries that carry D are drawn at random. It stands
in for the EBSD of the imaged areas until that exists.

The transport model and its parameters are those fitted to the 600 K line
profiles (``delic2026_fit.py``): one species per grain, one codimension-one
network, exchange ``k (c_grain - c_gb)``, diffusion-limited implantation on the
front with nearly closed boundary mouths, a sink on the back. Following the
temperature-series fit, a fraction ``p`` of the boundaries, drawn at random,
is conductive and the rest is not. The back surface is then read as the
experiment reads it: 5 um dots on a hexagonal array of 10 um pitch, each
collecting the lattice flux over its disc and the boundary flux along the
chord of trace it covers, through the fitted indicator response, normalised
per image and thresholded at 10 % of the range.

Three studies:

* ``check``: one planar boundary through a 3D slab is the 2D cross-section
  problem; the back-surface flux and the boundary outflow are compared with
  ``delic2026_crosssection.solve`` on its fine graded mesh.
* ``foils``: the 25 um and 50 um foils at 600 K, as heat maps to set beside
  the paper's Fig. 11(a, b), with the counts of its Tables 2 and 3.

* ``statistics``: the 50 um foil over several random microstructures, for
  the fraction of boundary dots that discolour.

    python examples/delic2026_3d.py check        # ~1 min
    python examples/delic2026_3d.py foils        # ~2 min
    python examples/delic2026_3d.py statistics   # ~3 min
"""

import argparse
import csv
import itertools
import json
import time

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np
import ufl
from delic2026_crosssection import (
    EXPOSURE,
    FOILS,
    GAMMA,
    OUTPUT_DIR,
    R_P,
    REFLECTION,
    UM,
    Case,
    _clustered,
    lattice_diffusivity,
)
from delic2026_crosssection import solve as solve_section
from scipy.spatial import Voronoi

import festim_microstructure as fm
from festim_microstructure.postprocessing import hydrogenography as hg

NETWORK_ID, FRONT_ID, BACK_ID = 10_000, 20_000, 30_000
DELTA = 1e-9  # m, boundary slab width

# The 600 K line-profile fit (fit_summary.json) and the temperature series.
D_RATIO_ACTIVE = 10**5.5  # lam = 316 um
D_RATIO_INACTIVE = 1e2  # lam = 0.1 um: carries nothing
K_EXCHANGE = 1e-4  # m/s
MOUTH_FEED = 0.13
CAPABLE = 0.44
DETECTION, SATURATION = 4.04e19, 7.51e19  # D m^-2


def voronoi_polygons(width, n_grains, rng, relax=3, snap=1.0):
    """Convex grain polygons tiling ``[0, width]^2``, counter-clockwise.

    Seeds are mirrored in the four walls, so every cell is cut off by them,
    and moved to their cell's centroid ``relax`` times, which evens the grain
    sizes out. Vertices are snapped to a grid of ``snap`` so that no edge is
    shorter than the mesh can follow. Lengths in whatever unit ``width`` is.
    """
    points = rng.uniform(0, width, (n_grains, 2))
    for sweep in range(relax + 1):
        mirrored = [points]
        for axis in (0, 1):
            for wall in (0.0, width):
                image = points.copy()
                image[:, axis] = 2 * wall - image[:, axis]
                mirrored.append(image)
        voronoi = Voronoi(np.vstack(mirrored))
        polygons = [
            voronoi.vertices[voronoi.regions[voronoi.point_region[i]]]
            for i in range(n_grains)
        ]
        if sweep < relax:
            points = np.array([polygon.mean(axis=0) for polygon in polygons])
    out = []
    for polygon in polygons:
        q = np.round(np.clip(polygon, 0, width) / snap) * snap
        q = q[np.any(q != np.roll(q, 1, axis=0), axis=1)]
        x, y = q[:, 0], q[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        out.append(q if area > 0 else q[::-1])
    return out


def strip_polygons(width, depth, boundaries):
    """Rectangles: planar boundaries at the abscissae ``boundaries``."""
    xs = [0.0, *boundaries, width]
    return [
        np.array([[a, 0.0], [b, 0.0], [b, depth], [a, depth]])
        for a, b in itertools.pairwise(xs)
    ]


def columnar_slab(polygons, thickness, n_layers, h_gb, h_bulk, grade=8.0):
    """Extrude the polygons into a tetrahedral slab; lengths in micrometres.

    Every grain is a tagged volume and every boundary plane, the wall between
    two grains, its own tagged surface, numbered from 1. The layers are graded
    towards both surfaces and the triangles towards the boundaries (``h_gb``
    on them, ``h_bulk`` beyond ``grade``). Returns the microstructure, in
    metres, and the boundary traces as an ``(n, 2, 2)`` array in entity order.
    """
    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    geo = gmsh.model.geo
    points, lines, surfaces = {}, {}, []

    def point(xy):
        """One gmsh point per vertex, shared by the polygons that meet there."""
        key = (round(xy[0] * 1e3), round(xy[1] * 1e3))
        if key not in points:
            points[key] = geo.addPoint(xy[0], xy[1], 0.0)
        return points[key]

    for polygon in polygons:
        tags = [point(vertex) for vertex in polygon]
        loop = []
        for a, b in zip(tags, tags[1:] + tags[:1], strict=True):
            key = (min(a, b), max(a, b))
            if key not in lines:
                lines[key] = geo.addLine(*key)
            loop.append(lines[key] if a < b else -lines[key])
        surfaces.append(geo.addPlaneSurface([geo.addCurveLoop(loop)]))
    heights = _clustered(n_layers, 0.9)[1:]
    extruded = geo.extrude(
        [(2, s) for s in surfaces],
        0,
        0,
        thickness,
        numElements=[1] * n_layers,
        heights=list(heights),
        recombine=False,
    )
    geo.synchronize()
    for i, volume in enumerate(tag for dim, tag in extruded if dim == 3):
        gmsh.model.addPhysicalGroup(3, [volume], i + 1)
    traces = []
    for _, surface in gmsh.model.getEntities(2):
        if len(gmsh.model.getAdjacencies(2, surface)[0]) != 2:
            continue
        gmsh.model.addPhysicalGroup(2, [surface], len(traces) + 1)
        corners = gmsh.model.getBoundary([(2, surface)], recursive=True)
        xy = np.array([gmsh.model.getValue(0, tag, []) for _, tag in corners])[:, :2]
        traces.append(np.unique(np.round(xy, 9), axis=0))
    walls = [
        line
        for line in lines.values()
        if len(gmsh.model.getAdjacencies(1, line)[0]) == 2
    ]
    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "CurvesList", walls)
    field.setNumber(1, "Sampling", 60)
    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", h_gb)
    field.setNumber(2, "SizeMax", h_bulk)
    field.setNumber(2, "DistMin", 0.0)
    field.setNumber(2, "DistMax", grade)
    field.setAsBackgroundMesh(2)
    for option in ("FromPoints", "FromCurvature", "ExtendFromBoundary"):
        gmsh.option.setNumber(f"Mesh.MeshSize{option}", 0)
    gmsh.model.mesh.generate(3)
    data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    data.mesh.geometry.x[:] *= UM
    traces = np.array(traces) * UM
    micro = fm.TaggedMicrostructure(
        mesh=data.mesh,
        cell_tags=data.cell_tags,
        facet_tags=data.facet_tags,
        gb_tag=list(range(1, len(traces) + 1)),
        tolerance=1e-3 * h_gb * UM,
        name=f"columnar slab, {len(polygons)} grains, {len(traces)} boundaries",
    )
    return micro, traces


def plane_flux(species, grains, c_gb, network, D, D_gb, traces, z, tol):
    """What crosses the mesh plane at height ``z``, along ``+z``.

    Lattice: the triangles of the plane (``(n, 3, 2)``) and the flux through
    each, read from the cell below it, where P1 gradients are constant.
    Network: the edges of the boundary planes in that plane (``(n, 2, 2)``)
    and the flux per unit length of each, ``delta D_gb dc/dz``, with the
    diffusivity of the boundary the edge lies on.
    """
    comm = MPI.COMM_WORLD
    triangles, lattice = [], []
    for spe, grain in zip(species, grains, strict=True):
        u = spe.subdomain_to_post_processing_solution[grain]
        tri, flux = _below(u, 2, z, tol)
        triangles.append(tri)
        lattice.append(D * flux)
    u = c_gb.subdomain_to_post_processing_solution[network]
    edges, gradient = _below(u, 1, z, tol)
    triangles = np.concatenate(comm.allgather(np.concatenate(triangles)))
    lattice = np.concatenate(comm.allgather(np.concatenate(lattice)))
    edges = np.concatenate(comm.allgather(edges))
    gradient = np.concatenate(comm.allgather(gradient))
    # Which boundary each edge lies on: the trace its midpoint is nearest to.
    mid = edges.mean(axis=1)
    a, b = traces[:, 0], traces[:, 1]
    t = np.einsum("esk,sk->es", mid[:, None, :] - a[None], b - a)
    t = np.clip(t / np.einsum("sk,sk->s", b - a, b - a)[None], 0.0, 1.0)
    nearest = np.linalg.norm(
        mid[:, None, :] - (a[None] + t[..., None] * (b - a)[None]), axis=2
    ).argmin(axis=1)
    return triangles, lattice, edges, DELTA * D_gb[nearest] * gradient, nearest


def _below(u, dim, z, tol):
    """Entities of dimension ``dim`` in the plane ``z`` and ``-du/dz`` below them."""
    sub = u.function_space.mesh
    tdim = sub.topology.dim
    sub.topology.create_connectivity(dim, tdim)
    entities = dolfinx.mesh.locate_entities(sub, dim, lambda x: np.abs(x[2] - z) < tol)
    entities = entities[entities < sub.topology.index_map(dim).size_local]
    if entities.size == 0:
        return np.empty((0, dim + 1, 2)), np.empty(0)
    e2c = sub.topology.connectivity(dim, tdim)
    cells = []
    for entity in entities:
        candidates = e2c.links(entity)
        height = dolfinx.mesh.compute_midpoints(sub, tdim, candidates)[:, 2]
        cells.append(candidates[np.argmin(height)])
    cells = np.array(cells, dtype=np.int32)
    centre = np.full((1, tdim), 1 / (tdim + 1))
    expr = dolfinx.fem.Expression(-ufl.grad(u)[2], centre)
    values = expr.eval(sub, cells).reshape(-1)
    nodes = dolfinx.mesh.entities_to_geometry(sub, dim, entities)
    return sub.geometry.x[nodes][:, :, :2], values


def area(triangles):
    """Areas of ``(n, 3, 2)`` triangles."""
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def solve(polygons, thickness, T, ratios, k, m, n_layers, h_gb, h_bulk):
    """Steady state of one slab; lengths of the geometry in micrometres.

    ``ratios`` is ``D_gb / D`` per boundary, or a function of the number of
    boundaries returning it.
    """
    micro, traces = columnar_slab(polygons, thickness, n_layers, h_gb, h_bulk)
    mesh, L = micro.mesh, thickness * UM
    D = lattice_diffusivity(T)
    ratios = ratios(len(traces)) if callable(ratios) else np.asarray(ratios)
    D_gb = D * ratios
    c_front = GAMMA * (1 - REFLECTION) * R_P / D

    grains = fm.grain_subdomains(micro, F.Material(D_0=D, E_D=0.0))
    network = fm.grain_boundary_network(
        NETWORK_ID, micro, None, diffusivity_by_entity=D_gb
    )
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    of = dict(zip((g.id for g in grains), species, strict=True))
    sources, flux_bcs = fm.exchange(species, c_gb, network, width=DELTA, k=k)
    tol = 1e-3 * L / n_layers
    front, front_mouths = fm.grain_surfaces(
        mesh, grains, lambda x: np.abs(x[2]) < tol, FRONT_ID
    )
    back, back_mouths = fm.grain_surfaces(
        mesh, grains, lambda x: np.abs(x[2] - L) < tol, BACK_ID
    )
    bcs = [
        F.FixedConcentrationBC(subdomain=p, value=c_front, species=of[p.grain_id])
        for p in front
    ]
    bcs += [
        F.FixedConcentrationBC(subdomain=p, value=0.0, species=of[p.grain_id])
        for p in back
    ]
    bcs.append(F.FixedConcentrationBC(subdomain=back_mouths, value=0.0, species=c_gb))
    subdomains = [*grains, network, *front, *back, back_mouths]
    if m > 0:
        bcs.append(
            F.FixedConcentrationBC(
                subdomain=front_mouths, value=m * c_front, species=c_gb
            )
        )
        subdomains.append(front_mouths)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=subdomains,
        species=[*species, c_gb],
        sources=sources,
        boundary_conditions=[*flux_bcs, *bcs],
        temperature=float(T),
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    model.initialise()
    fm.tune_direct_solver(model)
    model.run()

    args = (species, grains, c_gb, network, D, D_gb, traces)
    triangles, lattice, edges, line, entity = plane_flux(*args, L, tol)
    zs = np.unique(np.round(mesh.geometry.x[:, 2] / tol)) * tol
    cut = plane_flux(*args, zs[zs.size // 2], tol)
    length = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    cut_length = np.linalg.norm(cut[2][:, 1] - cut[2][:, 0], axis=1)
    outflow = np.sum(lattice * area(triangles)) + np.sum(line * length)
    through = np.sum(cut[1] * area(cut[0])) + np.sum(cut[3] * cut_length)
    return dict(
        traces=traces,
        ratios=ratios,
        triangles=triangles,
        lattice=lattice,
        edges=edges,
        line=line,
        edge_entity=entity,
        J_bulk=D * c_front / L,
        outflow=outflow,
        mouth_share=np.sum(line * length) / outflow,
        total_over_bulk=outflow / (D * c_front / L * np.sum(area(triangles))),
        balance=(through - outflow) / outflow,
        nodes=mesh.geometry.x.shape[0],
    )


def flux_lookup(result):
    """A function giving the back-surface lattice flux at ``(n, 2)`` points."""
    from matplotlib.tri import Triangulation

    flat = result["triangles"].reshape(-1, 2)
    vertices, index = np.unique(np.round(flat, 12), axis=0, return_inverse=True)
    mesh = Triangulation(vertices[:, 0], vertices[:, 1], index.reshape(-1, 3))
    find = mesh.get_trifinder()

    def flux_at(points):
        """Flux of the triangle each point falls in; 0 outside the surface."""
        tri = find(points[:, 0], points[:, 1])
        return np.where(tri >= 0, result["lattice"][tri], 0.0)

    return flux_at


def read_array(result, extent):
    """The hexagonal dot array's view of a solved slab, with the paper's counts."""
    pattern = hg.IndicatorPattern()
    centres = hg.hexagonal_centres(pattern, extent)
    inventory = hg.disc_signals(
        pattern,
        centres,
        flux_lookup(result),
        result["edges"],
        result["line"],
        exposure=EXPOSURE,
    )
    seen = hg.normalise(hg.response(inventory, DETECTION, SATURATION))
    on_gb = hg.on_boundary(pattern, centres, result["traces"])
    active = result["ratios"] > 1e3
    on_active = hg.on_boundary(pattern, centres, result["traces"][active])
    discoloured = seen > 0.1
    return dict(
        centres=centres,
        inventory=inventory,
        seen=seen,
        on_gb=on_gb,
        on_active=on_active,
        discoloured=discoloured,
        counts=hg.count(discoloured, on_gb),
    )


def check():
    """One planar boundary in 3D against the 2D cross-section."""
    rows = []
    for foil, layers in ((25, 12), (50, 16)):
        width, depth = 60.0, 24.0
        tic = time.perf_counter()
        res = solve(
            strip_polygons(width, depth, [width / 2]),
            float(foil),
            600.0,
            [D_RATIO_ACTIVE],
            K_EXCHANGE,
            MOUTH_FEED,
            layers,
            h_gb=1.5,
            h_bulk=4.0,
        )
        ref = solve_section(
            Case(
                name="reference",
                foil=foil,
                temperature=600.0,
                h=0.25e-6,
                width=width * UM,
                boundaries=(width / 2 * UM,),
                D_ratio=D_RATIO_ACTIVE,
                k=K_EXCHANGE,
                m=MOUTH_FEED,
            )
        )
        # Mean lattice flux in bands of distance from the boundary: pointwise
        # values on a 2 to 4 um surface mesh are too noisy to compare.
        bands = np.array([0.0, 10.0, 20.0, 30.0]) * UM
        centroid = np.abs(res["triangles"].mean(axis=1)[:, 0] - width / 2 * UM)
        weight = area(res["triangles"])
        left, right = ref["edges"][:-1], ref["edges"][1:]
        distance = np.abs(0.5 * (left + right) - width / 2 * UM)
        ratio = []
        for lo, hi in itertools.pairwise(bands):
            in_3d = (centroid >= lo) & (centroid < hi)
            in_2d = (distance >= lo) & (distance < hi)
            mean_3d = (
                np.sum(res["lattice"][in_3d] * weight[in_3d]) / weight[in_3d].sum()
            )
            mean_2d = np.sum(ref["flux"][in_2d] * (right - left)[in_2d]) / np.sum(
                (right - left)[in_2d]
            )
            ratio.append(mean_3d / mean_2d)
        length = np.linalg.norm(res["edges"][:, 1] - res["edges"][:, 0], axis=1)
        q_3d = np.sum(res["line"] * length) / (depth * UM)
        row = dict(
            foil=foil,
            mouth_3d_over_2d=q_3d / ref["mouths"][0, 2],
            lattice_3d_over_2d=ratio,
            bands_um=(bands / UM).tolist(),
            balance_3d=res["balance"],
            nodes=res["nodes"],
        )
        rows.append(row)
        print(
            f"{foil} um: boundary outflow 3D / 2D {row['mouth_3d_over_2d']:.3f}; "
            f"lattice flux 3D / 2D in bands {row['bands_um']} um from it "
            f"{np.round(ratio, 3).tolist()}; balance {res['balance']:+.1e} "
            f"({time.perf_counter() - tic:.0f} s)",
            flush=True,
        )
    with open(OUTPUT_DIR / "slab3d_check.json", "w") as fh:
        json.dump(rows, fh, indent=1, default=float)


SLABS = {25: (200.0, 12), 50: (250.0, 16)}  # foil: width (um), layers


def run_foil(foil, seed):
    """One synthetic foil at 600 K: ``(result, dots, summary)``."""
    width, layers = SLABS[foil]
    rng = np.random.default_rng(seed + foil)
    grain = FOILS[foil][1] / UM
    polygons = voronoi_polygons(width, round((width / grain) ** 2), rng)

    def ratios(n):
        """A random fraction of conductive boundaries."""
        return np.where(rng.uniform(size=n) < CAPABLE, D_RATIO_ACTIVE, D_RATIO_INACTIVE)

    tic = time.perf_counter()
    res = solve(
        polygons, float(foil), 600.0, ratios, K_EXCHANGE, MOUTH_FEED, layers, 1.5, 4.0
    )
    dots = read_array(res, ((0.0, width * UM), (0.0, width * UM)))
    c = dots["counts"]
    active = res["ratios"] > 1e3
    length = np.linalg.norm(res["traces"][:, 1] - res["traces"][:, 0], axis=1)
    summary = dict(
        foil=foil,
        seed=seed,
        grains=len(polygons),
        boundaries=int(active.size),
        conductive_fraction_by_length=float(length[active].sum() / length.sum()),
        dots=c["dots"],
        dots_on_gb=c["dots_on_gb"],
        dots_on_conductive_gb=int(dots["on_active"].sum()),
        pct_gb=c["pct_gb"],
        pct_gb_err=c["pct_gb_err"],
        pct_grain=c["pct_grain"],
        seen_on_conductive_gb=float(
            (dots["discoloured"] & dots["on_active"]).sum()
            / max(dots["on_active"].sum(), 1)
        ),
        seen_on_closed_gb=float(
            (dots["discoloured"] & dots["on_gb"] & ~dots["on_active"]).sum()
            / max((dots["on_gb"] & ~dots["on_active"]).sum(), 1)
        ),
        total_over_bulk=res["total_over_bulk"],
        mouth_share=res["mouth_share"],
        balance=res["balance"],
        nodes=res["nodes"],
        seconds=time.perf_counter() - tic,
    )
    return res, dots, summary


def foils(seed):
    """The 25 um and 50 um foils at 600 K through the dot array."""
    results = {}
    for foil in SLABS:
        res, dots, summary = run_foil(foil, seed)
        results[foil] = (res, dots, SLABS[foil][0])
        print(json.dumps(summary, indent=1, default=float), flush=True)
        with open(OUTPUT_DIR / f"slab3d_{foil}um.json", "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        with open(OUTPUT_DIR / f"slab3d_{foil}um-dots.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_m", "y_m", "D_per_m2", "normalised", "on_gb", "discoloured"])
            w.writerows(
                zip(
                    dots["centres"][:, 0],
                    dots["centres"][:, 1],
                    dots["inventory"],
                    dots["seen"],
                    dots["on_gb"].astype(int),
                    dots["discoloured"].astype(int),
                    strict=True,
                )
            )
    plot(results)


def statistics(n_seeds):
    """The 50 um foil over several random microstructures: the paper's count.

    Table 3 gives 44.9 +- 2.1 % of the boundary dots discoloured for this
    foil at 600 K. Here the conductive fraction is drawn with probability
    ``CAPABLE``, so each microstructure realises a somewhat different one.
    """
    rows = []
    for seed in range(1, n_seeds + 1):
        _, _, summary = run_foil(50, seed)
        rows.append(summary)
        print(
            f"seed {seed}: conductive {summary['conductive_fraction_by_length']:.0%} "
            f"of the boundary length, {summary['pct_gb']:.0f} % of "
            f"{summary['dots_on_gb']} boundary dots seen, in-grain "
            f"{summary['pct_grain']:.0f} %, balance {summary['balance']:+.1e} "
            f"({summary['seconds']:.0f} s)",
            flush=True,
        )
    on_gb = sum(r["dots_on_gb"] for r in rows)
    seen = sum(r["pct_gb"] / 100 * r["dots_on_gb"] for r in rows)
    pooled = dict(
        dots_on_gb=on_gb,
        pct_gb=100 * seen / on_gb,
        pct_gb_err=100 * np.sqrt(seen) / on_gb,
        conductive_fraction_by_length=float(
            np.mean([r["conductive_fraction_by_length"] for r in rows])
        ),
        seen_on_conductive_gb=float(
            np.mean([r["seen_on_conductive_gb"] for r in rows])
        ),
        seen_on_closed_gb=float(np.mean([r["seen_on_closed_gb"] for r in rows])),
        measured_pct_gb=44.9,
        measured_pct_gb_err=2.1,
    )
    print(json.dumps(pooled, indent=1, default=float))
    with open(OUTPUT_DIR / "slab3d_statistics.json", "w") as fh:
        json.dump(dict(pooled=pooled, runs=rows), fh, indent=1, default=float)


def plot(results):
    """Heat maps of the dot array with the boundary traces, and one dot row each."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(2, len(results), figsize=(5.6 * len(results), 8.2))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for col, (foil, (res, dots, width)) in enumerate(results.items()):
        ax = axes[0, col]
        ax.set_facecolor("#3b0f54")
        active = res["ratios"] > 1e3
        for mask, style in ((active, "-"), (~active, ":")):
            ax.add_collection(
                LineCollection(
                    res["traces"][mask] / UM,
                    colors="white",
                    linewidths=0.9,
                    linestyles=style,
                )
            )
        xy = dots["centres"] / UM
        # Marker area in points^2 for a 5 um dot on these axes.
        size = (5.0 / width * ax.get_window_extent().width * 72 / fig.dpi) ** 2
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=dots["seen"],
            s=size,
            cmap="viridis",
            vmin=0,
            vmax=1,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        ax.set_xlim(0, width)
        ax.set_ylim(0, width)
        ax.set_aspect("equal")
        ax.set_xlabel("position (um)")
        ax.set_ylabel("position (um)")
        fig.colorbar(sc, ax=ax, label="norm. indicator response", shrink=0.8)
        c = dots["counts"]
        ax.set_title(
            f"{foil} um foil, AR {FOILS[foil][1] / FOILS[foil][0]:.1f}, 600 K: "
            f"boundary dots seen {c['pct_gb']:.0f} %, in-grain {c['pct_grain']:.0f} %",
            fontsize=9,
        )
        # The dot row nearest mid-height, as the paper's line profiles.
        rows_y = np.unique(np.round(xy[:, 1], 6))
        row = np.isclose(xy[:, 1], rows_y[rows_y.size // 2])
        order = np.argsort(xy[row, 0])
        x, seen = xy[row, 0][order], dots["seen"][row][order]
        gb = dots["on_gb"][row][order]
        ax.axhline(rows_y[rows_y.size // 2], color="red", lw=0.6, ls="--")
        ax = axes[1, col]
        ax.plot(x, seen, "-", color="0.7", lw=0.8)
        ax.plot(x[~gb], seen[~gb], "s", color="tab:blue", label="in-grain dot")
        ax.plot(x[gb], seen[gb], "s", color="tab:red", label="dot on a boundary")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("position along the dashed row (um)")
        ax.set_ylabel("norm. indicator response")
        ax.legend(fontsize=7, loc="center left")
    fig.suptitle(
        "Synthetic microstructure (random Voronoi, columnar); solid traces: "
        "conductive boundaries, dotted: closed",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_3d.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", choices=["check", "foils", "statistics"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=6, help="for statistics")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.study == "check":
        check()
    elif args.study == "foils":
        foils(args.seed)
    else:
        statistics(args.seeds)


if __name__ == "__main__":
    main()
