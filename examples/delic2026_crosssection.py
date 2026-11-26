"""Grain-boundary permeation through a W foil, seen through indicator dots.

A first step towards validating the codimension-one model against the
patterned hydrogenography of Delic, Manhard and von Toussaint (2026), Nucl.
Fusion, doi:10.1088/1741-4326/aec6b0. Their recrystallised W foils (25, 50 and
100 um thick) are exposed to a 5 eV D plasma on the front for 96 h, and an array
of 5 um indicator dots on the back records where D comes out. They report three
things a model has to get right:

* at 600 K, with grains as wide as or narrower than the foil is thick, only the
  dots on grain boundaries discolour;
* in the thinnest foil (grain size / thickness 1.6), the centres of large
  grains discolour too, with a depleted band next to each boundary: an
  "inverse permeation halo";
* at 630 K and above, flux spreads into the grains next to some boundaries: a
  "permeation halo".

This script is the cheapest model that can show all three: a 2D cross-section
of the foil through columnar grains, as their SEM fractographs show for the
25 and 50 um foils (their Fig. 6). Every grain carries its own lattice species,
the boundaries are one codimension-one network, and they exchange through
``k (c_grain - c_gb / s)``, with ``s`` a segregation factor. The two surfaces:

* front (``y = 0``): diffusion-limited implantation, ``c = Gamma R_p / D`` in
  every grain. The boundary mouths there carry ``c_gb = m s c_front``: ``m = 1``
  is a mouth in equilibrium with the lattice beside it, ``m = 0`` a mouth the
  plasma cannot feed directly (an entry barrier), ``m > 1`` a mouth fed more
  than the lattice (for example by the radicals, which outnumber the ions a
  hundredfold and may enter more easily at a boundary).
* back (``y = L``): the indicator is taken as a perfect sink, ``c = 0`` for
  every species.

With ``m = 1`` the lattice and the network both carry linear profiles and never
exchange anything: the boundaries only add their own flux at the mouths. Every
halo, inverse or not, comes from a mouth out of equilibrium, ``m != 1``, and two
lengths decide what it does:

* ``lam = delta s D_gb / D``, the conductance of the boundary measured in
  lattice: a mouth outflow of ``lam * J_bulk``, so a dot on a boundary sees
  about ``lam / d`` times the bulk. A closed mouth (``m = 0``) can only drain the
  grains beside it if ``lam`` is comparable to the thickness ``L``.
* ``ell = sqrt(delta s D_gb / (2 k))``, the fin length over which the network
  relaxes to equilibrium with the lattice. The mouth condition reaches the
  back only if ``ell`` is comparable to ``L``; with ``ell << L`` the network
  is in local equilibrium everywhere but a thin layer at the front.

So an inverse halo wants ``m < 1``, ``lam >~ L`` and ``ell << L`` (a strongly
coupled short circuit with an entry barrier); a halo wants ``m > 1`` and
``ell >~ L`` (a weakly coupled boundary fed harder than the lattice). The
script reports the D a fed mouth takes in, against what the plasma brings to
a slab ``delta`` wide, since a fixed mouth concentration can demand more than
the plasma can supply.

The back-surface flux is then read the way the experiment reads it, through
:mod:`festim_microstructure.postprocessing.hydrogenography`: dots on a 10 um
pitch, each integrating the lattice flux over its footprint plus any boundary
mouth inside it, normalised per image and binarised.

Parameters below are plausible, not fitted. Lattice diffusivity is Frauenfelder
(1969); the boundary diffusivity ratio, segregation and exchange coefficient are
free, which is the point: the counts and line profiles of the paper are what
should pin them. Grain widths are drawn from a log-normal with the mean and
standard deviation of their Table 1, using the surface equivalent diameter as
the cross-section width (a section through a grain cuts a shorter chord, so
this slightly overstates the width).

Run all studies, or pick some:

    python examples/delic2026_crosssection.py
    python examples/delic2026_crosssection.py --study mouth
    python examples/delic2026_crosssection.py --study thickness --h 0.125e-6
"""

import argparse
import csv
import functools
import itertools
import json
import operator
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from mpi4py import MPI

import dolfinx
import festim as F
import numpy as np
import ufl

import festim_microstructure as fm
from festim_microstructure.postprocessing import hydrogenography as hg

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem

UM = 1e-6
NETWORK_ID = 10_000  # above every grain id
FRONT_ID, BACK_ID = 20_000, 30_000  # first patch id on each surface

# Plasma (Delic et al. 2026, Sect. 3.3) and implantation.
GAMMA = 6e19  # D m^-2 s^-1, incident ion flux
REFLECTION = 0.5  # assumed; low-energy D on W reflects strongly
R_P = 0.5e-9  # m, assumed implantation depth at 5 eV/D
EXPOSURE = 96 * 3600.0  # s

# Lattice diffusivity, Frauenfelder (1969).
D0_LATTICE = 4.1e-7  # m^2/s
E_LATTICE = 0.39  # eV

# Foils, Delic et al. 2026, Table 1: thickness, grain mean and std (m).
FOILS = {
    25: (25 * UM, 39.3 * UM, 18.6 * UM),
    50: (50 * UM, 54.7 * UM, 37.3 * UM),
    100: (100 * UM, 21.2 * UM, 12.7 * UM),
}


@dataclass(frozen=True)
class Case:
    """One foil, one temperature, one set of boundary parameters."""

    name: str
    foil: int = 50  # um, a key of FOILS
    temperature: float = 600.0  # K
    width: float = 300 * UM  # length of the cross-section
    h: float = 0.25 * UM  # mean mesh size; dots converge to ~1 %
    seed: int = 1
    delta: float = 1e-9  # m, boundary slab width
    D_ratio: float = 1e5  # D_gb / D_lattice
    s: float = 1.0  # segregation, equilibrium c_gb / c_grain
    k: float = 1e-2  # m/s, grain-to-network exchange coefficient
    m: float = 0.0  # front-mouth feed, c_gb(0) = m s c_front

    def lengths(self):
        """``(lam, ell)``, the conductance and fin lengths, in metres."""
        conductance = self.delta * self.s * self.D_ratio  # times D
        D_gb = self.D_ratio * lattice_diffusivity(self.temperature)
        return conductance, np.sqrt(self.delta * self.s * D_gb / (2 * self.k))


def lattice_diffusivity(T):
    """Frauenfelder's D in m^2/s."""
    return D0_LATTICE * np.exp(-E_LATTICE / (F.k_B * T))


def boundary_positions(width, mean, std, h, seed):
    """Columnar boundary abscissae, snapped to the mesh and kept interior.

    Widths are log-normal with the given mean and standard deviation; the
    first grain starts at a random point so that the left edge cuts a grain
    rather than sitting on a boundary.
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.log1p((std / mean) ** 2)
    mu = np.log(mean) - sigma2 / 2
    x = -rng.uniform(0, rng.lognormal(mu, np.sqrt(sigma2)))
    xs = []
    while True:
        x += rng.lognormal(mu, np.sqrt(sigma2))
        if x >= width - 2 * h:
            break
        if x > 2 * h:
            xs.append(round(x / h) * h)
    xs = np.unique(xs)
    # Two boundaries closer than a few cells would make a grain one cell wide.
    keep = np.concatenate([[True], np.diff(xs) > 4 * h])
    return xs[keep]


def _clustered(n, beta):
    """``n + 1`` points on ``[0, 1]``, bunched towards both ends.

    A blend of uniform and Chebyshev spacing: the end cells are ``1 - beta``
    times the uniform ones and shrink further towards the ends.
    """
    xi = np.linspace(0.0, 1.0, n + 1)
    return (1 - beta) * xi + beta * (1 - np.cos(np.pi * xi)) / 2


def columnar_section(width, thickness, boundaries, h, beta=0.9):
    """A mesh of ``[0, width] x [0, thickness]`` cut by the boundaries.

    Structured, ``h`` on average, and graded towards every boundary and both
    surfaces: where a boundary that drains or feeds the lattice meets a
    charged surface the concentration changes over the fin length, which is
    often well below a micrometre. ``beta = 0`` gives a uniform mesh.
    """
    breaks = np.concatenate([[0.0], boundaries, [round(width / h) * h]])
    xs = [np.array([0.0])]
    for a, b in itertools.pairwise(breaks):
        n = max(round((b - a) / h), 2)
        xs.append(a + (b - a) * _clustered(n, beta)[1:])
    xs = np.concatenate(xs)
    ny = max(round(thickness / h), 2)
    ys = round(thickness / h) * h * _clustered(ny, beta)
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([xs.size - 1.0, ny])],
        [xs.size - 1, ny],
        cell_type=dolfinx.mesh.CellType.triangle,
    )
    # Map the unit-spaced grid onto the graded one, vertex by vertex.
    geom = mesh.geometry.x
    geom[:, 1] = ys[np.rint(geom[:, 1]).astype(int)]
    geom[:, 0] = xs[np.rint(geom[:, 0]).astype(int)]
    h_min = min(np.diff(xs).min(), np.diff(ys).min())
    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local
    cells = np.arange(n_local, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    grain = 1 + np.searchsorted(boundaries, midpoints[:, 0]).astype(np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cells, grain)
    tol = 1e-3 * h_min

    def on_network(x):
        """Facets lying on one of the boundary lines."""
        return np.abs(x[0][:, None] - boundaries[None, :]).min(axis=1) < tol

    micro = fm.TaggedMicrostructure(
        mesh=mesh,
        cell_tags=cell_tags,
        tolerance=tol,
        network_locator=on_network,
        name=f"columnar section, {boundaries.size} boundaries",
    )
    return micro, xs[-1], ys[-1]


def lattice_flux(species, grains, D, y_line):
    """Lattice flux along ``+y`` through the mesh line ``y = y_line``.

    Piecewise constant on the facets of the line: P1 gradients are constant
    per cell, and each facet takes the cell below it (above it on the front
    surface, which has nothing below). Returns ``(left, right, flux)`` per
    facet, gathered from every rank and sorted along ``x``.
    """
    lefts, rights, values = [], [], []
    for spe, grain in zip(species, grains, strict=True):
        u = spe.subdomain_to_post_processing_solution[grain]
        sub = u.function_space.mesh
        tdim = sub.topology.dim
        sub.topology.create_connectivity(tdim - 1, tdim)
        facets = dolfinx.mesh.locate_entities(
            sub, tdim - 1, lambda x, y=y_line: np.isclose(x[1], y)
        )
        facets = facets[facets < sub.topology.index_map(tdim - 1).size_local]
        if facets.size == 0:
            continue
        f2c = sub.topology.connectivity(tdim - 1, tdim)
        cells = []
        for f in facets:
            pair = f2c.links(f)
            mid = dolfinx.mesh.compute_midpoints(sub, tdim, pair)[:, 1]
            cells.append(pair[np.argmin(mid)])
        cells = np.array(cells, dtype=np.int32)
        expr = dolfinx.fem.Expression(-D * ufl.grad(u)[1], np.array([[1 / 3, 1 / 3]]))
        values.append(expr.eval(sub, cells).reshape(-1))
        nodes = dolfinx.mesh.entities_to_geometry(sub, tdim - 1, facets)
        ends = np.sort(sub.geometry.x[nodes, 0], axis=1)
        lefts.append(ends[:, 0])
        rights.append(ends[:, 1])
    gather = MPI.COMM_WORLD.allgather
    left = np.concatenate(functools.reduce(operator.iadd, gather(lefts), []))
    right = np.concatenate(functools.reduce(operator.iadd, gather(rights), []))
    flux = np.concatenate(functools.reduce(operator.iadd, gather(values), []))
    order = np.argsort(left)
    return left[order], right[order], flux[order]


def mouth_fluxes(c_gb, network, boundaries, delta, D_gb, thickness, tol, y_cut):
    """``(x, Q_front, Q_back, Q_cut)`` per boundary, network flux along ``+y``.

    Per metre out of plane, read from the P1 network solution along each
    straight boundary: the segment next to each end, and the one below
    ``y_cut``. The front value is the flux at the middle of the first segment,
    so it is only the mouth's intake when the fin length is well resolved.
    """
    u = c_gb.subdomain_to_post_processing_solution[network]
    coords = u.function_space.tabulate_dof_coordinates()
    values = u.x.array
    gather = MPI.COMM_WORLD.allgather
    coords = np.concatenate(gather(coords))
    values = np.concatenate(gather(values))
    rows = []
    for xb in boundaries:
        on = np.abs(coords[:, 0] - xb) < tol
        y, idx = np.unique(coords[on, 1], return_index=True)
        c = values[on][idx]
        if y.size < 2 or not np.isclose(y[-1], thickness) or not np.isclose(y[0], 0):
            raise RuntimeError(f"boundary at x = {xb:.3e} is not a full line")
        q_front = -delta * D_gb * (c[1] - c[0]) / (y[1] - y[0])
        q_back = -delta * D_gb * (c[-1] - c[-2]) / (y[-1] - y[-2])
        j = np.searchsorted(y, y_cut - tol)
        q_cut = -delta * D_gb * (c[j] - c[j - 1]) / (y[j] - y[j - 1])
        rows.append((xb, q_front, q_back, q_cut))
    return np.array(rows)


def solve(case):
    """Solve one case at steady state and read the back surface off it."""
    thickness, mean, std = FOILS[case.foil]
    boundaries = boundary_positions(case.width, mean, std, case.h, case.seed)
    micro, width, thickness = columnar_section(
        case.width, thickness, boundaries, case.h
    )
    mesh = micro.mesh

    D = lattice_diffusivity(case.temperature)
    D_gb = case.D_ratio * D
    c_front = GAMMA * (1 - REFLECTION) * R_P / D

    grains = fm.grain_subdomains(micro, F.Material(D_0=D, E_D=0.0))
    network = fm.grain_boundary_network(
        NETWORK_ID, micro, F.Material(D_0=D_gb, E_D=0.0)
    )
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    of = dict(zip((g.id for g in grains), species, strict=True))

    k, s = case.k, case.s
    sources, flux_bcs = fm.exchange(
        species, c_gb, network, width=case.delta, flux=lambda cg, cn: k * (cg - cn / s)
    )
    front, front_mouths = fm.grain_surfaces(
        mesh, grains, lambda x: np.isclose(x[1], 0.0), FRONT_ID
    )
    back, back_mouths = fm.grain_surfaces(
        mesh, grains, lambda x: np.isclose(x[1], thickness), BACK_ID
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
    if case.m > 0:
        bcs.append(
            F.FixedConcentrationBC(
                subdomain=front_mouths, value=case.m * s * c_front, species=c_gb
            )
        )
        subdomains.append(front_mouths)
    # m = 0: no condition, i.e. no flux, at the front mouths.

    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(mesh),
        subdomains=subdomains,
        species=[*species, c_gb],
        sources=sources,
        boundary_conditions=[*flux_bcs, *bcs],
        temperature=case.temperature,
        settings=F.Settings(atol=1e-10, rtol=1e-10, transient=False),
    )
    model.initialise()
    fm.tune_direct_solver(model)
    model.run()

    # Every line y = const carries the whole steady flux. A cut half-way
    # through, away from the corners where the network meets the surfaces,
    # is the conservation check; the front is where the plasma's intake is.
    ys = np.unique(mesh.geometry.x[:, 1])
    y_cut = ys[ys.size // 2]
    left, right, j_back = lattice_flux(species, grains, D, thickness)
    fl, fr, j_front = lattice_flux(species, grains, D, 0.0)
    cl, cr, j_cut = lattice_flux(species, grains, D, y_cut)
    mouths = mouth_fluxes(
        c_gb, network, boundaries, case.delta, D_gb, thickness, micro.tolerance, y_cut
    )
    outflow = np.sum(j_back * (right - left)) + mouths[:, 2].sum()
    through_cut = np.sum(j_cut * (cr - cl)) + mouths[:, 3].sum()
    balance = (through_cut - outflow) / outflow
    # What the front mouths take in. The lattice intake is read from the
    # cells at the surface, which converges only at first order next to a
    # draining mouth, so closed mouths (which take in nothing) show this
    # estimate's own error.
    injected = outflow - np.sum(j_front * (fr - fl))
    edges = np.append(left, right[-1])
    if not np.allclose(left[1:], right[:-1]):
        raise RuntimeError("back-surface facets do not tile the surface")
    return {
        "case": case,
        "width": width,
        "thickness": thickness,
        "boundaries": boundaries,
        "edges": edges,
        "flux": j_back,
        "mouths": mouths,
        "J_bulk": D * c_front / thickness,
        "balance": balance,
        # Per mouth, against the ion flux falling on a slab delta wide.
        "injection_over_ion_budget": injected / boundaries.size / (GAMMA * case.delta),
    }


def read_dots(result, pattern, mode="range", threshold=0.1):
    """The indicator view of one solved case, with the paper's counts."""
    centres, signals = hg.dot_signals(
        pattern,
        result["edges"],
        result["flux"],
        mouths=[(row[0], row[2]) for row in result["mouths"]],
        exposure=EXPOSURE,
    )
    on_gb = hg.on_boundary(pattern, centres, result["boundaries"])
    discoloured = hg.classify(signals, mode, threshold)
    above = hg.classify(signals, "absolute", hg.DETECTION_LIMIT)
    detected = {f"{k}_abs": v for k, v in hg.count(above, on_gb).items()}
    return {
        "centres": centres,
        "signals": signals,
        "on_gb": on_gb,
        "discoloured": discoloured,
        "counts": {**hg.count(discoloured, on_gb), **detected},
        "above_detection": above,
    }


def save(result, dots, out):
    """CSV of the back flux and of the dots, and a JSON summary line."""
    case = result["case"]
    with open(out / f"{case.name}-flux.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_left_m", "x_right_m", "flux_m-2s-1"])
        e = result["edges"]
        w.writerows(zip(e[:-1], e[1:], result["flux"], strict=True))
    with open(out / f"{case.name}-dots.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_m", "D_per_m2", "on_gb", "discoloured", "above_detection"])
        w.writerows(
            zip(
                dots["centres"],
                dots["signals"],
                dots["on_gb"].astype(int),
                dots["discoloured"].astype(int),
                dots["above_detection"].astype(int),
                strict=True,
            )
        )
    return {
        **asdict(case),
        "AR": FOILS[case.foil][1] / FOILS[case.foil][0],
        "J_bulk": result["J_bulk"],
        "balance": result["balance"],
        "lam": case.lengths()[0],
        "ell": case.lengths()[1],
        "injection_over_ion_budget": result["injection_over_ion_budget"],
        "Q_back_over_dJ": float(
            np.mean(result["mouths"][:, 2]) / (5 * UM * result["J_bulk"])
        ),
        **dots["counts"],
    }


def plot(results, dots_list, title, path):
    """Back surface in the indicator's units, one panel per case.

    The line is what a dot would collect at each point (lattice flux times
    exposure), the markers what each dot does collect, mouths included, and
    the dashed line the indicator's detection limit. Red: dots on a
    boundary, orange: dots on a grain, filled when above the limit.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=True, squeeze=False)
    for ax, res, dots in zip(axes[:, 0], results, dots_list, strict=True):
        e = res["edges"]
        x = np.repeat(e, 2)[1:-1] / UM
        y = np.repeat(res["flux"] * EXPOSURE, 2)
        ax.plot(x, y, color="0.3", lw=1)
        for xb in res["boundaries"]:
            ax.axvline(xb / UM, color="tab:blue", lw=0.6, ls=":")
        ax.axhline(hg.DETECTION_LIMIT, color="k", lw=0.8, ls="--")
        colour = np.where(dots["on_gb"], "tab:red", "tab:orange")
        face = np.where(dots["above_detection"], colour, "none")
        ax.scatter(
            dots["centres"] / UM,
            dots["signals"],
            edgecolors=colour,
            facecolors=face,
            s=16,
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_ylabel("D m$^{-2}$ in 96 h")
        c = dots["counts"]
        ax.set_title(
            f"{res['case'].name}: above detection, GB {c['pct_gb_abs']:.0f} %, "
            f"grain {c['pct_grain_abs']:.0f} %; >10 % of range, "
            f"GB {c['pct_gb']:.0f} %, grain {c['pct_grain']:.0f} % (of GB dots)",
            fontsize=8,
            loc="left",
        )
    axes[-1, 0].set_xlabel("x (um)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def studies(base):
    """The two parameter studies, as lists of cases."""
    return {
        # Strongly coupled short circuits with an entry barrier (lam >~ L,
        # ell << L, m = 0) in the three foils at 600 K: the inverse halo and
        # its dependence on grain size / thickness. The 100 um foil is not
        # columnar in reality (their Fig. 6c); it shows only the trend.
        "thickness": [
            replace(base, name="foil-25um", foil=25),
            replace(base, name="foil-50um", foil=50),
            replace(base, name="foil-100um", foil=100),
        ],
        # Weakly coupled boundaries in the 50 um foil (ell ~ L), with the
        # front mouths closed, in equilibrium, and over-fed: the halo.
        "mouth": [
            replace(base, name="mouth-m0", k=4e-6, m=0.0),
            replace(base, name="mouth-m1", k=4e-6, m=1.0),
            replace(base, name="mouth-m3", k=4e-6, m=3.0),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=["mouth", "thickness"], nargs="*")
    parser.add_argument("--h", type=float, default=Case.h, help="mesh size, m")
    parser.add_argument("--width", type=float, default=Case.width)
    parser.add_argument("--mode", default="range", help="hydrogenography.classify")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = Case(name="base", h=args.h, width=args.width)
    pattern = hg.IndicatorPattern()
    chosen = args.study or ["mouth", "thickness"]
    summary = []
    for study, cases in studies(base).items():
        if study not in chosen:
            continue
        results, views = [], []
        for case in cases:
            tic = time.perf_counter()
            res = solve(case)
            dots = read_dots(res, pattern, args.mode, args.threshold)
            row = save(res, dots, OUTPUT_DIR)
            row["seconds"] = time.perf_counter() - tic
            summary.append(row)
            results.append(res)
            views.append(dots)
            if MPI.COMM_WORLD.rank == 0:
                print(
                    f"{case.name:12s} balance {res['balance']:+.1e}  "
                    f"detected GB {row['pct_gb_abs']:3.0f} % "
                    f"grain {row['pct_grain_abs']:3.0f} %  "
                    f"range GB {row['pct_gb']:3.0f} % grain {row['pct_grain']:3.0f} %  "
                    f"lam {row['lam'] / UM:.0f} um  ell {row['ell'] / UM:.1f} um  "
                    f"mouth/(d J_bulk) {row['Q_back_over_dJ']:.2f}  "
                    f"injection/ion budget {row['injection_over_ion_budget']:.2g}  "
                    f"({row['seconds']:.0f} s)"
                )
        if MPI.COMM_WORLD.rank == 0:
            plot(
                results,
                views,
                f"{study} study, {base.temperature:.0f} K, D_gb/D = {base.D_ratio:g}, "
                f"s = {base.s:g}, k = {cases[0].k:g} m/s, delta = {base.delta:g} m",
                OUTPUT_DIR / f"fig_{study}.png",
            )
    if MPI.COMM_WORLD.rank == 0:
        with open(OUTPUT_DIR / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
