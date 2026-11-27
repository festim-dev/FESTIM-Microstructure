"""Is 96 h steady state? Transient runs with traps (Delic et al. 2026).

The other ``delic2026_*`` scripts assume, with the paper, that the 96 h
exposures are at steady state, so that a dot's inventory is its flux times
96 h. Kapser et al. (2018), Nucl. Fusion 58 056027, modelled the same foils
in the same device with a background of 1.5e-5 traps per W atom, de-trapping
energy 1.5 eV and attempt frequency 1e13 s^-1, and fitted their permeation
data only from 48 h on because of the time lag. Those traps saturate below
about 550 K, and filling them is what the implanted D does first: with a
diffusion-limited source the filled zone grows as ``x^2 = 2 D c_front t /
N_t``, which crosses a foil of thickness ``L`` after ``N_t L^2 / (2 Gamma (1
- r) R_p)``, whatever the temperature: 20 h for 25 um, 79 h for 50 um, 317 h
for 100 um.

Two stages:

* ``bulk``: 1D lattice with those traps, 25, 50 and 100 um, 300 to 700 K: the
  D a dot far from any boundary has collected after 96 h, against the
  steady-state value. Also checks that slowing trapping and de-trapping
  together (which widens the 70 nm filling front so that a 2D mesh can
  resolve it) leaves the answer alone.
* ``section``: the 2D cross-section with one boundary between two half
  grains (mirror planes at the grain centres), closed mouth, the same traps
  in the grains: the inventory of the dot on the boundary and of the dots
  beside it after 96 h, against the boundary conductance ``lam``, and what
  the flux at the end would have given over 96 h.

* ``consequences``: the conductance a boundary needs to be seen, at 300 to
  500 K and at 600 K, and what that does to the temperature-series fit of
  ``delic2026_temperature.py`` and to its comparison with Kapser.

    python examples/delic2026_transient.py bulk           # ~2 min
    python examples/delic2026_transient.py section        # ~15 min
    python examples/delic2026_transient.py consequences   # ~3 min
"""

import argparse
import json
import time

from mpi4py import MPI

import festim as F
import numpy as np
from delic2026_crosssection import (
    D0_LATTICE,
    E_LATTICE,
    EXPOSURE,
    GAMMA,
    OUTPUT_DIR,
    R_P,
    REFLECTION,
    UM,
    Case,
    assemble,
    lattice_diffusivity,
    lattice_flux,
    mouth_fluxes,
)

import festim_microstructure as fm
from festim_microstructure.postprocessing import hydrogenography as hg

N_W = 6.3e28  # W atoms per m^3
TRAP_DENSITY = 1.5e-5 * N_W  # m^-3, Kapser et al. (2018)
E_DETRAP = 1.5  # eV
NU = 1e13  # s^-1
# Trapping rate coefficient: a jump to one of 6 sites, 110 pm away.
K0_TRAP = D0_LATTICE / ((1.1e-10) ** 2 * 6 * N_W)  # m^3/s
# Both rates are slowed by this factor in 2D: the equilibrium occupancy is
# unchanged and the filling front widens from 70 nm to 0.7 um. See ``bulk``.
RATE_SCALE = 1e-2

SOURCE = GAMMA * (1 - REFLECTION) * R_P  # D c_front, m^-1 s^-1


def front_time(thickness):
    """When a saturating trap-filling front crosses the foil, in s."""
    return TRAP_DENSITY * thickness**2 / (2 * SOURCE)


def stepsize():
    """Adaptive steps from 1 s, with at least 150 over the exposure."""
    return F.Stepsize(
        initial_value=1.0,
        growth_factor=1.15,
        cutback_factor=0.7,
        target_nb_iterations=5,
        max_stepsize=EXPOSURE / 150,
    )


def trap_reaction(mobile, trapped, volume, scale):
    """Kapser's background trap for one mobile species in one volume."""
    empty = F.ImplicitSpecies(n=TRAP_DENSITY, others=[trapped])
    return F.Reaction(
        reactant=[mobile, empty],
        product=[trapped],
        k_0=scale * K0_TRAP,
        E_k=E_LATTICE,
        p_0=scale * NU,
        E_p=E_DETRAP,
        volume=volume,
    )


def bulk(thickness, T, scale=1.0, cells=2000):
    """1D lattice with traps: ``(inventory after 96 h, steady-state value)``."""
    D = lattice_diffusivity(T)
    model = F.HydrogenTransportProblem()
    model.mesh = F.Mesh1D(np.linspace(0.0, thickness, cells + 1))
    volume = F.VolumeSubdomain1D(
        id=1, borders=[0.0, thickness], material=F.Material(D_0=D, E_D=0.0)
    )
    front = F.SurfaceSubdomain1D(id=2, x=0.0)
    back = F.SurfaceSubdomain1D(id=3, x=thickness)
    model.subdomains = [volume, front, back]
    mobile = F.Species("D")
    trapped = F.Species("D_trapped", mobile=False)
    model.species = [mobile, trapped]
    model.reactions = [trap_reaction(mobile, trapped, volume, scale)]
    model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=front, value=SOURCE / D, species=mobile),
        F.FixedConcentrationBC(subdomain=back, value=0.0, species=mobile),
    ]
    model.temperature = float(T)
    flux = F.SurfaceFlux(field=mobile, surface=back)
    model.exports = [flux]
    model.settings = F.Settings(
        atol=1e8, rtol=1e-10, final_time=EXPOSURE, stepsize=stepsize()
    )
    model.show_progress_bar = False
    model.initialise()
    model.run()
    t, j = np.array(flux.t), np.abs(np.array(flux.data))
    # Implicit Euler: each flux value holds over the step that ends at its time.
    inventory = float(np.sum(j * np.diff(np.concatenate([[0.0], t]))))
    return inventory, SOURCE / thickness * EXPOSURE


def bulk_study():
    """Bulk inventories after 96 h for the three foils and seven temperatures."""
    rows = []
    for L in (25, 50, 100):
        for T in (300, 450, 500, 600, 630, 660, 700):
            tic = time.perf_counter()
            inv, steady = bulk(L * UM, T)
            rows.append(
                dict(
                    foil_um=L,
                    T=T,
                    inventory=inv,
                    steady_inventory=steady,
                    ratio=inv / steady,
                    front_time_h=front_time(L * UM) / 3600,
                )
            )
            print(
                f"{L:3d} um {T} K: {inv:.2e} D/m2 after 96 h = {inv / steady:.2f} of "
                f"steady ({time.perf_counter() - tic:.0f} s)",
                flush=True,
            )
    # The rate scaling used in 2D, where it matters most and least.
    scaling = {}
    for L, T in ((25, 300), (50, 500), (25, 600), (50, 600)):
        exact = next(r for r in rows if r["foil_um"] == L and r["T"] == T)["inventory"]
        slowed, steady = bulk(L * UM, T, scale=RATE_SCALE, cells=400)
        scaling[f"{L}um-{T}K"] = dict(
            exact=exact / steady, slowed_coarse=slowed / steady
        )
        print(
            f"scaling check {L} um {T} K: {exact / steady:.3f} vs {slowed / steady:.3f}"
        )
    with open(OUTPUT_DIR / "transient_bulk.json", "w") as fh:
        json.dump(dict(rows=rows, rate_scaling=scaling), fh, indent=1)


def section(foil, T, lam, grain, traps=True, k=1e-2, m=0.0, h=0.5e-6):
    """One boundary between two half grains: dot inventories after 96 h.

    Returns the inventories of the dot on the boundary and of the dots every
    10 um from it up to the grain centre, and what the flux at the end of the
    exposure would have given had it held for the whole 96 h.
    """
    case = Case(
        name="transient",
        foil=foil,
        temperature=float(T),
        h=h,
        width=grain,
        boundaries=(grain / 2,),
        D_ratio=lam / 1e-9,
        k=k,
        m=m,
    )
    parts = assemble(case)
    species, grains = parts["species"], parts["grains"]
    all_species, reactions = [*species, parts["c_gb"]], []
    if traps:
        for mobile, g in zip(species, grains, strict=True):
            trapped = F.Species(f"t_{g.id}", mobile=False, subdomains=[g])
            all_species.append(trapped)
            reactions.append(trap_reaction(mobile, trapped, g, RATE_SCALE))
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(parts["mesh"]),
        subdomains=parts["subdomains"],
        species=all_species,
        reactions=reactions,
        sources=parts["sources"],
        boundary_conditions=parts["boundary_conditions"],
        temperature=case.temperature,
        settings=F.Settings(
            atol=1e6, rtol=1e-10, final_time=EXPOSURE, stepsize=stepsize()
        ),
    )
    model.show_progress_bar = False
    model.initialise()
    fm.tune_direct_solver(model)

    thickness, xb = parts["thickness"], parts["boundaries"][0]
    width = parts["width"]
    pattern = hg.IndicatorPattern(offset=xb % 10e-6)
    inventory, rate, steps = 0.0, None, 0
    while float(model.t) < EXPOSURE:
        t0 = float(model.t)
        model.iterate()
        left, right, j = lattice_flux(species, grains, parts["D"], thickness)
        mouths = mouth_fluxes(
            parts["c_gb"],
            parts["network"],
            parts["boundaries"],
            case.delta,
            parts["D_gb"],
            thickness,
            parts["micro"].tolerance,
            thickness / 2,
        )
        # Mirror the surface about the right-hand grain centre, so that a dot
        # sitting on that symmetry plane is whole.
        edges = np.append(left, right[-1])
        edges = np.concatenate([edges, 2 * width - edges[-2::-1]])
        centres, rate = hg.dot_signals(
            pattern,
            edges,
            np.concatenate([j, j[::-1]]),
            mouths=[(row[0], row[2]) for row in mouths],
            exposure=1.0,
        )
        inventory = inventory + rate * (float(model.t) - t0)
        steps += 1
    keep = (centres >= xb - 1e-9) & (centres <= width + 1e-9)
    return dict(
        distance_um=((centres[keep] - xb) / UM).round(3).tolist(),
        inventory=inventory[keep].tolist(),
        steady_inventory=(rate[keep] * EXPOSURE).tolist(),
        steps=steps,
    )


def section_study(which):
    """Boundary-dot inventories against conductance, with and without traps."""
    runs = []
    if "response" in which:
        for T in (300, 600):
            for lam in (1, 3, 10, 30, 100, 300, 1000):
                runs.append(dict(foil=50, T=T, lam=lam * UM, grain=60 * UM))
        runs.append(dict(foil=50, T=450, lam=30 * UM, grain=60 * UM))
    if "profiles" in which:
        # The 600 K line-profile best fit, in the two profiled grains.
        best = dict(T=600, lam=316 * UM, k=1e-4, m=0.13)
        runs.append(dict(foil=25, grain=70 * UM, **best))
        runs.append(dict(foil=50, grain=60 * UM, **best))
    out = []
    for run in runs:
        tic = time.perf_counter()
        res = section(**run)
        out.append({**run, **res})
        print(
            f"{run['foil']} um {run['T']} K lam {run['lam'] / UM:6.0f} um: "
            f"boundary dot {res['inventory'][0]:.2e} "
            f"(steady {res['steady_inventory'][0]:.2e}), "
            f"grain centre {res['inventory'][-1]:.2e} "
            f"(steady {res['steady_inventory'][-1]:.2e}), {res['steps']} steps, "
            f"{time.perf_counter() - tic:.0f} s",
            flush=True,
        )
        if MPI.COMM_WORLD.rank == 0:
            with open(
                OUTPUT_DIR / f"transient_section_{'_'.join(which)}.json", "w"
            ) as fh:
                json.dump(out, fh, indent=1)


def threshold_lam(rows, T, threshold):
    """Conductance at which the boundary dot reaches ``threshold`` at ``T``."""
    rows = sorted((r for r in rows if r["T"] == T), key=lambda r: r["lam"])
    lam = np.log([r["lam"] for r in rows])
    inv = np.log([r["inventory"][0] for r in rows])
    return float(np.exp(np.interp(np.log(threshold), inv, lam)))


def consequences(n_seeds=6):
    """What the transient changes in the temperature-series fit.

    The boundary dot needs a larger conductance to be seen while the traps
    are filling (300 to 500 K) than at 600 K. Refit the counts with those two
    thresholds, re-pin the distribution, and redo the Kapser comparison.
    """
    import delic2026_temperature as ts

    with open(OUTPUT_DIR / "transient_section_response.json") as fh:
        rows = json.load(fh)
    _, _, threshold = ts.indicator()
    lam_low = threshold_lam(rows, 300, threshold)
    lam_600 = threshold_lam(rows, 600, threshold)
    c_low, c_600 = np.log(lam_low / UM), np.log(lam_600 / UM)
    T = np.array([row[0] for row in ts.FITTED], dtype=float)
    # Up to 500 K the threshold is one number, so those three points test the
    # single population on their own.
    cold = [row for row in ts.FITTED if row[0] <= 500]
    single_cold = ts.fit_counts(False, rows=cold)
    out = dict(
        threshold=threshold,
        lam_threshold_um={"300-500K": lam_low / UM, "600K": lam_600 / UM},
        single_population_300_to_500K=dict(
            chi2=single_cold["chi2"], dof=single_cold["dof"]
        ),
        pinned={},
    )
    rng = np.random.default_rng(7)
    for lam_active in (316 * UM, 30 * UM):
        shift, sigma = np.zeros(T.size), None
        for _ in range(20):  # the 600 K shift depends on sigma: fixed point
            counts = ts.fit_counts(True, shift=shift)
            a, mu, sigma_new = ts.distribution(counts, c_low, lam_active, c_600)
            shift = np.where(T >= 600, (c_low - c_600) * ts.K_B * T / sigma_new, 0.0)
            if sigma is not None and abs(sigma_new - sigma) < 1e-6:
                break
            sigma = sigma_new
        fitted = dict(
            models={"capable": counts},
            chosen="capable",
            a=float(a),
            mu_eV=float(mu),
            sigma_eV=float(sigma_new),
        )
        kapser = ts.kapser_fluxes(fitted, n_seeds, 0.5e-6, rng)
        out["pinned"][f"{lam_active / UM:.0f}um"] = dict(
            p=counts["p"],
            chi2=counts["chi2"],
            dof=counts["dof"],
            T50=(counts["beta"]) / counts["alpha"],
            mu_eV=float(mu),
            sigma_eV=float(sigma_new),
            ln_prefactor=float(a),
            kapser_total_over_bulk={
                str(k): v["total_over_bulk"] for k, v in kapser.items()
            },
            kapser_ratio_450_300=kapser[450]["total_over_bulk"]
            / kapser[300]["total_over_bulk"],
        )
    with open(OUTPUT_DIR / "transient_consequences.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    plot(rows, out)


def plot(rows, cons):
    """Bulk inventory against temperature, boundary dot against conductance."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    with open(OUTPUT_DIR / "transient_bulk.json") as fh:
        bulk_rows = json.load(fh)["rows"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8))
    for L, marker in ((25, "o"), (50, "s"), (100, "^")):
        sel = [r for r in bulk_rows if r["foil_um"] == L]
        ax0.plot(
            [r["T"] for r in sel],
            [max(r["inventory"], 1e17) for r in sel],
            marker + "-",
            label=f"{L} um (front crosses at {sel[0]['front_time_h']:.0f} h)",
        )
        ax0.plot(
            [r["T"] for r in sel],
            [r["steady_inventory"] for r in sel],
            ":",
            color="0.5",
        )
    ax0.axhline(cons["threshold"], color="tab:red", ls="--", lw=1)
    ax0.set_yscale("log")
    ax0.set_ylim(1e18, 1e20)
    ax0.set_xlabel("T (K)")
    ax0.set_ylabel("in-grain dot, D m$^{-2}$ in 96 h")
    ax0.set_title("bulk with Kapser's traps (dotted: steady state)", fontsize=9)
    ax0.legend(fontsize=7)
    for T, colour in ((300, "tab:blue"), (600, "tab:orange")):
        sel = sorted((r for r in rows if r["T"] == T), key=lambda r: r["lam"])
        lam = [r["lam"] / UM for r in sel]
        ax1.plot(
            lam, [r["inventory"][0] for r in sel], "o-", color=colour, label=f"{T} K"
        )
        ax1.plot(lam, [r["steady_inventory"][0] for r in sel], ":", color=colour)
    ax1.axhline(cons["threshold"], color="tab:red", ls="--", lw=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("boundary conductance lam (um)")
    ax1.set_ylabel("boundary dot, D m$^{-2}$ in 96 h")
    th = cons["lam_threshold_um"]
    ax1.set_title(
        f"50 um, closed mouth: seen above {th['300-500K']:.0f} um (300 to 500 K), "
        f"{th['600K']:.1f} um (600 K)",
        fontsize=9,
    )
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_transient.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["bulk", "section", "consequences"])
    parser.add_argument("--which", nargs="*", default=["response", "profiles"])
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "bulk":
        bulk_study()
    elif args.stage == "section":
        section_study(args.which)
    else:
        consequences()


if __name__ == "__main__":
    main()
