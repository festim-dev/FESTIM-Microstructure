"""Fit the cross-section model to the line profiles of Delic et al. (2026).

The profiles below were read by eye from the accepted manuscript
(doi:10.1088/1741-4326/aec6b0), Fig. 11(a, b) at 600 K and Fig. 10(b) at 700 K,
to about +-0.03; their own error bars (interquartile ranges) are of the same
order. Each profile runs from one boundary dot to the next across one grain.
Positions are along their red arrows, which cross the grains obliquely, so the
grain widths here overstate the perpendicular ones by an unknown factor.

The fit has two stages, because only the first costs FESTIM solves:

* ``scan``: on a grid of the two transport parameters, ``D_gb / D`` (with
  ``delta = 1 nm`` and ``s = 1``; only ``delta s D_gb`` and ``k / s`` matter)
  and the exchange coefficient ``k``, solve each profile's geometry three
  times: both front mouths closed, and each fed alone. Everything is linear in
  the mouth concentrations, so these give the dots for any pair of feeds
  ``m``. The dot inventories are cached.
* ``fit``: from the cache, by brute force, the feeds and the indicator's
  response, a detection limit and a saturation (:func:`hydrogenography.response`).
  At 600 K one feed is shared by every boundary of both foils (25 um, AR 1.6,
  with in-grain signal; 50 um, AR 1.1, without). At 700 K each 50 um profile
  has its own left and right feed, since the halos are one-sided. Detection
  and saturation are shared at each temperature, not across them, since the
  WO3 response changes with temperature.

The bulk flux is not fitted: it is pinned to Kapser et al. (2018), see the
cross-section example. Model and data are both normalised to the left
boundary dot, which removes the unknown image maximum; the in-grain dots and
the right boundary dot, where the profile shows one, enter the misfit with an
uncertainty of 0.05.

    python examples/delic2026_fit.py scan --T 600 700   # ~30 min on 2 cores
    python examples/delic2026_fit.py fit                # seconds
"""

import argparse
import itertools
import json
import time
from dataclasses import replace

import numpy as np
from delic2026_crosssection import (
    EXPOSURE,
    OUTPUT_DIR,
    UM,
    Case,
    solve,
)

from festim_microstructure.postprocessing import hydrogenography as hg

SIGMA = 0.05  # uncertainty of a normalised dot value
FLANK = 40 * UM  # grain on either side of the profiled one
PITCH = 10 * UM

# Read from the figures: boundary dot positions, in-grain dot positions and
# values, and boundary dot values (None where the profile shows none).
PROFILES = {
    "600K-AR1.6": dict(
        T=600, foil=25, gb=(3, 72), gb_value=(0.87, 1.00),
        pos=(13, 23, 33, 43, 53, 63), value=(0.09, 0.44, 0.31, 0.35, 0.27, 0.00),
    ),
    "600K-AR1.1": dict(
        T=600, foil=50, gb=(4, 64), gb_value=(0.90, 0.89),
        pos=(15, 25, 35, 45, 55), value=(0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    "700K-a": dict(
        T=700, foil=50, gb=(3, 62), gb_value=(1.00, 0.90),
        pos=(13, 23, 33, 43, 53), value=(0.37, 0.27, 0.25, 0.15, 0.04),
    ),
    "700K-b": dict(
        T=700, foil=50, gb=(3, 64), gb_value=(0.90, None),
        pos=(14, 24, 34, 44, 54), value=(0.31, 0.31, 0.24, 0.13, 0.08),
    ),
    "700K-c": dict(
        T=700, foil=50, gb=(4, 64), gb_value=(1.00, 0.73),
        pos=(14, 24, 34, 44, 54), value=(0.60, 0.20, 0.05, 0.02, 0.05),
    ),
}  # fmt: skip

D_RATIOS = np.logspace(2, 6, 9)
KS = np.logspace(-7, -2, 6)  # m/s; beyond 1e-2 the network is in equilibrium
DETECTIONS = np.logspace(19, np.log10(3e20), 40)  # D m^-2
# Saturation as a multiple of the detection limit; the last is effectively none.
SATURATIONS = np.concatenate([np.logspace(np.log10(1.25), 2, 12), [1e6]])
FEEDS = np.concatenate([[0.0], np.logspace(-1, 4, 41)])


def names_at(T):
    """The profiles measured at ``T``."""
    return [name for name, p in PROFILES.items() if p["T"] == T]


def geometry(profile):
    """``(boundaries, spacing in dots)`` of the model for one profile.

    The profiled grain is snapped to a whole number of dot pitches so that a
    dot sits on each of its boundaries, as in the data.
    """
    n = round((profile["gb"][1] - profile["gb"][0]) / 10)
    return (FLANK, FLANK + n * PITCH), n


def data_vector(profile):
    """Fitted values over the left boundary dot, and their dot indices.

    The in-grain dots, then the right boundary dot if the profile has one.
    """
    left, right = profile["gb_value"]
    index = [round((p - profile["gb"][0]) / 10) for p in profile["pos"]]
    values = list(profile["value"])
    if right is not None:
        index.append(geometry(profile)[1])
        values.append(right)
    return np.array(values) / left, np.array(index)


def dot_inventories(result, n):
    """Inventory of the ``n + 1`` dots from the left to the right boundary."""
    pattern = hg.IndicatorPattern(offset=FLANK % PITCH)
    centres, signals = hg.dot_signals(
        pattern,
        result["edges"],
        result["flux"],
        mouths=[(row[0], row[2]) for row in result["mouths"]],
        exposure=EXPOSURE,
    )
    first = np.argmin(np.abs(centres - FLANK))
    return signals[first : first + n + 1]


def cache_path(T, h):
    """Where the scan at ``T`` and mesh size ``h`` keeps its dots."""
    return OUTPUT_DIR / f"scan_{T}K_h{h * 1e9:.0f}nm.npz"


def scan(T, h):
    """Solve every grid point at ``T``: closed mouths, then each fed alone.

    Caches, per grid point and profile, the dot inventories of the three runs
    and the D the mouths take in (both linear in the feeds), and the worst
    conservation error.
    """
    names = names_at(T)
    n_dots = max(geometry(PROFILES[name])[1] for name in names) + 1
    shape = (D_RATIOS.size, KS.size, len(names), 3)
    dots = np.full((*shape, n_dots), np.nan)
    intake = np.zeros(shape)
    balance = np.zeros(shape[:3])
    for (i, D_ratio), (j, k) in itertools.product(enumerate(D_RATIOS), enumerate(KS)):
        tic = time.perf_counter()
        for p, name in enumerate(names):
            prof = PROFILES[name]
            boundaries, n = geometry(prof)
            case = Case(
                name=name,
                h=h,
                foil=prof["foil"],
                temperature=float(T),
                D_ratio=D_ratio,
                k=k,
                boundaries=boundaries,
                width=boundaries[1] + FLANK,
            )
            for r, m in enumerate([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]):
                res = solve(replace(case, m=m))
                dots[i, j, p, r, : n + 1] = dot_inventories(res, n)
                intake[i, j, p, r] = res["injection_over_ion_budget"]
                balance[i, j, p] = max(balance[i, j, p], abs(res["balance"]))
        print(
            f"{T} K  D_gb/D {D_ratio:7.1e}  k {k:7.1e}  "
            f"worst balance {balance[i, j].max():.1e}  "
            f"({time.perf_counter() - tic:.0f} s)",
            flush=True,
        )
    # Per unit feed, for the dots and the intake alike.
    dots[..., 1:, :] -= dots[..., :1, :]
    intake[..., 1:] -= intake[..., :1]
    np.savez(
        cache_path(T, h),
        dots=dots,
        intake=intake,
        balance=balance,
        D_ratios=D_RATIOS,
        ks=KS,
        names=np.array(names),
    )


def misfit(inventory, data, index):
    """Chi-squared over every detection limit and saturation.

    ``inventory`` has the dots on its last axis; the result has its other
    axes followed by ``(detection, saturation)``.
    """
    det = DETECTIONS[:, None]
    sat = det * SATURATIONS[None, :]
    inv = inventory[..., None, None, :]
    g = np.clip((inv - det[..., None]) / (sat - det)[..., None], 0.0, 1.0)
    ref = g[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        model = g[..., index] / ref[..., None]
    chi2 = np.sum((model - data) ** 2, axis=-1) / SIGMA**2
    return np.where(ref > 0, chi2, np.inf)


def fit(T, h):
    """Best feeds and indicator response at every grid point, then the best."""
    cache = np.load(cache_path(T, h))
    dots, intake = cache["dots"], cache["intake"]
    names = [str(n) for n in cache["names"]]
    profiles = [PROFILES[name] for name in names]
    best_map = np.full(dots.shape[:2], np.inf)
    best = None
    for i, j in itertools.product(range(dots.shape[0]), range(dots.shape[1])):
        parts = []
        for p, prof in enumerate(profiles):
            _, n = geometry(prof)
            data, index = data_vector(prof)
            base, left, right = dots[i, j, p, :, : n + 1]
            if T == 600:
                # One shared feed: axes (feed, det, sat), summed over profiles.
                inv = base + FEEDS[:, None] * (left + right)
            else:
                inv = base + FEEDS[:, None, None] * left + FEEDS[None, :, None] * right
            parts.append(misfit(inv, data, index))
        if T == 600:
            total = sum(parts)
            a = np.unravel_index(np.argmin(total), total.shape)
            chi2 = total[a]
            feeds = [(FEEDS[a[0]], FEEDS[a[0]])] * len(profiles)
            det, sat = a[1], a[2]
        else:
            # Each profile picks its own feeds for every (det, sat).
            per = [part.reshape(-1, *part.shape[-2:]) for part in parts]
            total = sum(part.min(axis=0) for part in per)
            det, sat = np.unravel_index(np.argmin(total), total.shape)
            chi2 = total[det, sat]
            feeds = []
            for part in per:
                f = np.unravel_index(np.argmin(part[:, det, sat]), parts[0].shape[:2])
                feeds.append((FEEDS[f[0]], FEEDS[f[1]]))
        best_map[i, j] = chi2
        if best is None or chi2 < best["chi2"]:
            budget = [
                intake[i, j, p, 0]
                + feeds[p][0] * intake[i, j, p, 1]
                + feeds[p][1] * intake[i, j, p, 2]
                for p in range(len(profiles))
            ]
            best = dict(
                T=T,
                i=int(i),
                j=int(j),
                D_ratio=float(cache["D_ratios"][i]),
                k=float(cache["ks"][j]),
                chi2=float(chi2),
                n_dots=int(sum(data_vector(prof)[0].size for prof in profiles)),
                detection=float(DETECTIONS[det]),
                saturation=float(DETECTIONS[det] * SATURATIONS[sat]),
                feeds={
                    n: [float(f[0]), float(f[1])]
                    for n, f in zip(names, feeds, strict=True)
                },
                intake_over_ion_budget={
                    n: float(b) for n, b in zip(names, budget, strict=True)
                },
                worst_balance=float(cache["balance"][i, j].max()),
            )
    case = Case(name="best", temperature=float(T), D_ratio=best["D_ratio"], k=best["k"])
    lam, ell = case.lengths()
    best.update(lam_um=lam / UM, ell_um=ell / UM)
    return best, best_map


def model_profile(T, h, best, name):
    """Normalised model dots of one profile at the best fit, from the cache."""
    cache = np.load(cache_path(T, h))
    p = [str(n) for n in cache["names"]].index(name)
    prof = PROFILES[name]
    _, n = geometry(prof)
    base, left, right = cache["dots"][best["i"], best["j"], p, :, : n + 1]
    ml, mr = best["feeds"][name]
    g = hg.response(
        base + ml * left + mr * right, best["detection"], best["saturation"]
    )
    return 10 * np.arange(n + 1), g / g[0]


def report(temps, h):
    """Misfit maps and best-fit profiles against the data."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in PROFILES if PROFILES[n]["T"] in temps]
    rows = 1 + (len(names) + 1) // 2
    fig = plt.figure(figsize=(10, 3.1 * rows))
    grid = fig.add_gridspec(rows, 2)
    summary = {}
    for col, T in enumerate(temps):
        best, chi2 = fit(T, h)
        summary[T] = best
        ax = fig.add_subplot(grid[0, col])
        im = ax.pcolormesh(
            np.log10(D_RATIOS),
            np.log10(KS),
            np.log10(np.minimum(chi2, 1e5)).T,
            cmap="viridis_r",
            shading="nearest",
        )
        fig.colorbar(im, ax=ax, label="log10 chi2")
        ax.plot(np.log10(best["D_ratio"]), np.log10(best["k"]), "r*", ms=14)
        ax.set_xlabel("log10 D_gb / D  (delta = 1 nm)")
        ax.set_ylabel("log10 k (m/s)")
        ax.set_title(
            f"{T} K: best chi2 {best['chi2']:.0f} for {best['n_dots']} dots; "
            f"detection {best['detection']:.1e}, saturation "
            f"{best['saturation']:.1e} D/m2",
            fontsize=8,
        )
    for i, name in enumerate(names):
        prof = PROFILES[name]
        best = summary[prof["T"]]
        x, model = model_profile(prof["T"], h, best, name)
        data, index = data_vector(prof)
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        ax.errorbar(10 * index, data, SIGMA, fmt="s", color="tab:blue", label="Delic")
        ax.plot(x, model, "o-", color="tab:red", label="model")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.set_ylim(-0.1, 1.3)
        ax.set_xlabel("distance from left boundary dot (um)")
        ax.set_ylabel("dot / left boundary dot")
        ml, mr = best["feeds"][name]
        ax.set_title(
            f"{name}: m = ({ml:.3g}, {mr:.3g}); mouth intake "
            f"{best['intake_over_ion_budget'][name]:.2g} x ions on 1 nm",
            fontsize=8,
        )
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_fit.png", dpi=150)
    with open(OUTPUT_DIR / "fit_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["scan", "fit"])
    parser.add_argument("--T", type=int, choices=[600, 700], nargs="*")
    parser.add_argument("--h", type=float, default=0.25e-6, help="mesh size")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temps = args.T or [600, 700]
    if args.stage == "scan":
        for T in temps:
            scan(T, args.h)
    else:
        report([T for T in temps if cache_path(T, args.h).exists()], args.h)


if __name__ == "__main__":
    main()
