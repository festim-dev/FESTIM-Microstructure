"""Fit the fraction of active boundaries against temperature (Delic et al. 2026).

Their Table 2 counts, in the 50 um foil (AR 1.1), the boundary dots that
discolour after 96 h at each temperature: 6.6 % at 300 K, rising to 45 % at
600 K, with no in-grain signal below 630 K. The line-profile fit at 600 K
(``delic2026_fit.py``) found closed front mouths and strongly coupled
boundaries. This script asks what spread of boundary conductances turns that
into the measured fractions, and whether the result squares with the
temperature-independent permeation Kapser et al. (2018) measured through
25 um foils in the same device.

With closed, strongly coupled mouths (``m = 0``, ``k`` large, so ``ell`` well
below a grain) and a bulk flux that does not depend on temperature, the
inventory of a boundary dot depends on temperature only through the
boundary conductances ``lam = delta s D_gb / D``. Three stages:

* ``response``: random 50 um sections whose boundaries get log-uniform
  ``lam`` from 0.1 to 1000 um; the inventory of a dot centred on each
  boundary, against its ``lam``. A boundary counts as active above the
  threshold the 600 K fit implies (detection limit plus 10 % of the range to
  saturation, the paper's relative threshold when the brightest dots
  saturate); a logistic in ``ln lam`` summarises the result, geometric scatter
  included.
* ``fit``: every boundary has its own activation energy ``dE``, normal with
  mean ``mu`` and spread ``sigma`` across boundaries, and ``ln lam = a -
  dE / kT``. The response turns out to be a sharp threshold in ``lam``, so the
  active fraction is ``Phi(alpha T - beta)``, or ``p Phi(alpha T - beta)``
  when only a fraction ``p`` of the boundaries can carry D at all. The counts
  from 300 K to 600 K fix ``alpha``, ``beta`` (and ``p``) by binomial
  likelihood, but not ``a``, ``mu`` and ``sigma`` separately: those are pinned
  by the conductance the line-profile fit gives the active boundaries at
  600 K. Above 600 K halos appear, which closed mouths cannot make, so those
  points are shown, not fitted.
* ``check``: forward runs with ``lam`` drawn from the fit, through the full
  indicator (dot grid, response, per-image normalisation, 10 % threshold),
  against Table 2; and 25 um sections at 300 K and 450 K, against Kapser.

    python examples/delic2026_temperature.py response   # ~5 min
    python examples/delic2026_temperature.py fit
    python examples/delic2026_temperature.py check      # ~5 min
"""

import argparse
import itertools
import json
from dataclasses import replace

import numpy as np
from delic2026_crosssection import (
    EXPOSURE,
    FOILS,
    OUTPUT_DIR,
    UM,
    Case,
    boundary_positions,
    read_dots,
    solve,
)
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from festim_microstructure.postprocessing import hydrogenography as hg

K_B = 8.617333e-5  # eV/K
DELTA = 1e-9  # m, as in the other scripts; lam = DELTA * D_gb / D

# Delic et al. 2026, Table 2: T, discoloured boundary dots, boundary dots.
TABLE2 = [
    (300, 70, 1060),
    (450, 271, 772),
    (500, 375, 1041),
    (600, 402, 950),
    (600, 634, 1412),
    (630, 717, 1273),
    (660, 687, 1347),
    (700, 937, 1349),
]
FITTED = [row for row in TABLE2 if row[0] <= 600]
KAPSER_FLUX = 1.7e14  # D m^-2 s^-1, 24.5 um, 300 K and 450 K alike

WIDTH = 500 * UM
LAM_RANGE = (0.1 * UM, 1000 * UM)


def indicator():
    """``(detection, saturation, threshold)`` from the 600 K profile fit."""
    path = OUTPUT_DIR / "fit_summary.json"
    with open(path) as fh:
        best = json.load(fh)["600"]
    det, sat = best["detection"], best["saturation"]
    return det, sat, det + 0.1 * (sat - det)


def base_case(foil=50, h=0.25e-6):
    """Closed, strongly coupled mouths, as fitted at 600 K."""
    return Case(name="T-series", foil=foil, h=h, width=WIDTH, k=1e-2, m=0.0)


def centred_dots(result):
    """Inventory of a dot centred on each boundary, in D m^-2."""
    out = []
    for xb in result["boundaries"]:
        pattern = hg.IndicatorPattern(offset=xb % hg.IndicatorPattern().pitch)
        centres, signals = hg.dot_signals(
            pattern,
            result["edges"],
            result["flux"],
            mouths=[(row[0], row[2]) for row in result["mouths"]],
            exposure=EXPOSURE,
        )
        out.append(signals[np.argmin(np.abs(centres - xb))])
    return np.array(out)


def response(n_sections, h):
    """Sample boundary-dot inventories against log-uniform conductances."""
    rng = np.random.default_rng(2026)
    lams, invs, widths = [], [], []
    for seed in range(n_sections):
        case = replace(base_case(h=h), seed=seed)
        _, mean, std = FOILS[case.foil]
        n = boundary_positions(case.width, mean, std, case.h, seed).size
        lam = np.exp(rng.uniform(*np.log(LAM_RANGE), n))
        res = solve(replace(case, D_ratio=tuple(lam / DELTA)))
        b = res["boundaries"]
        gaps = np.diff(np.concatenate([[0.0], b, [res["width"]]]))
        lams.append(lam)
        invs.append(centred_dots(res))
        widths.append(np.minimum(gaps[:-1], gaps[1:]))
        print(
            f"section {seed}: {n} boundaries, balance {res['balance']:.1e}, "
            f"bulk inventory {res['J_bulk'] * EXPOSURE:.2e}",
            flush=True,
        )
    np.savez(
        OUTPUT_DIR / "tseries_response.npz",
        lam=np.concatenate(lams),
        inventory=np.concatenate(invs),
        narrower_grain=np.concatenate(widths),
    )


def logistic_fit(lam, active):
    """``(c, w)`` of ``P(active) = expit((ln lam - c) / w)``, by likelihood."""
    x = np.log(lam / UM)

    def nll(p):
        """Negative log-likelihood of the active flags."""
        q = np.clip(expit((x - p[0]) / np.exp(p[1])), 1e-12, 1 - 1e-12)
        return -np.sum(active * np.log(q) + (~active) * np.log(1 - q))

    sol = minimize(nll, x0=[np.median(x), 0.0], method="Nelder-Mead")
    return sol.x[0], np.exp(sol.x[1])


def fraction(T, alpha, beta, p=1.0):
    """Active fraction at ``T``: ``p Phi(alpha T - beta)``.

    With ``ln lam = a - dE / kT`` and ``dE`` normal (``mu``, ``sigma``) over the
    boundaries that can carry D at all (a fraction ``p``), a boundary is active
    when ``ln lam > c``, so the fraction is ``p Phi(((a - c) k T - mu) /
    sigma)``: only ``alpha = (a - c) k / sigma`` and ``beta = mu / sigma`` can
    be read from the counts.
    """
    return p * norm.cdf(alpha * np.asarray(T, dtype=float) - beta)


def fit_counts(capable):
    """Binomial fit of the counts up to 600 K, with or without ``p``."""
    T, k, n = (np.array(col, dtype=float) for col in zip(*FITTED, strict=True))

    def unpack(x):
        """``(alpha, beta, p)`` from unconstrained parameters."""
        return x[0], x[1], expit(x[2]) if capable else 1.0

    def nll(x):
        """Binomial negative log-likelihood."""
        f = np.clip(fraction(T, *unpack(x)), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(f) + (n - k) * np.log(1 - f))

    starts = itertools.product([0.005, 0.02, 0.05], [1.0, 3.0, 10.0], [-1.0, 1.0])
    sols = [minimize(nll, x0=x0, method="Nelder-Mead") for x0 in starts]
    sol = min(sols, key=lambda s: s.fun)
    alpha, beta, p = unpack(sol.x)
    pred = fraction(T, alpha, beta, p)
    chi2 = float(np.sum((k - n * pred) ** 2 / k))  # the paper's Poisson errors
    return dict(
        alpha=float(alpha),
        beta=float(beta),
        p=float(p),
        chi2=chi2,
        dof=len(FITTED) - (3 if capable else 2),
        T50=float(beta / alpha),
        T10_T90=float(2 * norm.ppf(0.9) / alpha),
        sigma_over_mu=float(1 / beta),
        predicted={f"{t:.0f}": float(f) for t, f in zip(T, pred, strict=True)},
    )


def distribution(counts, c, lam_active_600):
    """``(a, mu, sigma)`` in eV, pinned by the 600 K line-profile fit.

    The counts fix ``alpha`` and ``beta``; the third number is the median
    ``lam`` of the active (capable and above threshold) boundaries at 600 K,
    which the profile fit puts at ``lam_active_600``.
    """
    u = -(counts["alpha"] * 600 - counts["beta"])  # (c - m) / s at 600 K
    v = norm.ppf(1 - 0.5 * (1 - norm.cdf(u)))  # conditional median, in s
    s600 = (np.log(lam_active_600 / UM) - c) / (v - u)
    sigma = s600 * K_B * 600
    return c + counts["alpha"] * sigma / K_B, counts["beta"] * sigma, sigma


def fit(lam_active_600=None):
    """Response threshold, the two count models, and the pinned distribution."""
    data = np.load(OUTPUT_DIR / "tseries_response.npz")
    det, sat, threshold = indicator()
    active = data["inventory"] > threshold
    c, w = logistic_fit(data["lam"], active)
    if lam_active_600 is None:
        with open(OUTPUT_DIR / "fit_summary.json") as fh:
            lam_active_600 = json.load(fh)["600"]["lam_um"] * UM
    models = {"single": fit_counts(False), "capable": fit_counts(True)}
    best = min(models, key=lambda m: models[m]["chi2"] / max(models[m]["dof"], 1))
    a, mu, sigma = distribution(models[best], c, lam_active_600)
    out = dict(
        detection=det,
        saturation=sat,
        threshold=threshold,
        logistic_c_ln_um=c,
        logistic_w=w,
        lam_at_half_um=float(np.exp(c)),
        response_samples=int(active.size),
        response_active=int(active.sum()),
        models=models,
        chosen=best,
        lam_active_600_um=lam_active_600 / UM,
        a=float(a),
        mu_eV=float(mu),
        sigma_eV=float(sigma),
        beyond_600={
            f"{row[0]}": dict(
                measured=row[1] / row[2],
                closed_mouths_only=float(
                    fraction(
                        row[0],
                        models[best]["alpha"],
                        models[best]["beta"],
                        models[best]["p"],
                    )
                ),
            )
            for row in TABLE2
            if row[0] > 600
        },
    )
    with open(OUTPUT_DIR / "tseries_fit.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


def draw(rng, T, n, fitted):
    """``n`` boundary conductances at ``T`` from the fitted distribution.

    Boundaries outside the capable fraction get the bottom of the range.
    """
    p = fitted["models"][fitted["chosen"]]["p"]
    dE = rng.normal(fitted["mu_eV"], fitted["sigma_eV"], n)
    lam = np.exp(fitted["a"] - dE / (K_B * T)) * UM
    lam = np.where(rng.uniform(size=n) < p, lam, LAM_RANGE[0])
    return np.clip(lam, *LAM_RANGE)


def check(n_seeds, h):
    """Forward runs through the full indicator, and the Kapser comparison.

    The response is a sharp threshold in ``lam``, so the default mesh of
    0.5 um, which moves dot inventories by about 3 %, is enough here.
    """
    with open(OUTPUT_DIR / "tseries_fit.json") as fh:
        fitted = json.load(fh)
    det, sat = fitted["detection"], fitted["saturation"]
    rng = np.random.default_rng(7)
    pattern = hg.IndicatorPattern()
    forward = {}
    for T in sorted({row[0] for row in FITTED}):
        on, hit = 0, 0
        for seed in range(100, 100 + n_seeds):
            case = replace(base_case(h=h), seed=seed, temperature=float(T))
            _, mean, std = FOILS[case.foil]
            n = boundary_positions(case.width, mean, std, case.h, seed).size
            lam = draw(rng, T, n, fitted)
            res = solve(replace(case, D_ratio=tuple(lam / DELTA)))
            # Each boundary line crosses many rows of the hexagonal dot array
            # at different offsets; ten offsets stand for those rows. Each
            # section is one image: response, per-image normalisation and the
            # 10 % threshold, as the paper does.
            signals, on_gb = [], []
            for offset in np.arange(10) * UM:
                dots = read_dots(res, replace(pattern, offset=offset), detection=det)
                signals.append(dots["signals"])
                on_gb.append(dots["on_gb"])
            signals, on_gb = np.concatenate(signals), np.concatenate(on_gb)
            seen = hg.classify(hg.response(signals, det, sat), "range", 0.1)
            on += int(on_gb.sum())
            hit += int((seen & on_gb).sum())
        forward[T] = dict(model=hit / on, dots_on_gb=on)
        print(f"{T} K: forward {hit / on:.1%} of {on} boundary dots", flush=True)
    kapser = {}
    for T in (300, 450):
        ratios = []
        for seed in range(200, 200 + n_seeds):
            case = replace(base_case(foil=25, h=h), seed=seed, temperature=float(T))
            _, mean, std = FOILS[case.foil]
            n = boundary_positions(case.width, mean, std, case.h, seed).size
            lam = draw(rng, T, n, fitted)
            res = solve(replace(case, D_ratio=tuple(lam / DELTA)))
            w = np.diff(res["edges"])
            total = np.sum(res["flux"] * w) + res["mouths"][:, 2].sum()
            ratios.append(total / (res["J_bulk"] * w.sum()))
        kapser[T] = dict(total_over_bulk=float(np.mean(ratios)), spread=np.std(ratios))
        print(f"{T} K, 25 um: total / bulk {np.mean(ratios):.2f}", flush=True)
    out = dict(
        forward={
            int(T): {**v, "measured": [r[1] / r[2] for r in TABLE2 if r[0] == T]}
            for T, v in forward.items()
        },
        kapser={int(T): v for T, v in kapser.items()},
        kapser_ratio_450_300=kapser[450]["total_over_bulk"]
        / kapser[300]["total_over_bulk"],
    )
    with open(OUTPUT_DIR / "tseries_check.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    plot(fitted, out)
    return out


def plot(fitted, checked):
    """Measured and fitted active fraction against temperature."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(OUTPUT_DIR / "tseries_response.npz")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8))
    lam = data["lam"] / UM
    ax0.scatter(lam, data["inventory"], s=8, c="0.4")
    ax0.axhline(fitted["threshold"], color="tab:red", ls="--", lw=1)
    ax0.axhline(fitted["detection"], color="tab:red", ls=":", lw=1)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("boundary conductance lam (um)")
    ax0.set_ylabel("boundary dot, D m$^{-2}$ in 96 h")
    ax0.set_title(
        f"response, 50 um foil: half active at lam = {fitted['lam_at_half_um']:.1f} um",
        fontsize=9,
    )
    T = np.array([r[0] for r in TABLE2], dtype=float)
    k = np.array([r[1] for r in TABLE2], dtype=float)
    n = np.array([r[2] for r in TABLE2], dtype=float)
    ax1.errorbar(T, 100 * k / n, 100 * np.sqrt(k) / n, fmt="s", label="Delic Table 2")
    Ts = np.linspace(280, 720, 200)
    for name, style in (("single", ":"), ("capable", "-")):
        m = fitted["models"][name]
        label = f"{name}: chi2 {m['chi2']:.0f} / {m['dof']} dof"
        if name == "capable":
            label += f", p = {m['p']:.2f}"
        f = fraction(Ts, m["alpha"], m["beta"], m["p"])
        ax1.plot(Ts, 100 * f, style, color="tab:red", label=label)
    fw = checked["forward"]
    ax1.plot(
        [float(t) for t in fw],
        [100 * v["model"] for v in fw.values()],
        "o",
        mfc="none",
        color="tab:red",
        label="forward, full indicator",
    )
    ax1.axvspan(615, 720, color="0.92", zorder=0)
    ax1.text(620, 5, "halos:\nnot fitted", fontsize=8)
    ax1.set_xlabel("T (K)")
    ax1.set_ylabel("boundary dots discoloured (%)")
    ax1.set_title(
        f"pinned: dE = {fitted['mu_eV']:.2f} +- {fitted['sigma_eV']:.2f} eV; "
        f"25 um total flux, 450 K / 300 K = {checked['kapser_ratio_450_300']:.2f}",
        fontsize=9,
    )
    ax1.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_tseries.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["response", "fit", "check"])
    parser.add_argument("--sections", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--h", type=float, help="mesh size (0.25 um, 0.5 for check)")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "response":
        response(args.sections, args.h or 0.25e-6)
    elif args.stage == "fit":
        fit()
    else:
        check(args.seeds, args.h or 0.5e-6)


if __name__ == "__main__":
    main()
