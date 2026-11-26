"""Synthetic patterned hydrogenography: what an indicator-dot array would see.

Patterned hydrogenography (Manhard et al. 2023, Nucl. Mater. Energy 36 101498;
Delic et al. 2026, Nucl. Fusion, doi:10.1088/1741-4326/aec6b0) images the
permeation flux leaving the back of a foil through an array of hydrogen-sensitive
dots, 5 um wide on a 10 um pitch. Each dot integrates whatever reaches its
footprint over the whole exposure; the analysis then normalises the dots of one
image and classifies them as discoloured or not.

This module turns a simulated back-surface flux into the same numbers, so a
model is compared with the measurement on the measurement's own terms rather
than through a flux map the experiment never sees. It is pure NumPy and works
on a line of dots across a cross-section:

* the lattice flux is given piecewise constant on segments of the back
  surface, as a P1 solution's gradient is on the facets there, and is
  integrated exactly over each dot;
* the grain-boundary network leaves the back surface at points (its *mouths*);
  each carries a flux ``delta * D_gb * dc_gb/dn`` per unit out-of-plane length
  and is credited to the dot whose footprint contains it.

Two choices are the user's, because the papers leave them open:

* ``collect``: ``"footprint"`` assumes the hydrogen a dot absorbs spreads over
  the dot (the rim exclusion in the paper is then purely optical);
  ``"apex"`` assumes it stays where it entered, so only the analysed apex,
  the footprint minus its rim, counts.
* the classification mode in :func:`classify`; the paper's text ("the lowest
  10 % of the distribution ... non-discoloured") reads as a percentile, but its
  counts are only consistent with a threshold on the normalised range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IndicatorPattern",
    "classify",
    "count",
    "dot_signals",
    "on_boundary",
]

#: Detection limit of the yttrium indicator, Delic et al. 2026, Sect. 2.1.
DETECTION_LIMIT = 1e20  # D m^-2


@dataclass(frozen=True)
class IndicatorPattern:
    """A line of indicator dots across a cross-section.

    Lengths in metres. The defaults are the hexagonal mask of Delic et al.
    (2026), Sect. 2.2.2, cut along a row: 5 um dots on a 10 um pitch, with a
    1.5 um interference rim excluded from the optical analysis.
    """

    diameter: float = 5e-6
    pitch: float = 10e-6
    rim: float = 1.5e-6
    offset: float = 0.0

    def __post_init__(self):
        """Reject patterns whose dots overlap or have no analysable apex."""
        if not 0 < self.diameter <= self.pitch:
            raise ValueError("need 0 < diameter <= pitch")
        if not 0 <= 2 * self.rim < self.diameter:
            raise ValueError("the rim must leave a non-empty apex")

    def centres(self, extent):
        """Centres of every dot lying wholly inside ``extent = (x_min, x_max)``."""
        lo, hi = extent
        half = self.diameter / 2
        first = np.ceil((lo + half - self.offset) / self.pitch)
        last = np.floor((hi - half - self.offset) / self.pitch)
        return self.offset + self.pitch * np.arange(first, last + 1)

    def windows(self, extent, collect="footprint"):
        """``(centres, lo, hi)`` of the collecting window of each dot."""
        if collect not in ("footprint", "apex"):
            raise ValueError("collect must be 'footprint' or 'apex'")
        centres = self.centres(extent)
        half = self.diameter / 2 - (self.rim if collect == "apex" else 0.0)
        return centres, centres - half, centres + half


def _integrate_piecewise(edges, values, lo, hi):
    """Integral of a piecewise-constant function over each ``[lo_i, hi_i]``."""
    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=float)
    if edges.ndim != 1 or values.shape != (edges.size - 1,):
        raise ValueError("need len(edges) == len(values) + 1")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")
    cumulative = np.concatenate([[0.0], np.cumsum(values * np.diff(edges))])

    def primitive(x):
        """Antiderivative at ``x``, extended by zero outside the edges."""
        x = np.clip(x, edges[0], edges[-1])
        i = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, values.size - 1)
        return cumulative[i] + values[i] * (x - edges[i])

    return primitive(np.asarray(hi)) - primitive(np.asarray(lo))


def dot_signals(
    pattern,
    edges,
    flux,
    mouths=(),
    exposure=96 * 3600.0,
    collect="footprint",
    extent=None,
):
    """Hydrogen collected per unit dot area, in particles per square metre.

    ``edges``/``flux`` give the lattice flux leaving the back surface, in
    particles m^-2 s^-1, constant on each ``[edges[i], edges[i+1]]``.
    ``mouths`` is a sequence of ``(x, Q)`` pairs, ``Q`` in particles m^-1 s^-1,
    for the network leaving the surface at ``x``. ``extent`` defaults to the
    span of ``edges``. Returns ``(centres, signals)``.
    """
    if extent is None:
        extent = (edges[0], edges[-1])
    centres, lo, hi = pattern.windows(extent, collect)
    width = hi - lo
    collected = _integrate_piecewise(edges, flux, lo, hi)
    for x, q in mouths:
        inside = (lo <= x) & (x < hi)
        collected = collected + np.where(inside, q, 0.0)
    return centres, exposure * collected / width


def on_boundary(pattern, centres, boundaries):
    """True for every dot whose full footprint touches a grain boundary.

    This is how the paper counts a dot as "on a grain boundary": from the
    micrograph, whatever the optical analysis window does.
    """
    centres = np.asarray(centres, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float).reshape(-1)
    if boundaries.size == 0:
        return np.zeros(centres.shape, dtype=bool)
    distance = np.abs(centres[:, None] - boundaries[None, :]).min(axis=1)
    return distance <= pattern.diameter / 2


def classify(signals, mode="range", threshold=0.1):
    """Binarise dot signals as the paper does, returning a discoloured mask.

    ``mode``:

    * ``"range"``: discoloured when the min-max normalised signal of the
      image exceeds ``threshold`` (the reading consistent with the counts).
    * ``"percentile"``: the literal reading of the text; the lowest
      ``threshold`` fraction of the dots are non-discoloured, the rest are.
    * ``"absolute"``: discoloured above ``threshold`` particles m^-2, e.g.
      :data:`DETECTION_LIMIT`. This needs a calibrated indicator, which the
      paper does not yet have, so it is a model-side bound only.
    """
    signals = np.asarray(signals, dtype=float)
    if mode == "absolute":
        return signals > threshold
    if mode == "percentile":
        return signals > np.quantile(signals, threshold)
    if mode == "range":
        span = signals.max() - signals.min()
        if span == 0:
            return np.zeros(signals.shape, dtype=bool)
        return (signals - signals.min()) / span > threshold
    raise ValueError("mode must be 'range', 'percentile' or 'absolute'")


def count(discoloured, on_gb):
    """The paper's Table 2 quantities for one image.

    Both percentages are normalised to the number of dots on grain
    boundaries, so the on-grain one can exceed 100 % for bulk-dominated
    permeation. Errors are Poisson on the discoloured counts, as in the paper.
    """
    discoloured = np.asarray(discoloured, dtype=bool)
    on_gb = np.asarray(on_gb, dtype=bool)
    n_gb = int(on_gb.sum())
    gb = int((discoloured & on_gb).sum())
    grain = int((discoloured & ~on_gb).sum())
    scale = 100.0 / n_gb if n_gb else np.nan
    return {
        "dots": int(on_gb.size),
        "dots_on_gb": n_gb,
        "discoloured_on_gb": gb,
        "discoloured_on_grain": grain,
        "pct_gb": gb * scale,
        "pct_gb_err": np.sqrt(gb) * scale,
        "pct_grain": grain * scale,
        "pct_grain_err": np.sqrt(grain) * scale,
    }
