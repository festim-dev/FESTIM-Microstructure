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
on a line of dots across a cross-section (:func:`dot_signals`) or on the
hexagonal array itself over a surface (:func:`hexagonal_centres`,
:func:`disc_signals`). For the line:

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

What the camera sees is not the inventory itself: :func:`response` maps it to
a grayscale change that is zero below the indicator's detection limit and
saturates when the layer is full, and :func:`normalise` rescales one image to
``[0, 1]`` as the paper does. Classify the normalised response, not the raw
inventory, when comparing with the paper's binarised maps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "IndicatorPattern",
    "chord_lengths",
    "classify",
    "count",
    "disc_signals",
    "dot_signals",
    "hexagonal_centres",
    "normalise",
    "on_boundary",
    "response",
]

#: Detection limit of the yttrium indicator, Delic et al. 2026, Sect. 2.1.
DETECTION_LIMIT = 1e20  # D m^-2
#: Capacity of their 50 nm yttrium layer (full conversion to YH2), Sect. 2.1.
SATURATION = 3e21  # D m^-2


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


def response(inventory, detection=DETECTION_LIMIT, saturation=SATURATION):
    """Grayscale change of a dot holding ``inventory`` particles per m^2.

    Linear between the detection limit and saturation, as reported for
    yttrium (Delic et al. 2026, Sect. 2.1, after their ref. [20]), and 0 below
    the limit: a dot that has not reached it looks like the background. In
    units of the saturated contrast, so between 0 and 1. The limit is the
    least known number in the comparison; for the WO3 indicator it is only
    known to be of the same order as for yttrium.
    """
    if not 0 <= detection < saturation:
        raise ValueError("need 0 <= detection < saturation")
    g = (np.asarray(inventory, dtype=float) - detection) / (saturation - detection)
    return np.clip(g, 0.0, 1.0)


def normalise(values):
    """Min-max normalisation over one image, as the paper does per scan."""
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.zeros(values.shape)
    return (values - values.min()) / span


def hexagonal_centres(pattern, extent):
    """Centres of the hexagonal dot array lying wholly inside a rectangle.

    ``extent = ((x_min, x_max), (y_min, y_max))``. Rows are ``pitch * sqrt(3)
    / 2`` apart and every other row is shifted by half a pitch, as on the mask
    of Delic et al. (2026). Returns an ``(n, 2)`` array.
    """
    (x0, x1), (y0, y1) = extent
    half = pattern.diameter / 2
    dy = pattern.pitch * np.sqrt(3) / 2
    rows = np.arange(np.ceil((y0 + half) / dy), np.floor((y1 - half) / dy) + 1)
    centres = []
    for row in rows:
        shifted = replace(
            pattern, offset=pattern.offset + (row % 2) * pattern.pitch / 2
        )
        xs = shifted.centres((x0, x1))
        centres.append(np.column_stack([xs, np.full(xs.size, row * dy)]))
    return np.concatenate(centres) if centres else np.empty((0, 2))


def chord_lengths(centres, radius, segments):
    """Length of each segment inside each disc, as an ``(n_discs, n_segments)`` array.

    ``segments`` has shape ``(n, 2, 2)``: two end points, two coordinates.
    """
    centres = np.asarray(centres, dtype=float)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    p0, d = segments[:, 0], segments[:, 1] - segments[:, 0]
    length = np.linalg.norm(d, axis=1)
    f = p0[None, :, :] - centres[:, None, :]
    a = np.where(length > 0, length**2, 1.0)[None, :]
    b = 2 * np.einsum("dsk,sk->ds", f, d)
    c = np.einsum("dsk,dsk->ds", f, f) - radius**2
    disc = b**2 - 4 * a * c
    root = np.sqrt(np.clip(disc, 0.0, None))
    t0 = np.clip((-b - root) / (2 * a), 0.0, 1.0)
    t1 = np.clip((-b + root) / (2 * a), 0.0, 1.0)
    return np.where(disc > 0, (t1 - t0) * length[None, :], 0.0)


def disc_signals(
    pattern,
    centres,
    flux_at,
    segments=(),
    line_flux=(),
    exposure=96 * 3600.0,
    collect="footprint",
    samples=61,
):
    """Hydrogen collected per unit dot area by round dots on a surface.

    The two-dimensional counterpart of :func:`dot_signals`. ``flux_at`` maps an
    ``(n, 2)`` array of points on the back surface to the lattice flux there,
    in particles m^-2 s^-1; it is averaged over ``samples`` equal-area points
    of each dot. The grain-boundary network leaves the surface along
    ``segments`` (shape ``(n, 2, 2)``), each carrying ``line_flux`` particles
    m^-1 s^-1 per unit length of trace; a dot collects what leaves along the
    chord it covers, computed exactly.
    """
    if collect not in ("footprint", "apex"):
        raise ValueError("collect must be 'footprint' or 'apex'")
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    radius = pattern.diameter / 2 - (pattern.rim if collect == "apex" else 0.0)
    # Vogel's sunflower: equal areas per point.
    i = np.arange(samples)
    r = radius * np.sqrt((i + 0.5) / samples)
    theta = i * np.pi * (3 - np.sqrt(5))
    offsets = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    points = (centres[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    lattice = np.asarray(flux_at(points), dtype=float).reshape(centres.shape[0], -1)
    collected = lattice.mean(axis=1)
    if len(segments):
        chords = chord_lengths(centres, radius, segments)
        collected = collected + chords @ np.asarray(line_flux, dtype=float) / (
            np.pi * radius**2
        )
    return exposure * collected


def on_boundary(pattern, centres, boundaries):
    """True for every dot whose full footprint touches a grain boundary.

    This is how the paper counts a dot as "on a grain boundary": from the
    micrograph, whatever the optical analysis window does.
    """
    centres = np.asarray(centres, dtype=float)
    if centres.ndim == 2:
        # Dots on a surface, boundaries as trace segments of shape (n, 2, 2).
        if len(boundaries) == 0:
            return np.zeros(centres.shape[0], dtype=bool)
        chords = chord_lengths(centres, pattern.diameter / 2, boundaries)
        return chords.max(axis=1) > 0
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
        return normalise(signals) > threshold
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
