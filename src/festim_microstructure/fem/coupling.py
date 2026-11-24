"""Grain-to-network exchange, the one coupling FESTIM does not declare itself.

Everything else in a grain/network problem -- species, traps and reactions,
boundary conditions, settings, exports -- is an ordinary FESTIM declaration.
"""

import festim as F

__all__ = ["exchange"]


def _linear(k):
    """The linear exchange law ``k * (c_g - c_n)``."""
    return lambda c_g, c_n: k * (c_g - c_n)


def exchange(grain_species, network_species, network, *, width, k=None, flux=None):
    """Return ``(sources, flux_bcs)`` coupling every grain species to the network.

    Each grain loses ``flux(c_g, c_n)`` per unit boundary area through a
    :class:`festim.ParticleFluxBC` on ``network``, and the network gains it
    through a :class:`festim.ParticleSource`. The network equation is written
    per unit slab width, so its source is ``flux / width``; a wrong factor here
    breaks conservation without raising any error.

    Pass exactly one of ``k``, for the linear law ``k * (c_g - c_n)``, or
    ``flux``, a callable ``flux(c_g, c_n)`` for any other law, such as a
    segregation factor ``lambda c_g, c_n: k * (c_g - c_n / s)``. ``width`` and
    ``k`` may be numbers or DOLFINx constants; a constant can be changed between
    solves without recompiling the forms.

    Call once per pair of species, e.g. once for H and once for D. A rate that
    depends on temperature, time or position needs FESTIM's ``T``, ``t`` or
    ``x`` arguments, so declare that pair directly with FESTIM instead.
    """
    if (k is None) == (flux is None):
        raise TypeError("pass exactly one of k or flux")
    if isinstance(width, int | float) and not width > 0:
        raise ValueError("width must be positive")
    law = _linear(k) if flux is None else flux

    sources, flux_bcs = [], []
    for c_grain in grain_species:
        species = {"c_g": c_grain, "c_n": network_species}
        sources.append(
            F.ParticleSource(
                value=lambda c_g, c_n: law(c_g, c_n) / width,
                species=network_species,
                volume=network,
                species_dependent_value=species,
            )
        )
        flux_bcs.append(
            F.ParticleFluxBC(
                subdomain=network,
                species=c_grain,
                value=lambda c_g, c_n: -law(c_g, c_n),
                species_dependent_value=species,
            )
        )
    return sources, flux_bcs
