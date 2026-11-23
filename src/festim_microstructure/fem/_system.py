"""Common FESTIM declarations for a polycrystal and its boundary network."""

from operator import index

import dolfinx
import festim as F
import numpy as np

from festim_microstructure.exports.averages import inventory

from .subdomains import grain_boundary_network, grain_subdomains, grain_surfaces


def _first_free_id(used_ids, first_id, count):
    """Find or validate a consecutive range of positive meshtag IDs."""
    if first_id is None:
        first_id = 1
        for used in sorted(used_ids):
            if first_id + count <= used:
                break
            first_id = max(first_id, used + 1)
    else:
        first_id = index(first_id)

    last_id = first_id + count - 1
    if first_id < 1 or last_id > np.iinfo(np.int32).max:
        raise ValueError("subdomain IDs must be positive 32-bit integers")
    if any(first_id <= used <= last_id for used in used_ids):
        raise ValueError(
            f"subdomain IDs {first_id} through {last_id} overlap existing IDs"
        )
    return first_id


def _scalar_constant(mesh, value, name):
    if isinstance(value, dolfinx.fem.Constant):
        if value.ufl_shape or value.ufl_domain() != mesh.ufl_domain():
            raise ValueError(f"{name} must be a scalar constant on the parent mesh")
        return value
    if np.ndim(value):
        raise ValueError(f"{name} must be a scalar")
    return dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value))


class GrainBoundarySystem:
    """Build FESTIM subdomains and species for grain-boundary transport.

    Each grain has a lattice species and the codimension-one network has one
    species. ``width`` is the physical slab width in metres. A scalar or a
    DOLFINx constant on the parent mesh supplies that width. A tagged network
    can take diffusivity by entity in place of ``network_material``.

    Use ``reserved_ids`` for IDs of other subdomains and interfaces that the
    problem will include. Pass ``network_id`` to control the network's ID.
    The caller uses the declarations to build and run a FESTIM problem.
    """

    def __init__(
        self,
        micro,
        grain_material,
        network_material=None,
        *,
        width,
        network_id=None,
        reserved_ids=(),
        drop_exterior=None,
        diffusivity_by_entity=None,
    ):
        self.mesh = micro.mesh
        self._width = _scalar_constant(self.mesh, width, "width")
        width_value = self._width.value.item()
        if not np.isfinite(width_value) or width_value <= 0:
            raise ValueError("width must be finite and positive")

        self.grains = grain_subdomains(micro, grain_material)
        self._used_ids = {index(id_) for id_ in reserved_ids}
        self._used_ids.update(grain.id for grain in self.grains)
        network_id = _first_free_id(self._used_ids, network_id, 1)
        self.network = grain_boundary_network(
            network_id,
            micro,
            network_material,
            drop_exterior=drop_exterior,
            diffusivity_by_entity=diffusivity_by_entity,
        )
        self._used_ids.add(network_id)

        self._species_by_grain = {
            grain.id: F.Species(f"c_{grain.id}", subdomains=[grain])
            for grain in self.grains
        }
        self.grain_species = list(self._species_by_grain.values())
        self.c_gb = F.Species("c_gb", subdomains=[self.network])

    @property
    def width(self):
        """The slab-width constant, whose ``value`` can be updated for a sweep."""
        return self._width

    @property
    def subdomains(self):
        """The grain and network subdomains, in declaration order."""
        return [*self.grains, self.network]

    @property
    def species(self):
        """The lattice species followed by the network species."""
        return [*self.grain_species, self.c_gb]

    def exchange(self, k):
        """Return the network sources and lattice flux conditions for exchange.

        ``k`` in metres per second can be a scalar or a DOLFINx constant on the
        parent mesh. The network equation is per unit slab width.
        """
        k = _scalar_constant(self.mesh, k, "k")
        width = self.width
        sources, flux_bcs = [], []
        for c_grain in self.grain_species:
            species = {"c_g": c_grain, "c_n": self.c_gb}
            sources.append(
                F.ParticleSource(
                    value=lambda c_g, c_n: (k / width) * (c_g - c_n),
                    species=self.c_gb,
                    volume=self.network,
                    species_dependent_value=species,
                )
            )
            flux_bcs.append(
                F.ParticleFluxBC(
                    subdomain=self.network,
                    species=c_grain,
                    value=lambda c_g, c_n: k * (c_n - c_g),
                    species_dependent_value=species,
                )
            )
        return sources, flux_bcs

    def fixed_concentration(
        self, locator, value, *, include_mouths=True, first_id=None
    ):
        """Return the surface subdomains and fixed concentration conditions.

        Pass ``include_mouths=False`` for a surface that only charges grains.
        Each call reserves an ID per grain and one for the network mouths.
        ``value`` can be a scalar or a FESTIM concentration callable.
        """
        count = len(self.grains) + 1
        first_id = _first_free_id(self._used_ids, first_id, count)
        patches, mouths = grain_surfaces(self.mesh, self.grains, locator, first_id)
        surface_subdomains = list(patches)
        bcs = [
            F.FixedConcentrationBC(
                subdomain=patch,
                value=value,
                species=self._species_by_grain[patch.grain_id],
            )
            for patch in patches
        ]
        if include_mouths:
            surface_subdomains.append(mouths)
            bcs.append(
                F.FixedConcentrationBC(subdomain=mouths, value=value, species=self.c_gb)
            )
        self._used_ids.update(range(first_id, first_id + count))
        return surface_subdomains, bcs

    def inventory(self, window=None):
        """Integrate the concentrations from FESTIM using the current width."""
        return inventory(
            self.grains,
            self.grain_species,
            self.network,
            self.c_gb,
            self.width.value.item(),
            window=window,
        )
