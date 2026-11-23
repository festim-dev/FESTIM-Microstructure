"""Lattice diffusivity tensors and fields for polycrystals.

Material parameters are FESTIM's business: a grain-boundary material is an
ordinary ``festim.Material(D_0=..., E_D=...)`` and a grain material wraps the
lattice tensor field built here as ``festim.Material(D=field)``. The network
subdomain handles diffusivity that differs from one boundary to the next.
See ``diffusivity_by_entity`` on
:class:`~festim_microstructure.fem.subdomains.GrainBoundaryNetwork`.
"""

from __future__ import annotations

from typing import cast

import dolfinx
import numpy as np

from .microstructure import PlanarLatticeOrientations

__all__ = [
    "crystal_diffusivity_field",
    "crystal_tensor",
    "fill_crystal_diffusivity_field",
]


def crystal_tensor(D_bulk, orientation, crystal_anisotropy=1.0):
    """Return the 2D lattice diffusivity tensor of one grain, in the specimen frame.

    ``crystal_anisotropy`` is the ratio of the two principal diffusivities. It is
    set by the metal's lattice: 1 for a cubic metal, where Neumann's principle
    forbids a second-rank property from depending on grain orientation, and
    possibly larger for HCP or tetragonal metals. ``orientation`` is the angle
    from the crystal frame to the specimen frame, in radians.
    """
    root = np.sqrt(crystal_anisotropy)
    principal = np.diag([D_bulk * root, D_bulk / root])
    c, s = np.cos(orientation), np.sin(orientation)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ principal @ rotation.T


def crystal_diffusivity_field(micro, D_bulk, crystal_anisotropy=1.0):
    """Create a parent-mesh DG0 tensor field with one lattice tensor per grain.

    ``micro`` supplies planar ``orientations`` in radians when
    ``crystal_anisotropy`` differs from one. Isotropic fields use ``D_bulk * I``.
    Returns the field and the grain-id-to-tensor mapping. Hand the field to
    every grain through ``festim.Material(D=field)``.
    """
    mesh = micro.mesh
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))
    # UFL's inherited __new__ also advertises Cofunction for dual spaces.
    D = cast(dolfinx.fem.Function, dolfinx.fem.Function(V, name="D_lattice"))
    return D, fill_crystal_diffusivity_field(D, micro, D_bulk, crystal_anisotropy)


def fill_crystal_diffusivity_field(D, micro, D_bulk, crystal_anisotropy=1.0):
    """Update an existing lattice field in place and return its grain-to-tensor
    mapping. Coefficient sweeps refill the field this way rather than rebuild
    it, so the compiled forms are reused."""
    V = D.function_space
    gdim = V.mesh.geometry.dim
    isotropic = crystal_anisotropy == 1.0
    orientations = (
        micro.orientations
        if not isotropic and isinstance(micro, PlanarLatticeOrientations)
        else None
    )
    if not isotropic and orientations is None:
        raise ValueError("anisotropic lattice diffusivity requires planar orientations")

    isotropic_tensor = np.eye(gdim) * D_bulk
    tensors = {}
    n = gdim * gdim
    for grain_id in micro.grain_ids:
        tensor = isotropic_tensor.copy()
        if orientations is not None:
            tensor[:2, :2] = crystal_tensor(
                D_bulk, orientations[grain_id - 1], crystal_anisotropy
            )
        tensors[grain_id] = tensor
        cells = micro.cell_tags.find(grain_id)
        dofs = V.dofmap.list[cells].reshape(-1)
        flat = tensor.reshape(-1)
        for component in range(n):
            D.x.array[n * dofs + component] = flat[component]
    D.x.scatter_forward()
    return tensors
