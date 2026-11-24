"""Microstructures for FESTIM: meshes, tagged subdomains, coefficient fields.

The package creates and interprets polycrystals -- Voronoi and EBSD --
converts them into FESTIM subdomains and diffusivity fields, and supplies the
grain-to-network exchange terms and post-processing for grain-boundary
transport. FESTIM owns species, reactions, boundary conditions, settings and
solving. See ``examples/`` for the pattern.

The public API uses ordinary imports and requires the FESTIM/DOLFINx stack.
"""

from . import (
    check,
    ebsd,
    exports,
    fem,
    materials,
    microstructure,
    msh4,
    voronoi,
)
from .ebsd import EbsdMicrostructure, EbsdOptions, UpxoMeshOptions, run_ebsd_pipeline
from .fem import (
    Grain,
    GrainBoundaryNetwork,
    GrainSurface,
    exchange,
    grain_boundary_network,
    grain_subdomains,
    grain_surfaces,
)
from .materials import crystal_diffusivity_field
from .microstructure import (
    BoundaryTopology,
    MeshedMicrostructure,
    Microstructure,
    TaggedPolycrystal,
)
from .voronoi import MeshSizing, VoronoiMicrostructure

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"

__all__ = [
    "BoundaryTopology",
    "EbsdMicrostructure",
    "EbsdOptions",
    "Grain",
    "GrainBoundaryNetwork",
    "GrainSurface",
    "MeshSizing",
    "MeshedMicrostructure",
    "Microstructure",
    "TaggedPolycrystal",
    "UpxoMeshOptions",
    "VoronoiMicrostructure",
    "__version__",
    "check",
    "crystal_diffusivity_field",
    "ebsd",
    "exchange",
    "exports",
    "fem",
    "grain_boundary_network",
    "grain_subdomains",
    "grain_surfaces",
    "materials",
    "microstructure",
    "msh4",
    "run_ebsd_pipeline",
    "voronoi",
]
