"""Microstructures for FESTIM: meshes, tagged subdomains, coefficient fields.

The package creates and interprets polycrystals -- Voronoi and EBSD --
converts them into FESTIM subdomains and diffusivity fields, and supplies common
declarations and post-processing for grain-boundary transport. The caller builds
and runs the transport problem with FESTIM. See ``examples/`` for the pattern.

The public API uses ordinary imports and requires the FESTIM/DOLFINx stack.
"""

from . import (
    check,
    ebsd,
    exports,
    fem,
    formats,
    materials,
    meshing,
    microstructure,
    plotting,
    voronoi,
)
from .fem import (
    Grain,
    GrainBoundaryNetwork,
    GrainBoundarySystem,
    GrainSurface,
    grain_boundary_network,
    grain_subdomains,
    grain_surfaces,
)
from .materials import crystal_diffusivity_field
from .meshing.ebsd import EbsdMicrostructure, EbsdOptions, run_ebsd_pipeline
from .meshing.upxo import UpxoMeshOptions
from .microstructure import (
    BoundaryNetwork,
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
    "BoundaryNetwork",
    "BoundaryTopology",
    "EbsdMicrostructure",
    "EbsdOptions",
    "Grain",
    "GrainBoundaryNetwork",
    "GrainBoundarySystem",
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
    "exports",
    "fem",
    "formats",
    "grain_boundary_network",
    "grain_subdomains",
    "grain_surfaces",
    "materials",
    "meshing",
    "microstructure",
    "plotting",
    "run_ebsd_pipeline",
    "voronoi",
]
