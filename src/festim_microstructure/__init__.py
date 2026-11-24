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
    fem,
    materials,
    microstructure,
    msh4,
    postprocessing,
    voronoi,
)
from .ebsd import EbsdMicrostructure, EbsdOptions, UpxoMeshOptions, run_ebsd_pipeline
from .fem import (
    Grain,
    GrainBoundaryNetwork,
    GrainBoundarySystem,
    GrainSurface,
    grain_boundary_network,
    grain_subdomains,
    grain_surfaces,
    tune_direct_solver,
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
    "fem",
    "grain_boundary_network",
    "grain_subdomains",
    "grain_surfaces",
    "materials",
    "microstructure",
    "msh4",
    "postprocessing",
    "run_ebsd_pipeline",
    "tune_direct_solver",
    "voronoi",
]
