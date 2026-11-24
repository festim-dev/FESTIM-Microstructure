"""Run the EBSD import, meshing, and FESTIM checks."""

import json
from pathlib import Path

from mpi4py import MPI

from ._grains import convert
from ._mesh_checks import mesh_diagnostics, unit_name, write_network_png
from ._options import UpxoMeshOptions
from ._topology import EbsdMicrostructure
from ._upxo import mesh_grains, resolve_python

__all__ = ["run_ebsd_pipeline"]


def run_ebsd_pipeline(
    options, workdir: str | Path = "results", force=True
) -> EbsdMicrostructure:
    """Import, mesh, check source geometry/orientations and load FESTIM tags.

    Accepted meshes carry their native EBSD archive and SI metadata. Grain and
    boundary tags are checked through the actual DOLFINx importer, and the
    reconstructed FESTIM network is reported alongside geometry diagnostics.
    The returned microstructure's ``base`` is the path stem of the saved files.
    """
    options.__post_init__()
    if not isinstance(options.mesh, UpxoMeshOptions):
        raise TypeError("EbsdOptions.mesh must be UpxoMeshOptions")
    # Only one rank imports/publishes files; all ranks load the accepted mesh.
    # Propagate preparation failures before entering collective mesh import.
    comm = MPI.COMM_WORLD
    base, failure = None, None
    if comm.rank == 0:
        try:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            executable = resolve_python(options.python)
            imported = convert(
                settings=options.import_settings,
                output=workdir / f"{options.stem}-import.npz",
                force=force,
                python=executable,
                timeout=options.timeout,
                check_images=options.check_images,
            )
            base = mesh_grains(
                imported.archive,
                options.mesh,
                workdir=workdir,
                stem=options.stem,
                unit=options.unit,
                force=force,
                python=executable,
                timeout=options.timeout,
            )
            mesh_diagnostics(
                base, unit_name(options.unit), check_images=options.check_images
            )
        except Exception as exc:
            failure = exc
    base, failure = comm.bcast((base, failure), root=0)
    if failure is not None:
        raise failure
    micro = EbsdMicrostructure.load(base, theta_min=options.theta_min, comm=comm)
    micro.check_orientations()
    if micro.mesh.comm.rank == 0:
        report = dict(
            n_grains=micro.n_grains,
            n_edges=len(micro.edges),
            interior_edges=int(micro.interior_mask.sum()),
            surface_edges=micro.surface_edges,
            triple_junctions=micro.triple_junctions,
            theta_min=options.theta_min,
            orientation_source=f"{base.name}-ebsd.npz",
            extent_m=list(micro.extent),
        )
        Path(f"{base}-festim.json").write_text(json.dumps(report, indent=2) + "\n")
    if options.check_images:
        write_network_png(
            base,
            micro.mesh,
            micro,
            f"{base}-ebsd.npz",
            options.unit,
            unit_name(options.unit),
        )
    return micro
