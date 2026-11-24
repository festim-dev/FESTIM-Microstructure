# FESTIM-Microstructure

⚠️ This repo is in its early stages and is under active development. APIs & interfaces are subject to breaking changes without notice. ⚠️ 

Microstructure-scale hydrogen transport modelling built on
[FESTIM](https://festim.readthedocs.io/).

`festim-microstructure` is a companion package to FESTIM, similar to
other FESTIM add-ons such as `festim-gui` and `festim-niuq`. Import it
alongside FESTIM in your scripts. [festim-gui](https://github.com/festim-dev/festim-gui) · [festim-niuq](https://pypi.org/project/festim-niuq/)

```python
import festim as F
import festim_microstructure as fm
```

## What it provides

| Area | Contents |
| --- | --- |
| Microstructure generation | 2D and 3D Voronoi polycrystals via Gmsh |
| EBSD import | Conversion of EBSD orientation maps into conforming grain meshes |
| FESTIM declarations | Tagged grains, the codimension-one network, per-grain surface patches, lattice and boundary diffusivity fields, species, and exchange terms |
| Post-processing | Grain-plus-network inventories and flux averages, network topology checks |
| Benchmarks | Diaz-Rodriguez baseline, dimensional and non-dimensionalised |

The transport model itself is FESTIM's. Every grain is a
`festim.VolumeSubdomain` with a `festim.Species` of its own, the grain-boundary
network is one codimension-one `VolumeSubdomain` carrying one species, and each
grain exchanges with it through one `ParticleFluxBC`/`ParticleSource` pair.
`GrainBoundarySystem` creates these objects and the caller configures the
FESTIM problem, reactions, settings, and exports.

## Requirements

The package uses a Python 3.13 conda environment for microstructure generation,
EBSD meshing and FESTIM transport. `environment.yml` provides the full stack:

- **conda-forge** supplies DOLFINx, the Gmsh Python API, compiled dependencies
  (`scifem`, `io4dolfinx`) and scientific packages.
- **pip** supplies the pinned FESTIM and EBSD dependencies.

**Platforms:** linux-64, osx-64, osx-arm64. Windows users should work inside
WSL2 for the FEniCSx stack.

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda),
then run:

```bash
git clone https://github.com/festim-dev/FESTIM-Microstructure.git
cd FESTIM-Microstructure
conda env create -f environment.yml
conda activate festim-microstructure
python -m pip install --no-deps -e .
fm-check --ebsd
```

The editable install makes the modules under `src/` available to examples and
scripts. `--no-deps` preserves the dependency stack installed by the environment
file. The active interpreter runs both the solver and EBSD mesher.

CI creates this environment from scratch and runs the complete test suite;
missing dependencies or skipped tests fail that job. To check EBSD mesh
reconstruction, boundary exchange and inventory conservation locally, run:

```bash
FM_REQUIRE_FULL_STACK=1 python -m pytest -q test/test_ebsd_si_mesh.py
```

The environment also includes test and lint tools. For VS Code, select the
`festim-microstructure` Conda environment with **Python: Select Interpreter**.
Running examples requires the editable install in the selected environment.
If imports stay underlined after changing environments, run
**Developer: Reload Window**.

### Verifying the install

```bash
fm-check
```

prints the versions of the Python stack and probes the EBSD
worker. It exits non-zero if dependencies fail to import or the worker fails
its health check. It looks like:

```
python-side stack
  festim_microstructure  0.1.dev15
  ...
  dolfinx                0.10.0
  festim                 2.2rc2
  gmsh (python-gmsh)     4.15.2
  ...
  EBSD interpreter       /home/you/miniforge3/envs/festim-microstructure/bin/python
  EBSD worker            OK (runtime imports verified)
    upxo                 1.3.1
    defdap               0.93.6
```

For MPI, additionally:

```bash
mpirun -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

The `gmsh (python-gmsh)` line matters: on conda-forge the `gmsh` package
installs only the executable and shared library, and the Python module comes
from the separate `python-gmsh` package. `environment.yml` includes the
Python API used by the mesh builders.

### Updating

From the repository checkout, update the standard environment and reinstall:

```bash
conda env update -n festim-microstructure -f environment.yml --prune
conda activate festim-microstructure
python -m pip install --no-deps -e .
fm-check --ebsd
FM_REQUIRE_FULL_STACK=1 python -m pytest -q test/test_ebsd_si_mesh.py
```

This also upgrades an existing Python 3.12 `festim-microstructure` environment
to the current Python 3.13 stack. If you previously used a differently named
environment, create the standard environment using the installation commands
above and select it in your editor. Clear any old `FM_UPXO_PYTHON` override
with `unset FM_UPXO_PYTHON` to use the active interpreter.

Import and mesh cache identities probe the selected worker before reuse. They
record its Python executable/version, all installed distribution versions,
source provenance and installation-record fingerprints. Changing
`FM_UPXO_PYTHON` or reinstalling dependencies invalidates the matching cache.
The probe also rejects incompatible UPXO/DefDAP before a cached result is used.
Manual edits to installed dependency files or editable dependency source trees
are not fingerprinted: use immutable dependency installations, or `force=True`
after such edits. Project worker/importer code is fingerprinted separately.

`fm-check` probes an available EBSD worker and reports dependency versions and
runtime-import failures. `fm-check --ebsd` also fails when no worker is configured;
use it before running EBSD examples. The worker probe uses the selected
interpreter and does not require UPXO in a separate solver interpreter.

### Version coupling

FESTIM is pinned in `environment.yml`, and DOLFINx is pinned to a minor range
alongside it. These two must be bumped together: FESTIM moved to DOLFINx 0.11
and replaced `adios4dolfinx` with `io4dolfinx` in
[festim-dev/FESTIM#1173](https://github.com/festim-dev/FESTIM/pull/1173),
merged one day before the 2.1 release. An unpinned DOLFINx will drift out from
under a pinned FESTIM and fail at import or at assembly.

If `pip` attempts to build `scifem` from source during installation, the
conda-forge `scifem` build is older than the pinned FESTIM requires. Either
relax the FESTIM pin or install it without resolving dependencies:

```bash
pip install --no-deps festim==2.2rc2
pip check
```

## Repository layout

```
FESTIM-Microstructure/
├── environment.yml             # Python 3.13 environment for all workflows
├── pyproject.toml
├── README.md
├── docs/
│   └── gb_homogenisation.md    # the homogenisation study, with figures
├── src/festim_microstructure/
│   ├── __init__.py              # public API with ordinary imports
│   ├── check.py                # fm-check environment diagnostic
│   ├── microstructure.py       # microstructure protocols, TaggedPolycrystal
│   ├── materials.py            # lattice diffusivity tensors and fields
│   ├── msh4.py                  # generic tagged Gmsh mesh readers
│   ├── ebsd/                    # CTF import, UPXO meshing, and EBSD topology
│   │   ├── options.py
│   │   ├── pipeline.py
│   │   ├── topology.py
│   │   ├── orientation.py
│   │   ├── ctf.py
│   │   ├── archive.py
│   │   ├── provenance.py
│   │   ├── grains.py
│   │   ├── upxo.py
│   │   ├── import_checks.py
│   │   ├── mesh_checks.py
│   │   └── plotting.py          # diagnostic raster colours and scale bars
│   ├── voronoi/                 # generated 2D/3D tessellations and Gmsh meshes
│   │   ├── geometry.py
│   │   ├── gmsh_builder.py
│   │   └── polycrystal.py
│   ├── fem/
│   │   ├── _system.py          # common FESTIM transport declarations
│   │   ├── solvers.py          # tune_direct_solver: the MUMPS workspace workaround
│   │   └── subdomains.py       # Grain, GrainBoundaryNetwork, GrainSurface + factories
│   └── exports/
│       ├── measures.py         # submesh measures and network topology checks
│       └── averages.py         # grain+network averages, inventory, fields, VTX output
├── examples/                   # runnable workflows; not part of the library API
│   ├── gb_homogenisation.py    # RVE identification, validation and figures
│   ├── li2022_fig4*.py         # Li et al. (2022) reproduction scripts
│   └── ...                     # usage examples and data/
└── test/                       # pytest in the standard environment
```

Nothing importable sits at the repository root. The package imports its public
API directly and requires the FESTIM/DOLFINx stack. Run the examples, conversion
tools, `fm-check`, and tests from the installed
`festim-microstructure` environment.

## Usage

Build a tagged microstructure and use its grain-boundary system to supply
subdomains, species, and exchange terms to FESTIM.

```python
import festim as F
import numpy as np

import festim_microstructure as fm

micro = fm.VoronoiMicrostructure.create(size=100e-6, n_seeds=64)
y_top = micro.size

D_lattice, _ = fm.crystal_diffusivity_field(micro, D_bulk=1e-11)
system = fm.GrainBoundarySystem(
    micro,
    F.Material(D=D_lattice),
    F.Material(D_0=2.85e-7, E_D=0.12),
    width=1e-9,
)
sources, flux_bcs = system.exchange(k=3.0)
surfaces, surface_bcs = system.fixed_concentration(
    lambda x: np.isclose(x[1], y_top), value=1.0
)

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(micro.mesh),
    subdomains=[*system.subdomains, *surfaces],
    species=system.species,
    sources=sources,
    boundary_conditions=[*flux_bcs, *surface_bcs],
    temperature=600.0,
    settings=F.Settings(atol=1e-25, rtol=1e-10, transient=False),
)
model.initialise()
fm.fem.tune_direct_solver(model)  # MUMPS workspace for many-subdomain problems
model.run()
total = system.inventory()
```

`fixed_concentration` charges the grain patches and the network mouths on
the selected surface. Pass `include_mouths=False` to charge only the grains
on a surface. Call it once per surface when applying different concentrations.
Use `reserved_ids` when combining the system with other FESTIM subdomains
or interfaces.

For a 3D Voronoi mesh, pass `dim=3` to `create`. Elongate grains with
`aspect=3, aspect_axis=2` along z. `fm.TaggedPolycrystal` adapts other
meshes with grain tags. The model uses metres for mesh coordinates and
the grain-boundary width.

### Measured EBSD meshes

```python
options = fm.EbsdOptions(
    import_settings=fm.ebsd.Settings(
        ctf="examples/data/D7 PBF SS316L.ctf",
        min_pixels=15,
        max_mad=1.5,
        allow_error=True,
        crop=(0, 306, 0, 306),
    ),
)
micro = fm.run_ebsd_pipeline(options, workdir="results", force=False)
y_top = micro.extent[1]
```

Use this `micro` and `y_top` in the same transport setup. The path stem of
the saved files is `micro.base`. To open an earlier mesh, pass its path stem
to `fm.EbsdMicrostructure.load()`. For tagged
networks, `diffusivity_by_entity` gives one diffusivity per boundary in
tag order. See the [EBSD workflow guide](docs/ebsd_workflow.md) for mesh
settings, saved files, and cache behavior.

### Examples

```bash
python examples/voronoi_polycrystal_2d.py
python examples/voronoi_polycrystal_3d.py
python examples/ebsd_ctf_to_mesh.py
python examples/ebsd_gb_diffusion.py
python examples/fisher_grain_boundary.py
python examples/gb_homogenisation.py --sizes 2e-6 3e-6 4e-6
python examples/li2022_fig4.py
python examples/li2022_fig4_codim.py
```

See the [example workflows and output locations](examples/README.md).
Use `mpirun` to launch parallel FESTIM runs, for example
`mpirun -n 8 python examples/gb_homogenisation.py`.

## Testing

```bash
pytest
pytest --cov=festim_microstructure --cov-report=term-missing
ruff check src test
ruff format --check src test
```

## Citing

If this package contributes to published work, please cite FESTIM alongside it —
see `CITATION.cff` in the [FESTIM
repository](https://github.com/festim-dev/FESTIM).

## Contributing

Issues and pull requests are welcome. Run `ruff check` and `pytest` before
opening a PR. FESTIM's own [developer
guide](https://festim.readthedocs.io/en/latest/devguide/index.html) is a
reasonable reference for style and review conventions.

## Getting help

For FESTIM questions rather than microstructure questions, use the FESTIM
[Discourse](https://festim.discourse.group/) or Slack channel.

## License

[Apache-2.0](LICENSE).
