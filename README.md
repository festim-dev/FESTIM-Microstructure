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
| FESTIM subdomains and fields | Tagged grains, the codimension-one network, per-grain surface patches, lattice and boundary diffusivity fields, and the grain-to-network exchange terms |
| Post-processing | Grain-plus-network inventories and flux averages, network topology checks |
| Examples | Voronoi and EBSD transport, homogenisation, and paper reproductions. [Runnable examples](examples/README.md) |

The transport model itself is FESTIM's. Every grain is a
`festim.VolumeSubdomain` with a `festim.Species` of its own, the grain-boundary
network is one codimension-one `VolumeSubdomain` carrying one species, and each
grain exchanges with it through one `ParticleFluxBC`/`ParticleSource` pair.
`fm.exchange` builds those pairs, the one coupling FESTIM does not declare
itself. FESTIM owns species, reactions, boundary conditions, settings, solving
and standard exports.

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
file. The active interpreter runs both the solver and EBSD mesher. Set
`FM_EBSD_PYTHON` to run the EBSD worker in a different interpreter.

CI creates this environment from scratch and runs the complete test suite;
missing dependencies or skipped tests fail that job. To check EBSD mesh
reconstruction, boundary exchange and inventory conservation locally, run:

```bash
FM_REQUIRE_FULL_STACK=1 python -m pytest -q test/ebsd/test_pipeline.py
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
  dolfinx                0.11.0
  festim                 2.2
  gmsh (python-gmsh)     4.15.2
  ...
  EBSD interpreter       /home/you/miniforge3/envs/festim-microstructure/bin/python
  EBSD worker            OK (runtime imports verified)
    upxo                 1.5.0
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
conda env update -n festim-microstructure -f environment.yml
conda activate festim-microstructure
unset FM_UPXO_PYTHON
python -m pip install --no-deps -e .
python -m pip check
fm-check --ebsd
FM_REQUIRE_FULL_STACK=1 python -m pytest -q
```

This also upgrades an existing Python 3.12 `festim-microstructure` environment
to the current Python 3.13 stack. If you previously used a differently named
environment, create the standard environment using the installation commands
above and select it in your editor. Use `FM_EBSD_PYTHON` for a separate EBSD
interpreter and clear the old override with `unset FM_UPXO_PYTHON`.

Import and mesh cache identities probe the selected worker before reuse. They
record its Python executable/version, all installed distribution versions,
source provenance and installation-record fingerprints. Changing
`FM_EBSD_PYTHON` or reinstalling dependencies invalidates the matching cache.
The probe also rejects incompatible UPXO/DefDAP before a cached result is used.
Manual edits to installed dependency files or editable dependency source trees
are not fingerprinted: use immutable dependency installations, or `force=True`
after such edits. Project worker/importer code is fingerprinted separately.

`fm-check` probes an available EBSD worker and reports dependency versions and
runtime-import failures. `fm-check --ebsd` also fails when no worker is configured;
use it before running EBSD examples. The worker probe uses the selected
interpreter and does not require UPXO in a separate solver interpreter.

### Version coupling

FESTIM is pinned in `environment.yml` alongside a DOLFINx version range.
The FESTIM 2.2 stack was verified with DOLFINx 0.11.0. FESTIM moved to
DOLFINx 0.11 and replaced `adios4dolfinx` with `io4dolfinx` in
[festim-dev/FESTIM#1173](https://github.com/festim-dev/FESTIM/pull/1173),
merged one day before the 2.1 release. An unpinned DOLFINx will drift out from
under a pinned FESTIM and fail at import or at assembly.

If `pip` attempts to build `scifem` from source, check that the active Conda
environment supplies `scifem>=0.20.0`. Once Conda supplies FESTIM's
dependencies, you can install the pinned wheel directly.

```bash
python -m pip install --no-deps festim==2.2
python -m pip check
```

## Repository layout

```
FESTIM-Microstructure/
├── docs/
├── examples/
├── src/festim_microstructure/
│   ├── microstructure.py       # shared contract and tagged-mesh adapter
│   ├── materials.py            # lattice diffusivity fields
│   ├── ebsd/                   # measured microstructures
│   ├── voronoi/                # tessellation and meshing
│   ├── fem/                    # FESTIM subdomains and grain-to-network coupling
│   └── postprocessing/         # measures, averages, and VTX output
├── test/
└── tools/
```

Find workflow guides, runnable cases, regression tests, and evaluation utilities
in [docs/](docs/), [examples/](examples/), [test/](test/), and [tools/](tools/).
Run the examples, tools, `fm-check`, and tests in the installed
`festim-microstructure` environment.

## Usage

The package supplies subdomains, coefficient fields and the grain-to-network
exchange terms. Species, boundary conditions, traps, reactions, settings and
the problem itself are declared with FESTIM.

```python
import festim as F
import numpy as np

import festim_microstructure as fm

micro = fm.VoronoiMicrostructure.create(size=100e-6, n_seeds=64)
y_top = micro.size
NETWORK_ID, SURFACE_ID_0 = 1_000_000, 2_000_000  # above every grain id
delta = 1e-9  # grain-boundary slab width, m

# Subdomains and coefficient fields from the package.
D_lattice, _ = fm.crystal_diffusivity_field(micro, D_bulk=1e-11)
grains = fm.grain_subdomains(micro, F.Material(D=D_lattice))
network = fm.grain_boundary_network(
    NETWORK_ID, micro, F.Material(D_0=2.85e-7, E_D=0.12)
)

# Species from FESTIM: one lattice species per grain and one on the network.
H = {g.id: F.Species(f"H_{g.id}", subdomains=[g]) for g in grains}
H_gb = F.Species("H_gb", subdomains=[network])

# Exchange k (c_grain - c_gb) through one ParticleFluxBC/ParticleSource pair
# per grain; the network side is per unit slab width.
sources, bcs = fm.exchange(H.values(), H_gb, network, width=delta, k=3.0)

# A charged surface: the patch each grain owns and the network's mouths on it.
patches, mouths = fm.grain_surfaces(
    micro.mesh, grains, lambda x: np.isclose(x[1], y_top), SURFACE_ID_0
)
bcs += [
    F.FixedConcentrationBC(subdomain=p, value=1.0, species=H[p.grain_id])
    for p in patches
]
bcs.append(F.FixedConcentrationBC(subdomain=mouths, value=1.0, species=H_gb))

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(micro.mesh),
    subdomains=[*grains, network, *patches, mouths],
    species=[*H.values(), H_gb],
    sources=sources,
    boundary_conditions=bcs,
    temperature=600.0,
    settings=F.Settings(atol=1e-25, rtol=1e-10, transient=False),
)
model.initialise()
fm.tune_direct_solver(model)  # MUMPS workspace for many-subdomain problems
model.run()
total = fm.postprocessing.averages.inventory(grains, H.values(), network, H_gb, delta)
```

Any FESTIM boundary condition goes on a patch the same way, with
`species=H[p.grain_id]`; leave out `mouths` to charge only the grains. Number a
second surface from `mouths.id + 1`. `NETWORK_ID` must differ from every volume
*and* surface id, because FESTIM stores a manifold in the facet tags. Pass
`flux=lambda c_g, c_n: ...` to `fm.exchange` in place of `k` for another
exchange law, and call it once per species pair for several isotopes.

Traps and reactions are FESTIM declarations on each grain. Add them before
`model.initialise()`, for example:

```python
trapped = {g.id: F.Species(f"T_{g.id}", mobile=False, subdomains=[g]) for g in grains}
model.species += list(trapped.values())
model.reactions = [
    F.Reaction(
        reactant=[H[g.id], F.ImplicitSpecies(n=1e25, others=[trapped[g.id]])],
        product=[trapped[g.id]],
        k_0=1e-17,
        E_k=0.12,
        p_0=1e13,
        E_p=1.0,
        volume=g,
    )
    for g in grains
]
```

For a 3D Voronoi mesh, pass `dim=3` to `create`. Elongate grains with
`aspect=3, aspect_axis=2` along z. `fm.TaggedMicrostructure` adapts other
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
to `fm.EbsdMicrostructure.load()`. For tagged networks, pass
`diffusivity_by_entity=values` to `fm.grain_boundary_network()` with `None`
for the material. Supply one diffusivity per boundary in tag order.
The EBSD workflow guide covers mesh settings, saved files, and cache behavior.
[EBSD workflow guide](docs/ebsd_workflow.md)

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
