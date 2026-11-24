# Examples

Scripts in this directory are complete workflows, parameter studies, and paper
reproductions. They are intentionally not included in the installed
`festim_microstructure` API: their defaults, outputs, figures, and input data
belong to a particular study rather than being stable library contracts.

The grain-boundary examples take their subdomains from `fm.grain_subdomains`,
`fm.grain_boundary_network` and `fm.grain_surfaces`, and their grain-to-network
exchange terms from `fm.exchange`. Species, boundary conditions and the problem
itself are FESTIM declarations in each script.
`gb_homogenisation.py` contains the complete RVE study and shares one
cell-problem declaration across identification and validation.

- `voronoi_polycrystal_2d.py` and `voronoi_polycrystal_3d.py` demonstrate
  short-circuit diffusion.
- `ebsd_ctf_to_mesh.py` and `ebsd_gb_diffusion.py` demonstrate the EBSD path.
- `materialite_slice_to_mesh.py` crops a saved Materialite section and compares
  exact and smoothed native EBSD meshes, without running hydrogen transport.
- `gb_homogenisation.py` identifies effective diffusivity, validates it against
  permeation and uptake, and generates the study figures in one run.
- `li2022_fig4.py` and `li2022_fig4_codim.py` reproduce and compare the two
  grain-boundary formulations used for Li et al. (2022), Fig. 4.

## Outputs

Each script writes generated files under `examples/results/<script_name>/`,
independent of the directory from which it is run. For example, the 2D Voronoi
exports are in `examples/results/voronoi_polycrystal_2d/`. Meshes, diagnostics,
figures, and JSON files use the same per-script layout. Rerunning a script can
replace its own outputs; different scripts use separate folders.

The EBSD example runs the complete CTF → UPXO/DefDAP → UPXO mesh → FESTIM
chain. It writes `results/ebsd_ctf_to_mesh/poly.msh4` in metres, with native
`poly-ebsd.npz` grain/pixel data, SI extent metadata, source provenance, and
validation reports. It also generates quality, orientation, mesh, area and
FESTIM network images. `ebsd_gb_diffusion.py` reads these prepared files; its
simulation exports go to `results/ebsd_gb_diffusion/`. See the
[EBSD workflow guide](../docs/ebsd_workflow.md) for import settings and files.
Activate the environment in `environment.yml`.
The original CTF stays in `examples/data/`.

### Materialite section to mesh

Use the saved twenty-layer `slice_xz.npz` from the Materialite feasibility run:

```bash
python examples/materialite_slice_to_mesh.py --source-slice \
  ../materialite/TESTING/results/am_multilayer/layers_20/slice_xz.npz
```

This example needs the FESTIM/UPXO environment, not a Materialite installation
or a GPU. It selects two fully processed rectangles: a 195 x 140 µm balanced
region and a 65 x 510 µm strip along the build direction. Each crop is relabelled
into four-connected pieces while retaining its parent grain and spin mapping,
pixel orientations, original coordinate offset, and the existing `(x,z,-y)`
frame. Native map coordinates start at zero; `vox=(5,5)` is in µm and the mesher
uses `unit=1e-6` to export metres. No pixels are filled or grains removed.

The two exact meshes run first, followed by default five-iteration Taubin
smoothing. A rejected smoothed case gets one gentler attempt: one iteration
with 1.5-pixel thin-grain protection. Both use the existing geometry-preservation
checks; rejection is a reported outcome, never a reason to relax those limits.
The shared budget is 15 minutes by default (`--budget-seconds`), checked between
attempts and used to limit worker timeouts. Diagnostics may add a little time.

Results are in `results/materialite_slice_to_mesh/`, or `--output`:

- `crop_selection.png` shows both source windows.
- Each crop has `crop.npz`, coordinate/identity provenance, and `comparison.png`.
- Each accepted attempt has `poly.msh4`, copied native grain data, the existing
  mesh/area/network checks, and `mesh.png` showing all triangle edges and a
  central 50 x 50 µm zoom.
- `summary.json` and `summary.csv` report accepted/rejected attempts, settings,
  element counts, mesh quality, timing, and DOLFINx/orientation checks.

The coloured mesh and raster use matching grain colours. Their vertical local
coordinate represents original build direction z, even though the unchanged
2D package diagnostics label that coordinate y. No further orientation rotation
is applied. No transport problem or species is created.

The first measured run completed all comparisons in 53.8 seconds:

| Crop | Grain pieces | Exact triangles | Smoothed triangles | Max smoothed grain-area change |
| --- | ---: | ---: | ---: | ---: |
| Balanced, 195 x 140 µm | 65 | 6,432 | 6,343 | 0.85% |
| Tall, 65 x 510 µm | 140 | 10,080 | 9,948 | 1.10% |

Both default five-iteration smoothed geometries were rejected by the existing
5% per-grain area limit (maximum changes 40.9% and 30.3%). Both one-iteration,
thin-grain-protected retries passed. All four accepted meshes retain the grain
and boundary tags, load into DOLFINx, and preserve the orientation table exactly.

View the [crop selection](results/materialite_slice_to_mesh/crop_selection.png),
[balanced comparison](results/materialite_slice_to_mesh/balanced/comparison.png),
and [tall comparison](results/materialite_slice_to_mesh/tall/comparison.png).
The selected smoothed settings are identified in each comparison. These are
grain-ID colours, not IPF colours.

This check also exposed a precision issue in the adapter: UPXO's default
coordinate snapping scales with the raster span, introducing relative area
errors above `1e-7` in one-pixel grains on long maps. The adapter now passes
`snap_tol=1e-9` in its native pixel-coordinate units. Geometry-preservation and
mesh acceptance limits are unchanged. A long-raster regression reproduces the
original failure and verifies the fix; 58 crop/mesh-only tests pass.

The consolidated study writes `identified.json`, `validation.json`, and
`fig_*.png` under `results/gb_homogenisation/`. Validation and the field map
reuse the first size/seed's identification and transport parameters. Use
`--skip-validation`, `--skip-transient`, `--skip-figures`, or `--skip-fields`
to omit stages. `--plot-only` regenerates summary figures from the JSON without
solving; the field map is generated during a simulation run. `--out` and
`--validation-out` select JSON filenames within the study's results directory.
The exchange-rate plot requires `--k-sweep`; the RVE plot requires multiple
identifications. Field exports include cell size and seed in their names.

`plot_li2022_fig4.py` recreates the two Li Fig. 4 comparison PNGs from their
CSV outputs, requiring only NumPy and Matplotlib:

```bash
python examples/plot_li2022_fig4.py
```

Running the file reads `results/li2022_fig4/li2022-fig4.csv` and
`results/li2022_fig4_codim/li2022-fig4-codim-thin.csv`, relative to the script,
and saves each PNG beside its CSV. No command-line options are needed.
The analytical reference curves are recomputed; all simulation markers are
read from the CSV files.
