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
- `gb_homogenisation.py` identifies effective diffusivity, validates it against
  permeation and uptake, and generates the study figures in one run.
- `li2022_fig4.py` and `li2022_fig4_codim.py` reproduce and compare the two
  grain-boundary formulations used for Li et al. (2022), Fig. 4.
- `delic2026_crosssection.py` models a W foil cross-section with columnar
  boundaries under plasma loading and reads the back surface through synthetic
  indicator dots, for comparison with the patterned hydrogenography of
  Delic et al. (2026). Its two studies show the inverse permeation halo
  (closed boundary mouths, strongly coupled network) and the halo (over-fed
  mouths, weakly coupled network). Parameters are plausible, not fitted.

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
