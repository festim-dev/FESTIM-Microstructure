# EBSD workflow

`fm.run_ebsd_pipeline` imports a CTF map through UPXO/DefDAP, meshes the
grains with UPXO, and returns the `EbsdMicrostructure` after checking its mesh
and tags in DOLFINx. The path stem of the saved files is `micro.base`. Start with
the [EBSD quick start](../README.md#measured-ebsd-meshes). The
[UPXO evaluation](upxo_evaluation.md) describes the backend and validation
results.

## Import settings

Set the CTF path in `fm.ebsd.Settings` and pass those settings to
`fm.EbsdOptions(import_settings=...)`. Supply the crop as a tuple or list
`(xmin, xmax, ymin, ymax)` in the source coordinate units.
The pipeline, import, and mesh options are immutable. Use `dataclasses.replace()`
to vary a setting across runs.

Quality and phase filters and the crop apply before UPXO's `EBSDReader` detects
grains. DefDAP supplies grain labels and cubic symmetry equivalents. The importer
aligns quaternion signs before averaging to avoid DefDAP 0.93.6 cancellation; the
package translates its quaternion convention and applies explicit sample-frame
corrections. The current boundary-angle diagnostics support cubic m-3m (Laue 11).

`Settings(fill=True)` assigns unindexed/pruned pixels to the nearest surviving
grain for meshing. The archive retains their original membership and quality
mask, and those pixels never contribute to grain means or indexed orientation
error. Use `fill=False` to retain unmeshed regions. `min_pixels` and `threshold`
control DefDAP grain detection. `flip_y` mirrors rows and rotates both pixel and
grain orientations by 180° about sample x. `euler_correction` sets the
Euler-to-map frame correction.

## Meshing

Configure smoothing and mesh sizes with `UpxoMeshOptions`. Defaults use five
Taubin passes (λ=0.25, μ=−0.265), explicit pixel seeds, and no grain merging or
diagonal repairs. `smoothing="none"` preserves raster boundaries. Sizes are in
pixel coordinates; both pixel spacings are applied when exporting to metres.
The pipeline rejects changes to grain IDs, components, holes, neighbor pairs,
coverage, mesh conformity or grain areas. Default smoothing limits are 5%
per-grain area change and one pixel of boundary displacement. Use
`smoothing="none"` with `fill=False` to preserve void boundaries and pass the
coverage check.

`thin_grain_px=1.5` enables optional protection for thin grains. The default
is `0` (disabled). The adapter sets `close_staircase=False` to preserve grain
labels and uses the same preservation checks with thin-grain protection.
Before accepting a mesh, the adapter reads back the exported file and checks
each triangle's physical grain ID and each boundary segment's physical ID against
the UPXO mesh, allowing node and element reordering.

## Files and diagnostics

Outputs include `<stem>.msh4`, `<stem>-ebsd.npz` (grain/pixel orientations,
labels, masks and spacings), `<stem>-metadata.json` (SI extent), import provenance,
`<stem>-validation.json`, and `<stem>-festim.json`. Quantitative diagnostics cover
source orientation transcription, indexed/filled error populations, grain area
and equivalent-diameter changes, and a centroid-sampled displaced-area estimate.
Physical grain IDs pair the areas. Raster overlap measures geometric displacement.
The validation report records geometry, mesh, and export checks separately.
Mesh diagnostic images are `<stem>-check-area.png` (grain-area changes),
`<stem>-check-mesh.png` (mesh over source pixels), and
`<stem>-check-network.png` (FESTIM boundary selection and disorientation).
Optional images show quality rejection, grains and cubic IPF-Z, orientation
error, mesh overlays, and the actual FESTIM boundary network.

`EbsdMicrostructure.grain_quats` retains the archive's unit quaternions in
Bunge convention. The default lattice diffusivity is isotropic. The planar
anisotropy model requires planar angles. Inspect boundary arrays on
`micro.edges`, which exposes `grain_pairs` (zero marks the exterior),
`interior_mask`, `theta`, `length`, `ymin`, and `ymax`. Rows follow boundary
tags starting at 1, and lengths and extents are in metres.

## Reuse and provenance

Import provenance uses format version 2 and stores crop bounds as a JSON array.
With `force=False`, unsupported provenance versions trigger a fresh import.

For separate import/inspection, use
`fm.ebsd.convert(ctf, "grains.npz", ...)` and
`fm.ebsd.measure_against_ctf(ctf, "grains.npz")`.
`fm.ebsd.mesh_grains("grains.npz", ...)` meshes an existing native archive.
`force=False` checks source/settings/code and output hashes before cache reuse.
For meshing, the code digest covers the executable worker and the cache key
also includes the serialized mesh options. An older digest triggers a rebuild.
A failed meshing worker retains the previous accepted mesh and writes a failure
validation report and log.

In the standard environment, the active interpreter runs both the mesher and
solver. `EbsdOptions(python=...)` selects one interpreter for both worker stages.
Otherwise the pipeline uses `FM_EBSD_PYTHON` or automatic discovery. Set
`timeout` to adjust the per-invocation limit of 600 seconds. Standalone
`convert` and `mesh_grains` accept `python` and `timeout` keyword arguments.

`EbsdOptions(check_images=False)` skips import and mesh plots while producing
the numeric reports. Standalone `convert` generates plots with
`check_images=True`.
`environment.yml` pins UPXO's evaluated commit and its DefDAP extra.
UPXO remains an external dependency with its upstream GPL-3.0 license.
