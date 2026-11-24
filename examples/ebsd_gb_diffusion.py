"""Short-circuit diffusion through the GB network of a polycrystal measured by
EBSD, in 2D.

The mesh is UPXO's validated meshing of the raster, so the grain boundaries are
the measured ones and each boundary's disorientation comes from the two grains'
measured orientations. See :mod:`festim_microstructure.ebsd.topology`.

Every measured grain carries a lattice species of its own and exchanges with the
one network species at the rate ``k``; the network is the list of edge ids above
the disorientation threshold.

Prerequisite: the SI ``poly.msh4`` and accompanying extent/orientation files
written by ``examples/ebsd_ctf_to_mesh.py``.
"""

from pathlib import Path

import festim as F
import numpy as np

import festim_microstructure as fm

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HERE = Path(__file__).resolve().parent
NETWORK_ID = 1_000_000  # above every grain id
SURFACE_ID_0 = 2_000_000  # the per-grain surface patches are numbered from here

# Simulation parameters
base = HERE / "results" / "ebsd_ctf_to_mesh" / "poly"
theta_min = 10.0

D_B = 1e-14  # lattice diffusivity     [m^2/s]
D_GB = 1e-8  # GB diffusivity          [m^2/s]
delta = 5e-9  # GB width               [m]
k_exchange = 1e-4  # bulk <-> GB exchange [m/s]
c0 = 1.0
t_end = 36000.0
dt = 600.0
theta_dependent_D = False  # D_GB above theta_c only, D_B below; CHECK before use
theta_c = 15.0  # degrees

# Build the microstructure and simulation
micro = fm.EbsdMicrostructure.load(base, theta_min=theta_min)
mesh = micro.mesh
LX, LY = micro.extent
T = 500.0  # with E_D = 0 on both phases, nothing depends on it

D_lattice, _ = fm.crystal_diffusivity_field(micro, D_B)
grains = fm.grain_subdomains(micro, F.Material(D=D_lattice))
# One material for every boundary, or a diffusivity per measured boundary from
# its disorientation; the network builds that field on its submesh.
if theta_dependent_D:
    network = fm.grain_boundary_network(
        NETWORK_ID,
        micro,
        None,
        diffusivity_by_entity=np.where(micro.theta >= theta_c, D_GB, D_B),
    )
else:
    network = fm.grain_boundary_network(
        NETWORK_ID, micro, F.Material(D_0=D_GB, E_D=0.0)
    )
grain_species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
c_gb = F.Species("c_gb", subdomains=[network])
species_of = dict(zip((g.id for g in grains), grain_species, strict=True))
sources, flux_bcs = fm.exchange(grain_species, c_gb, network, width=delta, k=k_exchange)
# The charged surface is the top edge of the map.
patches, mouths = fm.grain_surfaces(
    mesh, grains, lambda x: np.isclose(x[1], LY), SURFACE_ID_0
)
surface_bcs = [
    F.FixedConcentrationBC(subdomain=p, value=c0, species=species_of[p.grain_id])
    for p in patches
]
surface_bcs.append(F.FixedConcentrationBC(subdomain=mouths, value=c0, species=c_gb))

model = F.HydrogenTransportProblemDiscontinuous(
    mesh=F.Mesh(mesh),
    subdomains=[*grains, network, *patches, mouths],
    species=[*grain_species, c_gb],
    sources=sources,
    boundary_conditions=[*flux_bcs, *surface_bcs],
    temperature=T,
    settings=F.Settings(
        atol=1e-14, rtol=1e-12, transient=True, final_time=t_end, stepsize=dt
    ),
)
model.initialise()
fm.fem.tune_direct_solver(model)
model.run()
fm.exports.averages.write_vtx(
    grains,
    grain_species,
    network,
    c_gb,
    OUTPUT_DIR / "ebsd",
    time=t_end,
)
cgb = c_gb.subdomain_to_post_processing_solution[network]

# Inspect the mesh and boundary network
print(micro.report())
len_mesh, len_tess = (
    fm.exports.measures.submesh_measure(network),
    micro.network_measure,
)
n_tri = mesh.topology.index_map(2).size_global
print(f"  mesh                            : {n_tri} triangles")
print(f"  grains (tagged pieces)          : {micro.n_grains}")
print(f"  network captured by the submesh : {100 * len_mesh / len_tess:.2f} %")
print(
    f"  connected components            : "
    f"{fm.exports.measures.component_count(network)}"
)
print(f"  interior facets                 : {model.manifold_is_interior(network)}")

# Analyse grain-boundary transport
depth = micro.junction_only_below(LY)
gb_y = cgb.function_space.tabulate_dof_coordinates()[:, 1]
deep = gb_y < depth
c_deep = cgb.x.array[deep].max() if deep.any() else 0.0
total = fm.exports.averages.inventory(grains, grain_species, network, c_gb, delta)
print(f"\n  inventory                      : {total:.4e}")
# Le Claire's type-B parameter.
beta = delta * (D_GB / D_B - 1) / (2 * np.sqrt(D_B * t_end))
print(f"  type-B parameter beta          : {beta:.0f}  (needs beta >> 1)")
grain_size = np.sqrt(LX * LY / micro.n_grains)
resistance = (2.0 / k_exchange) / (grain_size / D_B)
print(f"  interface / lattice resistance : {resistance:.1e}  (Fisher is the 0 limit)")

c_grain_deep = fm.exports.averages.lattice_mean_below(
    grains, grain_species, axis=1, depth=depth
)
print("\njunction transport: no boundary touching the charged edge reaches below")
print(f"y = {depth:.4g}, so everything the network holds there has")
print("crossed at least one triple junction.")
print(f"  max c on the network there     : {c_deep:.4e}")
print(f"  mean c in the grains there     : {c_grain_deep:.4e}")
# both are solver noise around zero whenever the boundary tail is shorter
# than the junction-only depth, and noise has a sign; the ratio is then 0/0
if c_grain_deep > 1e-12 * c0:
    print(f"  ratio                          : x {c_deep / c_grain_deep:.0f}")
else:
    print("  ratio                          : n/a - nothing has reached this depth")
