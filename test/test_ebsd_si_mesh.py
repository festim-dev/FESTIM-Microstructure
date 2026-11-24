"""Exercise the real UPXO worker and its tagged DOLFINx reconstruction."""

import json
from pathlib import Path

import numpy as np
import pytest

_ISOLATED_WORKER = """
import importlib.abc
import runpy
import sys

class BlockSolverImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'festim_microstructure', 'festim', 'dolfinx', 'mpi4py', 'petsc4py',
            'ufl', 'basix',
        }:
            raise AssertionError(f'unexpected worker import: {fullname}')

sys.meta_path.insert(0, BlockSolverImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


@pytest.mark.fenics
def test_pipeline_exports_si_mesh_and_preserves_enclosed_grain(
    tmp_path, monkeypatch, ctf_writer
):
    import festim as F

    import festim_microstructure as fm
    from festim_microstructure import ebsd
    from festim_microstructure.ebsd import upxo
    from festim_microstructure.ebsd.archive import read_ebsd, read_extent
    from festim_microstructure.ebsd.options import Settings, settings_from_provenance
    from festim_microstructure.ebsd.orientation import euler_bunge_to_quat
    from festim_microstructure.msh4 import read_mesh

    executable = upxo.resolve_python(required=False)
    if executable is None:
        pytest.skip("configure FM_UPXO_PYTHON for the UPXO integration test")
    monkeypatch.setenv("FM_UPXO_PYTHON", "/missing-upxo-python")
    worker_path = str(Path(upxo.__file__).resolve())
    worker_calls = []
    subprocess_run = upxo.subprocess.run

    def isolated_run(command, **kwargs):
        if command[1:3] == ["-I", worker_path]:
            assert command[0] == executable
            worker_calls.append((command[3], kwargs["timeout"]))
            command = [executable, "-I", "-B", "-c", _ISOLATED_WORKER, *command[2:]]
        return subprocess_run(command, **kwargs)

    monkeypatch.setattr(upxo.subprocess, "run", isolated_run)
    assert upxo.probe_worker(executable, health=True)["health"] == "ok"
    labels = np.ones((12, 16), dtype=np.int32)
    labels[4:8, 6:10] = 2
    orientations = np.zeros((*labels.shape, 3))
    orientations[labels == 1] = [37, 52, 131]
    orientations[labels == 2] = [21, 43, 73]
    ctf = ctf_writer(tmp_path / "map.ctf", orientations, spacing=(0.5, 0.75))
    options = ebsd.EbsdOptions(
        unit=1e-6,
        check_images=False,
        python=executable,
        timeout=120.0,
        import_settings=Settings(ctf=str(ctf), min_pixels=1, crop=(0, 8, 0, 9)),
        mesh=ebsd.UpxoMeshOptions(smoothing="none"),
    )
    base = fm.run_ebsd_pipeline(options, workdir=tmp_path, force=False)
    assert {mode for mode, _ in worker_calls} == {
        "--health",
        "--environment",
        "--import",
        "--worker",
    }
    assert all(timeout == 120.0 for mode, timeout in worker_calls if mode != "--health")
    assert not list(tmp_path.glob("*.png"))
    assert Path(f"{base}-areachange.csv").is_file()
    assert Path(f"{base}-festim.json").is_file()
    provenance = base.with_name(f"{base.name}-import-provenance.json")
    assert settings_from_provenance(provenance).crop == (0, 8, 0, 9)
    mesh, cells, facets = read_mesh(base, gdim=2)
    np.testing.assert_allclose(mesh.geometry.x.max(axis=0), [8e-6, 9e-6, 0])
    np.testing.assert_array_equal(np.unique(cells.values), [1, 2])
    np.testing.assert_array_equal(np.unique(facets.values), [1, 2])
    np.testing.assert_allclose(read_extent(base), [8e-6, 9e-6])
    np.testing.assert_allclose(
        np.abs(read_ebsd(f"{base}-ebsd.npz").grain_quats),
        np.abs(
            euler_bunge_to_quat(
                np.array([37, 21]), np.array([52, 43]), np.array([131, 73])
            )
        ),
        atol=1e-6,
    )
    micro = ebsd.EbsdMicrostructure.from_mesh(
        base, mesh, cells, facets, read_extent(base)
    )
    assert micro.n_grains == 2
    assert micro.triple_junctions == 0
    assert micro.surface_edges == 1
    assert micro.interior_mask.sum() == 1
    np.testing.assert_allclose(sorted(micro.edges.length), [10e-6, 34e-6])
    np.testing.assert_array_equal(
        micro.grain_quats, read_ebsd(f"{base}-ebsd.npz").grain_quats
    )
    assert micro.edges.grain_pairs.dtype.kind in "iu"
    assert micro.edges.interior_mask.dtype.kind == "b"
    np.testing.assert_array_equal(
        micro.edges.grain_pairs[micro.interior_mask], [[1, 2]]
    )
    np.testing.assert_array_equal(
        micro.edges.grain_pairs[~micro.interior_mask], [[1, 0]]
    )
    micro.check_orientations()
    loaded = fm.EbsdMicrostructure.load(base, theta_min=0.0, comm=mesh.comm)
    assert isinstance(loaded, fm.MeshedMicrostructure)
    assert isinstance(loaded, fm.BoundaryTopology)
    assert (
        micro.mesh is mesh and micro.cell_tags is cells and micro.facet_tags is facets
    )
    np.testing.assert_array_equal(loaded.grain_ids, [1, 2])
    np.testing.assert_array_equal(loaded.cell_tags.values, cells.values)
    np.testing.assert_array_equal(loaded.facet_tags.values, facets.values)
    np.testing.assert_array_equal(loaded.theta, micro.theta)
    np.testing.assert_array_equal(loaded.gb_tag, micro.network_ids)
    np.testing.assert_array_equal(loaded.grain_quats, micro.grain_quats)
    assert loaded.extent == micro.extent
    assert loaded.theta_min == 0.0
    assert not hasattr(loaded, "orientations")
    assert not hasattr(loaded, "locator")

    cartesian = fm.TaggedPolycrystal(loaded.mesh, loaded.cell_tags)
    isotropic, tensors = fm.crystal_diffusivity_field(loaded, 1e-12)
    field, _ = fm.crystal_diffusivity_field(cartesian, 1e-12)
    np.testing.assert_array_equal(isotropic.x.array, field.x.array)
    for tensor in tensors.values():
        np.testing.assert_array_equal(tensor, np.eye(2) * 1e-12)
    with pytest.raises(ValueError, match="requires planar orientations"):
        fm.crystal_diffusivity_field(loaded, 1e-12, crystal_anisotropy=2.0)

    # The interior grain is enclosed, so the top edge has only a grain patch.
    system = fm.GrainBoundarySystem(
        loaded,
        F.Material(D=isotropic),
        width=1e-7,
        diffusivity_by_entity=np.full_like(loaded.theta, 1e-10),
    )
    surfaces, fixed = system.fixed_concentration(
        lambda x: np.isclose(x[1], loaded.extent[1]),
        lambda x: 1.0 + x[0],
        include_mouths=False,
    )
    assert len(surfaces) == len(fixed) == 1
    assert surfaces[0].grain_id == 1
    sources, flux = system.exchange(1e-7)
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(loaded.mesh),
        subdomains=[*system.subdomains, *surfaces],
        species=system.species,
        sources=sources,
        boundary_conditions=[*flux, *fixed],
        temperature=500.0,
        settings=F.Settings(atol=1e-25, rtol=1e-10, transient=False),
    )
    model.initialise()

    assert json.loads(Path(f"{base}-validation.json").read_text())["accepted"]
    original = base.with_suffix(".msh4").read_bytes()

    run = upxo.subprocess.run
    with monkeypatch.context() as context:

        def unexpected_worker(command, **kwargs):
            if "--worker" in command or "--import" in command:
                raise AssertionError("the matching cache must not rerun UPXO")
            return run(command, **kwargs)

        context.setattr(upxo.subprocess, "run", unexpected_worker)
        options.check_images = True
        options.timeout = 150.0
        assert fm.run_ebsd_pipeline(options, workdir=tmp_path, force=False) == base
    assert base.with_suffix(".msh4").read_bytes() == original
    assert {path.name for path in tmp_path.glob("*.png")} == {
        f"{base.name}-{suffix}.png"
        for suffix in (
            "import-segerr",
            "import-quality",
            "import-ipfz",
            "check-mesh",
            "check-area",
            "check-network",
        )
    }

    options.check_images = False
    options.unit = 1e-3
    ebsd.run_ebsd_pipeline(options, workdir=tmp_path, force=False)
    scaled, _, _ = read_mesh(base, gdim=2)
    np.testing.assert_allclose(scaled.geometry.x.max(axis=0), [8e-3, 9e-3, 0])
    np.testing.assert_allclose(read_extent(base), [8e-3, 9e-3])
    assert not list(tmp_path.glob("*.tesr"))
    assert not list(tmp_path.glob("*.sttesr"))
    assert not list(tmp_path.glob("poly-unscaled*"))
    assert base.with_suffix(".msh4").is_file()
    assert Path(f"{base}-validation.json").is_file()


@pytest.mark.fenics
@pytest.mark.parametrize("smoothing", ["none", "taubin"])
def test_ctf_transport_conserves_inventory_and_exchanges_with_boundary(
    tmp_path, ctf_writer, smoothing
):
    """A closed two-grain system transfers particles to its initially empty GB."""
    import dolfinx
    import festim as F

    import festim_microstructure as fm
    from festim_microstructure import ebsd
    from festim_microstructure.ebsd import upxo
    from festim_microstructure.ebsd.options import Settings
    from festim_microstructure.exports.averages import inventory, solutions

    if upxo.resolve_python(required=False) is None:
        pytest.skip("UPXO interpreter unavailable")
    labels = np.ones((12, 16), dtype=np.int32)
    labels[4:8, 6:10] = 2
    euler = np.zeros((*labels.shape, 3))
    euler[labels == 2] = [0, 0, 20]
    ctf = ctf_writer(tmp_path / "transport.ctf", euler, spacing=(0.5, 0.75))
    options = ebsd.EbsdOptions(
        unit=1e-6,
        check_images=False,
        import_settings=Settings(ctf=str(ctf), min_pixels=1),
        mesh=ebsd.UpxoMeshOptions(smoothing=smoothing),
    )
    base = fm.run_ebsd_pipeline(options, workdir=tmp_path)
    micro = fm.EbsdMicrostructure.load(base)
    delta = 1e-7
    k = dolfinx.fem.Constant(micro.mesh, dolfinx.default_scalar_type(0.0))
    width = dolfinx.fem.Constant(micro.mesh, dolfinx.default_scalar_type(2 * delta))
    lattice, _ = fm.crystal_diffusivity_field(micro, 1e-12)
    system = fm.GrainBoundarySystem(
        micro, F.Material(D=lattice), F.Material(D_0=1e-10, E_D=0.0), width=width
    )
    assert system.width is width
    sources, flux = system.exchange(k)
    settings = F.Settings(
        atol=1e-25, rtol=1e-10, transient=True, final_time=0.2, stepsize=0.02
    )
    model = F.HydrogenTransportProblemDiscontinuous(
        mesh=F.Mesh(micro.mesh),
        subdomains=system.subdomains,
        species=system.species,
        sources=sources,
        boundary_conditions=flux,
        temperature=500.0,
        settings=settings,
    )
    model.show_progress_bar = False
    grains, species = system.grains, system.grain_species
    network, c_gb = system.network, system.c_gb
    model.initial_conditions = [
        F.InitialConcentration(value=1.0, species=c, volume=g)
        for c, g in zip(species, grains, strict=True)
    ]
    model.initialise()
    fm.fem.tune_direct_solver(model)
    assert model.manifold_is_interior(network)
    # A sweep updates the constants after FESTIM has compiled the forms.
    k.value = 1e-7
    width.value = delta
    # FESTIM initializes the previous-step fields; expose them for the t=0 check.
    for subdomain in model.volume_subdomains:
        subdomain.u.x.array[:] = subdomain.u_n.x.array
    model.post_processing()
    initial = system.inventory()
    np.testing.assert_allclose(initial, 8e-6 * 9e-6, rtol=1e-10, atol=0)
    model.run()
    final = system.inventory()
    np.testing.assert_allclose(final, initial, rtol=1e-8, atol=0)
    gb_solution = c_gb.subdomain_to_post_processing_solution[network]
    assert gb_solution.x.array.mean() > 0.01
    assert all(c.x.array.mean() < 1.0 for c in solutions(species, grains))
    width.value = 2 * delta
    assert system.inventory() == pytest.approx(
        inventory(grains, species, network, c_gb, 2 * delta), rel=1e-10, abs=0
    )
