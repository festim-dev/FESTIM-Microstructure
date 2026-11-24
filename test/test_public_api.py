"""The advertised public exports and package boundaries."""

import ast
import importlib
import pathlib

import pytest

import festim_microstructure

PACKAGES = [
    "festim_microstructure",
    "festim_microstructure.fem",
    "festim_microstructure.ebsd",
    "festim_microstructure.postprocessing",
    "festim_microstructure.voronoi",
]

SRC = pathlib.Path(festim_microstructure.__file__).parent


@pytest.mark.parametrize("package", PACKAGES)
def test_every_advertised_name_is_imported(package):
    """Public exports are available immediately in each package namespace."""
    module = importlib.import_module(package)
    assert set(module.__all__) <= vars(module).keys()


@pytest.mark.parametrize("package", PACKAGES)
def test_unknown_attributes_raise_attribute_error(package):
    """Verify unknown attributes raise attribute error."""
    module = importlib.import_module(package)
    with pytest.raises(AttributeError, match="no attribute"):
        module.definitely_not_a_real_name


def _module_level_imports(path):
    """Root package names imported at module scope.

    Only ``tree.body`` is walked; function-local imports are excluded.
    """
    roots = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "path",
    sorted(
        p for p in SRC.glob("*.py") if p.name not in {"_version.py", "materials.py"}
    ),
    ids=lambda p: p.name,
)
def test_top_level_solver_imports_are_confined_to_materials(path):
    """Direct solver imports at the package root stay in materials.py."""
    forbidden = {"dolfinx", "festim", "ufl", "basix", "petsc4py", "mpi4py"}
    assert _module_level_imports(path) & forbidden == set()


def test_fem_is_where_the_fenics_glue_lives():
    """Verify fem is where the fenics glue lives."""
    fem = SRC / "fem"
    assert fem.is_dir()
    heavy = {
        p.name
        for p in fem.glob("*.py")
        if _module_level_imports(p) & {"dolfinx", "festim", "petsc4py"}
    }
    assert heavy == {"coupling.py", "solvers.py", "subdomains.py"}


def test_ebsd_solver_imports_are_confined_to_mesh_integration():
    """The worker and data preparation modules have no solver imports."""
    solver = {"dolfinx", "festim", "ufl", "basix", "petsc4py", "mpi4py"}
    allowed = {"topology", "pipeline", "mesh_checks"}
    for path in (SRC / "ebsd").glob("*.py"):
        if path.stem in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & solver
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in solver
