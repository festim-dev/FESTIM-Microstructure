"""Network location, grain patches, and FESTIM surface declarations."""

import numpy as np
import pytest


@pytest.fixture
def network():
    """Return the grain-boundary network class for constructor checks."""
    from festim_microstructure.fem.subdomains import GrainBoundaryNetwork

    return GrainBoundaryNetwork


def tags(values):
    """A stand-in for dolfinx MeshTags: values, and the facets carrying them."""
    values = np.asarray(values, dtype=np.int32)
    indices = np.arange(values.size, dtype=np.int32)
    return type("Tags", (), {"values": values, "indices": indices})()


def test_a_locator_and_facet_tags_are_alternatives_not_a_pair(network):
    """Verify a locator and facet tags are alternatives not a pair."""
    with pytest.raises(TypeError, match="both"):
        network(
            1,
            object(),
            1,
            facet_tags=tags([1]),
            entity_ids=[1],
            locator=lambda x: np.ones(x.shape[1], dtype=bool),
        )


def test_a_network_has_to_be_located_somehow(network):
    """Verify a network has to be located somehow."""
    with pytest.raises(TypeError, match="neither"):
        network(1, object(), 1)


def test_facet_tags_without_ids_name_no_boundaries(network):
    """Verify facet tags without ids name no boundaries."""
    with pytest.raises(TypeError, match="entity_ids"):
        network(1, object(), 1, facet_tags=tags([1, 2]))


def test_one_tag_and_a_sequence_of_them_are_both_accepted(network):
    """EBSD tags every edge separately; a Voronoi mesh uses one marker."""
    assert network(1, object(), 2, facet_tags=tags([7]), entity_ids=7).entity_ids == [7]
    many = network(1, object(), 2, facet_tags=tags([7]), entity_ids=[3, 7])
    assert list(many.entity_ids) == [3, 7]


def test_the_exterior_filter_follows_the_route_unless_it_is_asked_not_to(network):
    """On for a predicate, which cannot tell the network from its trace on the
    wall of the box; off for tags, which say what a facet is."""
    located = network(1, object(), 1, locator=lambda x: x)
    tagged = network(1, object(), 1, facet_tags=tags([1]), entity_ids=[1])
    assert located.drop_exterior
    assert not tagged.drop_exterior
    loose = network(1, object(), 1, locator=lambda x: x, drop_exterior=False)
    assert not loose.drop_exterior
    assert network(
        1, object(), 1, facet_tags=tags([1]), entity_ids=[1], drop_exterior=True
    ).drop_exterior


def test_a_tagged_network_starts_with_nothing_mapped_back(network):
    """``entity_ids_of_facets`` is filled by the lookup, not by the constructor."""
    net = network(1, object(), 1, facet_tags=tags([1]), entity_ids=[1])
    assert net.entity_ids_of_facets is None
    assert net.n_dropped == 0


def _layered_polycrystal(dim):
    from mpi4py import MPI

    import dolfinx

    import festim_microstructure as fm

    if dim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 3, 3, 3)
    cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
    heights = dolfinx.mesh.compute_midpoints(mesh, dim, cells)[:, dim - 1]
    grain_ids = np.array([2, 7, 11], dtype=np.int32)
    values = grain_ids[np.searchsorted([1 / 3, 2 / 3], heights)]
    tags = dolfinx.mesh.meshtags(mesh, dim, cells, values)
    return fm.TaggedPolycrystal(
        mesh,
        tags,
        network_locator=lambda x: (
            np.isclose(x[dim - 1], 1 / 3) | np.isclose(x[dim - 1], 2 / 3)
        ),
    )


@pytest.mark.fenics
@pytest.mark.parametrize("dim", [2, 3])
def test_grain_surfaces_assign_facets_in_one_lookup(dim):
    import dolfinx
    import festim as F

    import festim_microstructure as fm

    micro = _layered_polycrystal(dim)
    grains = fm.grain_subdomains(micro, F.Material(D_0=1.0, E_D=0.0))
    calls = []

    def exterior_ends(x):
        calls.append(x.shape[1])
        return np.isclose(x[dim - 1], 0) | np.isclose(x[dim - 1], 1)

    patches, mouths = fm.grain_surfaces(micro.mesh, grains, exterior_ends, 100)
    assert [p.grain_id for p in patches] == [2, 11]
    assert [p.id for p in patches] == [100, 102]
    assert mouths.id == 103 and mouths.dim == dim - 2
    located = [p.locate_boundary_facet_indices(micro.mesh) for p in patches]
    assert len(calls) == 1

    facets = dolfinx.mesh.exterior_facet_indices(micro.mesh.topology)
    heights = dolfinx.mesh.compute_midpoints(micro.mesh, dim - 1, facets)[:, dim - 1]
    for height, actual in zip((0, 1), located, strict=True):
        expected = facets[np.isclose(heights, height)]
        np.testing.assert_array_equal(np.sort(actual), np.sort(expected))


@pytest.mark.fenics
def test_grain_subdomains_take_one_material_or_one_per_grain():
    import festim as F

    import festim_microstructure as fm

    micro = _layered_polycrystal(2)
    shared = F.Material(D_0=1.0, E_D=0.0)
    assert all(g.material is shared for g in fm.grain_subdomains(micro, shared))
    by_grain = {g: F.Material(D_0=float(g), E_D=0.0) for g in (2, 7, 11)}
    grains = fm.grain_subdomains(micro, by_grain.__getitem__)
    assert [g.material for g in grains] == [by_grain[g.id] for g in grains]


@pytest.mark.fenics
def test_exchange_pairs_every_grain_species_with_the_network():
    import festim as F

    import festim_microstructure as fm

    micro = _layered_polycrystal(2)
    material = F.Material(D_0=1.0, E_D=0.0)
    grains = fm.grain_subdomains(micro, material)
    network = fm.grain_boundary_network(20, micro, material)
    species = [F.Species(f"c_{g.id}", subdomains=[g]) for g in grains]
    c_gb = F.Species("c_gb", subdomains=[network])
    sources, flux_bcs = fm.exchange(species, c_gb, network, width=0.5, k=2.0)
    assert [s.species for s in sources] == [c_gb] * len(grains)
    assert [bc.species for bc in flux_bcs] == species
    assert all(bc.subdomain is network for bc in flux_bcs)
    with pytest.raises(TypeError, match="exactly one"):
        fm.exchange(species, c_gb, network, width=0.5)
    with pytest.raises(TypeError, match="exactly one"):
        fm.exchange(species, c_gb, network, width=0.5, k=2.0, flux=lambda a, b: a)
