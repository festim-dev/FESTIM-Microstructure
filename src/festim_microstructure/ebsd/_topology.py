"""Reconstruct grain-boundary topology from the tagged EBSD mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

from festim_microstructure.microstructure import _global_tag_values
from festim_microstructure.msh4 import read_mesh

from ._archive import read_ebsd, read_extent
from ._orientation import cubic_disorientation_angle, qconj, qmul

__all__ = ["BoundaryEdges", "EbsdMicrostructure"]


def _ids(mask):
    """Boolean mask over entities in id order -> 1-based ids."""
    return np.flatnonzero(mask).astype(np.int32) + 1


def _allreduce(comm, arr, op):
    """Reduce an array over MPI ranks into a new array."""
    out = np.empty_like(arr)
    comm.Allreduce(np.ascontiguousarray(arr), out, op=op)
    return out


@dataclass
class BoundaryEdges:
    """Boundary arrays in tag order; row ``k`` belongs to boundary ``k + 1``.

    ``grain_pairs`` contains integer grain IDs, with zero as the second ID
    for a surface boundary. ``interior_mask`` explicitly selects boundaries
    shared by two grains. Lengths and y-extents are in metres.
    """

    grain_pairs: np.ndarray
    interior_mask: np.ndarray
    theta: np.ndarray
    length: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray

    def __len__(self):
        """Return the number of tagged boundaries."""
        return len(self.grain_pairs)


def _edge_grain_pairs(comm, facet_edge, cell_grain, f2c, n_edge):
    """Which grains sit either side of each tagged boundary.

    Gathered across ranks: an edge can be cut by the partition, so no single
    rank sees both of its grains. Returns integer grain pairs and a boolean
    interior mask in boundary-tag order.
    """
    local = {}
    for f in np.flatnonzero(facet_edge):
        local.setdefault(int(facet_edge[f]), set()).update(
            int(cell_grain[c]) for c in f2c.links(f)
        )
    grains = [set() for _ in range(n_edge + 1)]
    for part in comm.allgather({k: sorted(v) for k, v in local.items()}):
        for k, v in part.items():
            grains[k].update(v)
    sides = np.array([len(g) for g in grains[1:]])
    if sides.min() < 1 or sides.max() > 2 or 0 in set().union(*grains[1:]):
        bad = _ids((sides < 1) | (sides > 2))
        raise RuntimeError(
            f"edges {bad[:10]} touch {sides[bad[:10] - 1]} grains; every "
            "boundary tag must lie between two grains or between one "
            "grain and the domain. Check the grain and boundary tags "
            "in the mesh."
        )
    pair = np.array(
        [sorted(g) + [0] * (2 - len(g)) for g in grains[1:]], dtype=np.int32
    )
    return pair, sides == 2


def _edge_extents(comm, mesh, facet_edge, fdim, fmap, n_edge):
    """Length and y-range of every edge set, summed/reduced over owned facets."""
    owned = np.arange(fmap.size_local, dtype=np.int32)
    owned = owned[facet_edge[owned] > 0]
    nodes = dolfinx.mesh.entities_to_geometry(mesh, fdim, owned, False)
    x = mesh.geometry.x[nodes]  # (nf, 2, 3)
    e = facet_edge[owned]
    length = np.bincount(
        e, weights=np.linalg.norm(x[:, 1] - x[:, 0], axis=1), minlength=n_edge + 1
    )
    ymin = np.full(n_edge + 1, np.inf)
    ymax = np.full(n_edge + 1, -np.inf)
    np.minimum.at(ymin, e, x[:, :, 1].min(axis=1))
    np.maximum.at(ymax, e, x[:, :, 1].max(axis=1))
    return (
        _allreduce(comm, length[1:], MPI.SUM),
        _allreduce(comm, ymin[1:], MPI.MIN),
        _allreduce(comm, ymax[1:], MPI.MAX),
    )


def _edge_orientations(base, max_grain_id, pair, interior_mask):
    """Native grain quaternions and boundary disorientations for this mesh."""
    data = read_ebsd(f"{base}-ebsd.npz")
    if data.ncells != max_grain_id:
        raise RuntimeError(
            f"native EBSD archive has {data.ncells} grains "
            f"but mesh grain tags reach {max_grain_id}"
        )
    q = data.grain_quats
    theta = np.zeros(len(pair))
    a, b = pair[interior_mask, 0] - 1, pair[interior_mask, 1] - 1
    theta[interior_mask] = cubic_disorientation_angle(qmul(qconj(q[a]), q[b]))
    return q, theta


def _count_triple_junctions(comm, mesh, facet_edge, v2f, vmap, extent):
    """Owned vertices where 3+ distinct edge ids meet, off the box boundary."""
    n_v = vmap.size_local
    verts = np.arange(n_v, dtype=np.int32)
    xv = dolfinx.mesh.compute_midpoints(mesh, 0, verts)
    lx, ly = extent
    tol = 1e-9 * max(lx, ly)
    on_box = (
        (np.abs(xv[:, 0]) < tol)
        | (np.abs(xv[:, 0] - lx) < tol)
        | (np.abs(xv[:, 1]) < tol)
        | (np.abs(xv[:, 1] - ly) < tol)
    )
    edgenb = np.fromiter(
        (len(set(facet_edge[v2f.links(v)].tolist()) - {0}) for v in verts),
        dtype=int,
        count=n_v,
    )
    return comm.allreduce(int(((edgenb >= 3) & ~on_box).sum()), op=MPI.SUM)


@dataclass
class EbsdMicrostructure:
    """Rebuild GB topology and disorientations from the tagged UPXO EBSD mesh.

    Facet tags identify boundaries and cell tags retain grain IDs. Grain
    orientations are the native archive's unit quaternions in Bunge convention.
    Junction counts are exact in serial and lower bounds in parallel.
    Implements :class:`~festim_microstructure.microstructure.BoundaryTopology`.
    Use :meth:`load` or :meth:`from_mesh` to reconstruct the boundary topology
    and retain the mesh and its tags for FESTIM.
    """

    base: Path
    extent: tuple
    mesh: dolfinx.mesh.Mesh
    cell_tags: dolfinx.mesh.MeshTags
    facet_tags: dolfinx.mesh.MeshTags
    grain_ids: np.ndarray
    grain_quats: np.ndarray  #: Unit quaternions indexed by grain ID - 1
    facet_edge: np.ndarray  #: edge id per local facet, 0 off the network
    edges: BoundaryEdges
    triple_junctions: int
    surface_edges: int
    theta_min: float = 10.0

    @classmethod
    def load(cls, base, *, theta_min=10.0, comm=None, rank=0):
        """Load the SI mesh, extent and orientation sidecars at the path stem.

        When using a communicator, call this on all of its ranks. ``rank`` is
        the rank that reads the Gmsh file, as in ``msh4.read_mesh``.
        """
        extent = read_extent(base)
        mesh, cells, facets = read_mesh(base, gdim=2, comm=comm, rank=rank)
        return cls.from_mesh(base, mesh, cells, facets, extent, theta_min=theta_min)

    @classmethod
    def from_mesh(
        cls, base, mesh, cell_tags, facet_tags, extent, theta_min=10.0, crysym="cubic"
    ):
        """Reconstruct topology from the pipeline's grain and boundary tags."""
        if crysym != "cubic":
            raise NotImplementedError(
                "theta is computed with the closed-form cubic disorientation "
                f"in EBSD; crysym = {crysym!r} needs a general "
                "symmetry-operator search"
            )
        comm = mesh.comm
        top = mesh.topology
        tdim = top.dim
        fdim = tdim - 1
        top.create_connectivity(fdim, tdim)
        top.create_connectivity(0, fdim)
        top.create_connectivity(0, tdim)
        f2c = top.connectivity(fdim, tdim)
        v2f = top.connectivity(0, fdim)
        fmap, cmap, vmap = top.index_map(fdim), top.index_map(tdim), top.index_map(0)

        facet_edge = np.zeros(fmap.size_local + fmap.num_ghosts, dtype=np.int32)
        facet_edge[facet_tags.indices] = facet_tags.values
        cell_grain = np.zeros(cmap.size_local + cmap.num_ghosts, dtype=np.int32)
        cell_grain[cell_tags.indices] = cell_tags.values
        n_edge = comm.allreduce(int(facet_edge.max(initial=0)), op=MPI.MAX)
        grain_ids = _global_tag_values(mesh, cell_tags)
        max_grain_id = int(grain_ids.max(initial=0))
        if n_edge == 0 or max_grain_id == 0:
            raise RuntimeError("the mesh carries no boundary or grain tags")

        pair, interior_mask = _edge_grain_pairs(
            comm, facet_edge, cell_grain, f2c, n_edge
        )
        length, ymin, ymax = _edge_extents(comm, mesh, facet_edge, fdim, fmap, n_edge)
        grain_quats, theta = _edge_orientations(base, max_grain_id, pair, interior_mask)

        return cls(
            base=Path(base),
            extent=extent,
            mesh=mesh,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            grain_ids=grain_ids,
            grain_quats=grain_quats,
            facet_edge=facet_edge,
            edges=BoundaryEdges(
                grain_pairs=pair,
                interior_mask=interior_mask,
                theta=theta,
                length=length,
                ymin=ymin,
                ymax=ymax,
            ),
            triple_junctions=_count_triple_junctions(
                comm, mesh, facet_edge, v2f, vmap, extent
            ),
            surface_edges=int((~interior_mask).sum()),
            theta_min=theta_min,
        )

    @property
    def n_grains(self):
        """Number of tagged grains in the mesh."""
        return len(self.grain_ids)

    @property
    def tolerance(self):
        """Distance tolerance for comparing coordinates in diagnostics."""
        return 1e-9 * max(self.extent)

    @property
    def interior_mask(self):
        """Mark edges shared by two grains rather than the exterior."""
        return self.edges.interior_mask

    @property
    def network_mask(self):
        """Mark interior edges above the disorientation threshold."""
        return self.interior_mask & (self.theta > self.theta_min)

    @property
    def network_ids(self):
        """1-based ids of the edges in the network."""
        return _ids(self.network_mask)

    @property
    def gb_tag(self):
        """Facet tag values for the selected grain-boundary network."""
        return self.network_ids

    @property
    def theta(self):
        """Disorientation of every edge, degrees, in id order."""
        return self.edges.theta

    @property
    def network_measure(self):
        """Total length of the boundaries in the network."""
        return float(self.edges.length[self.network_mask].sum())

    def junction_only_below(self, y_top, tol=1e-12):
        """Deepest point reached by a boundary that touches the charged edge.

        Below this depth no boundary is fed directly, so whatever the network
        holds there has crossed at least one triple junction.
        """
        touching = self.network_mask & (self.edges.ymax > y_top - tol)
        if not touching.any():
            return y_top
        return float(self.edges.ymin[touching].min())

    def check_orientations(self):
        """Check unit grain quaternions and cubic boundary-angle bounds.

        Identity orientations and zero-angle boundaries can be physically valid;
        source readback checks and grain IDs establish their provenance.
        """
        theta = self.theta[self.interior_mask]
        q = np.asarray(self.grain_quats)
        if (
            q.shape != (self.grain_ids[-1], 4)
            or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q, axis=1), 1.0)
            or not np.isfinite(theta).all()
            or np.any(theta < 0)
            or np.any(theta > 63)
        ):
            raise RuntimeError(
                "grain orientations or cubic disorientations are invalid"
            )
        return self.n_grains

    def report(self):
        """Summarize the EBSD mesh geometry and selected boundary network."""
        kept, total = int(self.network_mask.sum()), int(self.interior_mask.sum())
        theta = self.theta[self.network_mask]
        lx, ly = self.extent
        lines = [
            f"microstructure: {self.n_grains} grains from {self.base.name} "
            "(2D, raster mesh)",
            f"  domain                          : {lx:g} x {ly:g}",
            f"  edges                           : {len(self.edges)}",
            f"  on the specimen surface         : {self.surface_edges}",
            f"  grain boundaries (interior)     : {total}",
            f"  kept above {self.theta_min:g} deg    : {kept}",
        ]
        if kept:
            lines.append(
                f"  disorientation                  : "
                f"{theta.min():.1f} - {theta.max():.1f} deg (mean {theta.mean():.1f})"
            )
        lines += [
            f"  triple junctions                : {self.triple_junctions}",
            f"  boundary length                 : {self.network_measure:.4g}",
        ]
        return "\n".join(lines)
