"""Draw clean xz sections of the three saved 3D Voronoi boundary networks.

Run ``python examples/plot_voronoi_polycrystal_3d.py`` after the 3D example.
Reads only mesh geometry and connectivity from its network BP files. Writes
three individual black-on-white PNGs and a labelled comparison beside them,
under results/voronoi_polycrystal_3d/. Each section passes through y=L/2;
x is horizontal and z is vertical. Requires ADIOS2, NumPy and Matplotlib.
"""

from pathlib import Path

import adios2
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "voronoi_polycrystal_3d"
CASES = (
    ("voronoi3d", "Equiaxed"),
    ("voronoi3d_parallel", "Parallel to loading"),
    ("voronoi3d_perpendicular", "Perpendicular to loading"),
)


def read_boundary_mesh(path):
    """Read the first mesh frame, joining rank-local triangle blocks."""
    points, triangles = [], []
    offset = 0
    with adios2.FileReader(str(path)) as reader:
        blocks = reader.all_blocks_info("geometry")[0]
        for block_id in range(len(blocks)):
            xyz = reader.read(
                "geometry", block_id=block_id, step_selection=[0, 1]
            ).reshape(-1, 3)
            cells = reader.read(
                "connectivity", block_id=block_id, step_selection=[0, 1]
            ).reshape(-1, 4)
            if not np.all(cells[:, 0] == 3):
                raise ValueError(f"{path}: expected a triangular boundary network")
            points.append(xyz)
            triangles.append(cells[:, 1:] + offset)
            offset += len(xyz)
    return np.concatenate(points), np.concatenate(triangles)


def xz_section(points, triangles, y):
    """Intersect boundary triangles with y=constant, retaining only xz traces."""
    tol = 1e-10 * np.ptp(points, axis=0).max()
    segments = {}
    for triangle in points[triangles]:
        distances = triangle[:, 1] - y
        distances[np.abs(distances) <= tol] = 0.0
        if np.all(distances > 0) or np.all(distances < 0):
            continue
        if np.all(distances == 0):
            raise ValueError(
                "The slice coincides with a boundary face; choose another y"
            )
        hits = [point[[0, 2]] for point in triangle[distances == 0]]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = distances[a], distances[b]
            if da * db < 0:
                fraction = da / (da - db)
                point = triangle[a] + fraction * (triangle[b] - triangle[a])
                hits.append(point[[0, 2]])
        if len(hits) != 2 or np.linalg.norm(hits[1] - hits[0]) <= tol:
            continue
        segment = np.asarray(hits)
        key = tuple(sorted(tuple(np.rint(point / tol).astype(int)) for point in hits))
        segments.setdefault(key, segment)
    if not segments:
        raise ValueError(f"No grain boundaries intersect y={y:g}")
    return np.asarray(list(segments.values()))


def draw_section(ax, segments, bounds):
    """Draw black boundary lines and a square specimen frame on white."""
    ax.set_facecolor("white")
    ax.add_collection(
        LineCollection(segments, colors="black", linewidths=1.75, capstyle="round")
    )
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 2], bounds[1, 2])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3.0)


def main():
    """Write three uncluttered sections and one labelled comparison image."""
    sections = []
    for prefix, title in CASES:
        source = OUTPUT_DIR / f"{prefix}_network.bp"
        if not source.is_dir():
            raise FileNotFoundError(f"Run voronoi_polycrystal_3d.py first: {source}")
        points, triangles = read_boundary_mesh(source)
        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        y = float(bounds[:, 1].mean())
        segments = xz_section(points, triangles, y)
        sections.append((title, segments, bounds))

        fig = plt.figure(figsize=(6, 6), facecolor="white")
        ax = fig.add_axes((0.035, 0.035, 0.93, 0.93))
        draw_section(ax, segments, bounds)
        path = OUTPUT_DIR / f"{prefix}_xz_slice.png"
        fig.savefig(path, dpi=400, facecolor="white")
        plt.close(fig)
        print(f"{path.name}: y={y:g}, {len(segments)} boundary trace segments")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.11, top=0.83, wspace=0.12)
    for ax, (title, segments, bounds) in zip(axes, sections, strict=True):
        draw_section(ax, segments, bounds)
        ax.set_title(title, fontsize=18, pad=12)
        ax.set_xlabel("x", fontsize=18, labelpad=8)
    axes[0].set_ylabel("z", fontsize=18, rotation=0, labelpad=14)
    fig.suptitle("Grain boundaries: central xz section (y = L/2)", fontsize=17, y=0.95)
    path = OUTPUT_DIR / "voronoi3d_xz_slices.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(path.name)


if __name__ == "__main__":
    main()
