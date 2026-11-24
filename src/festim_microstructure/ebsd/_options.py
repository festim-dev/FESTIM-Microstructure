"""Quality, crop, frame and UPXO/DefDAP import settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np

from ._orientation import euler_bunge_to_quat
from ._provenance import read_provenance

__all__ = [
    "CUBIC_LAUE",
    "EbsdOptions",
    "Settings",
    "UpxoMeshOptions",
    "settings_from_provenance",
]


DEFAULT_WORKER_TIMEOUT = 600.0


def _validate_timeout(timeout):
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker timeout must be finite and positive")


def settings_from_provenance(path):
    """Rebuild the `Settings` of a conversion from its provenance json."""
    rec = read_provenance(path)
    known = {f.name for f in dataclass_fields(Settings)}
    return Settings(**{k: v for k, v in rec.items() if k in known})


#: Laue 11 (m-3m) only: Laue 10 (m-3) has 12 proper rotations, not 24.
CUBIC_LAUE = (11,)


@dataclass(frozen=True)
class Settings:
    """Configuration for :func:`convert`, including saved provenance."""

    ctf: str
    #: Crop before segmentation with (xmin, xmax, ymin, ymax) in source units.
    crop: tuple[float, float, float, float] | list[float] | None = None
    phase: int = 1  #: phase to keep
    threshold: float = 10.0  #: grain boundary misorientation, degrees
    max_mad: float = 1.0  #: MAD cutoff
    min_bands: int = 0  #: minimum Bands; 0 disables the test
    allow_error: bool = False  #: keep points whose Error column is non-zero
    min_pixels: int = 5  #: discard grains smaller than this
    #: Multiply both source pixel spacings by this factor.
    scale: float = 1.0
    #: Proper frame change: mirror rows and rotate orientations 180 deg about x.
    flip_y: bool = False
    #: Rotation taking the Euler frame onto the map frame (Bunge degrees).
    euler_correction: tuple[float, float, float] | list[float] | np.ndarray | None = (
        None
    )
    #: Assign unindexed/pruned pixels to the nearest surviving grain for geometry.
    #: Original membership is retained and these pixels do not affect grain means.
    fill: bool = True

    def __post_init__(self):
        if self.crop is not None:
            crop = np.asarray(self.crop, dtype=float)
            if crop.shape != (4,):
                raise ValueError("crop must contain (xmin, xmax, ymin, ymax)")
            if not np.isfinite(crop).all():
                raise ValueError("crop bounds must be finite")
            object.__setattr__(self, "crop", tuple(crop.tolist()))
        if not isinstance(self.phase, int) or self.phase < 1:
            raise ValueError("phase must be a positive integer")
        if not isinstance(self.min_pixels, int) or self.min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer")
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 63:
            raise ValueError("threshold must be between 0 and 63 degrees")
        if not np.isfinite(self.max_mad) or self.max_mad < 0 or self.min_bands < 0:
            raise ValueError("quality cutoffs must be finite and nonnegative")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if self.euler_correction is not None:
            angles = np.asarray(self.euler_correction, dtype=float)
            if angles.shape != (3,) or not np.isfinite(angles).all():
                raise ValueError(
                    "euler_correction must be three finite Bunge angles in degrees"
                )
            object.__setattr__(self, "euler_correction", tuple(angles.tolist()))

    @property
    def euler_correction_quat(self):
        """``euler_correction`` as a unit quaternion, or ``None``."""
        if self.euler_correction is None:
            return None
        return euler_bunge_to_quat(*self.euler_correction)

    @property
    def unit(self):
        """Return the display unit implied by the spatial scale."""
        return "um" if np.isclose(self.scale, 1.0) else f"x{self.scale:g} um"


@dataclass(frozen=True)
class UpxoMeshOptions:
    """UPXO mesh sizes and controlled smoothing, measured in pixel coordinates.

    ``smoothing='none'`` preserves the raster boundaries exactly. Taubin uses
    explicit pixel seeds and disables grain merging and diagonal repairs.
    ``thin_grain_px`` enables optional thin-grain protection without changing
    pixel labels. For unfilled voids, use ``smoothing="none"``.
    Geometry that fails the preservation checks raises before publication.
    """

    smoothing: str = "taubin"
    smooth_iter: int = 5
    smooth_lambda: float = 0.25
    smooth_mu: float = -0.265
    thin_grain_px: float = 0.0
    mesh_size_gb: float = 0.75
    mesh_size_bulk: float = 4.5
    n_threads: int = 1
    max_area_change: float = 0.05
    max_boundary_displacement: float = 1.0

    def __post_init__(self):
        if self.smoothing not in ("none", "taubin"):
            raise ValueError("smoothing must be 'none' or 'taubin'")
        for name in ("smooth_iter", "n_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.smooth_iter < 0 or self.n_threads < 1:
            raise ValueError("smooth_iter must be >= 0 and n_threads must be >= 1")
        for name in ("mesh_size_gb", "mesh_size_bulk"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_area_change", "max_boundary_displacement", "thin_grain_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.smooth_lambda < 1 or not -1 < self.smooth_mu < 0:
            raise ValueError(
                "Taubin requires 0 < smooth_lambda < 1 and -1 < smooth_mu < 0"
            )


@dataclass(frozen=True)
class EbsdOptions:
    """Full CTF → UPXO/DefDAP grains → UPXO mesh → FESTIM checks."""

    import_settings: Settings
    unit: float = 1e-6
    theta_min: float = 10.0
    mesh: UpxoMeshOptions | None = field(default_factory=UpxoMeshOptions)
    stem: str = "poly"
    check_images: bool = True
    timeout: float = DEFAULT_WORKER_TIMEOUT

    def __post_init__(self):
        if not isinstance(self.import_settings, Settings):
            raise TypeError("EbsdOptions.import_settings must be Settings")
        if self.mesh is None:
            object.__setattr__(self, "mesh", UpxoMeshOptions())
        elif not isinstance(self.mesh, UpxoMeshOptions):
            raise TypeError("EbsdOptions.mesh must be UpxoMeshOptions")
        _validate_timeout(self.timeout)
        if not np.isfinite(self.unit) or self.unit <= 0:
            raise ValueError("unit must be finite and positive")
        if not np.isfinite(self.theta_min) or not 0 <= self.theta_min < 63:
            raise ValueError("theta_min must be between 0 and 63 degrees")
        if (
            not self.stem
            or Path(self.stem).name != self.stem
            or self.stem in (".", "..")
        ):
            raise ValueError("stem must be a filename without directory components")
