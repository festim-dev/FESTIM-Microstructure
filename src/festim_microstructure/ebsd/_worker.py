"""Select and launch isolated EBSD workers from the parent process."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import _upxo
from ._archive import read_ebsd
from ._options import DEFAULT_WORKER_TIMEOUT, UpxoMeshOptions, _validate_timeout

WORKER_SCRIPT = Path(_upxo.__file__).resolve()
PYTHON_ENV = "FM_EBSD_PYTHON"


def _worker_command(executable, mode, path):
    return [executable, "-I", str(WORKER_SCRIPT), mode, str(path)]


def resolve_python(explicit=None, required=True):
    """Choose the EBSD interpreter: explicit path, FM_EBSD_PYTHON, then local.

    The named conda environment beside the current one is also discovered.
    An explicitly configured missing interpreter is an error, not a fallback.
    """
    if os.environ.get("FM_UPXO_PYTHON"):
        warnings.warn(
            f"Set {PYTHON_ENV} for the EBSD worker and unset FM_UPXO_PYTHON.",
            stacklevel=2,
        )
    configured = explicit or os.environ.get(PYTHON_ENV)
    if configured:
        executable = shutil.which(str(configured))
        if executable:
            return executable
    else:
        if sys.version_info >= (3, 13) and importlib.util.find_spec("upxo"):
            return sys.executable
        sibling = Path(sys.prefix).parent / "festim-microstructure" / "bin" / "python"
        if (
            sibling.is_file()
            and sibling.resolve() != Path(sys.executable).resolve()
            and os.access(sibling, os.X_OK)
        ):
            return str(sibling)
    if required:
        raise FileNotFoundError(
            "UPXO Python interpreter not found. Create and activate "
            f"environment.yml, or set {PYTHON_ENV} "
            "to an interpreter with UPXO installed."
        )
    return None


def probe_worker(executable=None, *, health=False, timeout=30.0):
    """Observe the selected worker before cache reuse, or check its EBSD APIs.

    Run in isolation just like the importer/mesher. No probe is cached, so an
    environment replacement at the same interpreter path invalidates results.
    """
    try:
        executable = resolve_python(executable)
    except FileNotFoundError as exc:
        raise RuntimeError(f"UPXO environment probe failed: {exc}") from exc
    mode = "--health" if health else "--environment"
    with tempfile.TemporaryDirectory(prefix="upxo-probe-") as scratch:
        output = Path(scratch) / "environment.json"
        try:
            result = subprocess.run(
                _worker_command(executable, mode, output),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise RuntimeError(f"UPXO environment probe failed: {detail}") from exc
        try:
            return json.loads(output.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"UPXO environment probe returned no valid report: {result.stdout}"
            ) from exc


def mesh_grains(
    archive,
    options: UpxoMeshOptions | None = None,
    *,
    workdir="results",
    stem="poly",
    unit=1e-6,
    force=True,
    python=None,
    timeout=DEFAULT_WORKER_TIMEOUT,
):
    """Mesh a native EBSD archive and write tagged SI and diagnostic outputs.

    Cache reuse requires matching input bytes, options, units, interpreter and
    observed installed distributions and worker code, plus output checksums.
    Failed runs preserve accepted outputs and keep a log and validation report.
    """
    _validate_timeout(timeout)

    if options is None:
        options = UpxoMeshOptions()
    elif not isinstance(options, UpxoMeshOptions):
        raise TypeError("mesh_grains options must be UpxoMeshOptions")
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("unit must be finite and positive")
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("stem must be a filename without directory components")
    archive = Path(archive).resolve()
    data = read_ebsd(archive)
    labels, vox = data.labels, np.asarray(data.vox)
    executable = resolve_python(python)
    base = Path(workdir).resolve() / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    settings = asdict(options)
    identity = {
        "protocol": _upxo._PROTOCOL,
        "upxo_revision": _upxo.UPXO_REVISION,
        "worker_environment": probe_worker(executable, timeout=timeout),
        "worker_sha256": hashlib.sha256(WORKER_SCRIPT.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "options": settings,
        "unit": unit,
    }
    manifest = Path(f"{base}-upxo.json")
    if not force and manifest.is_file():
        try:
            saved = json.loads(manifest.read_text())
            outputs = saved["outputs"]
            if (
                saved["identity"] == identity
                and outputs
                and all(
                    (base.parent / name).is_file()
                    and hashlib.sha256((base.parent / name).read_bytes()).hexdigest()
                    == digest
                    for name, digest in outputs.items()
                )
            ):
                return base
        except (ValueError, KeyError, OSError):
            pass
    with tempfile.TemporaryDirectory(
        prefix=f".{stem}-upxo-", dir=base.parent
    ) as scratch:
        stage = Path(scratch)
        np.savez(stage / "input.npz", labels=labels.astype(np.int32), vox=vox)
        request = stage / "request.json"
        _upxo._save_json(request, {"options": settings, "unit": unit, "stem": stem})
        log = Path(f"{base}-upxo.log")
        with log.open("w") as stream:
            try:
                subprocess.run(
                    _worker_command(executable, "--worker", request),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                failed = stage / f"{stem}-validation.json"
                reason = "worker exited before validation"
                if failed.is_file():
                    shutil.copyfile(failed, f"{base}-failed-validation.json")
                    reason = json.loads(failed.read_text()).get("error", reason)
                if isinstance(exc, subprocess.TimeoutExpired):
                    reason = f"exceeded {timeout:g} seconds"
                raise RuntimeError(f"UPXO meshing failed: {reason}; see {log}") from exc
        shutil.copyfile(archive, stage / f"{stem}-ebsd.npz")
        outputs = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in stage.iterdir()
            if p.name.startswith(stem) and p.is_file()
        }
        for name in outputs:
            (stage / name).replace(base.parent / name)
        _upxo._save_json(manifest, {"identity": identity, "outputs": outputs})
    return base


def import_ebsd(
    ctf,
    window,
    indexed,
    settings,
    log=print,
    *,
    python=None,
    timeout=DEFAULT_WORKER_TIMEOUT,
):
    """Canonicalize the cropped, filtered CTF and invoke UPXO's EBSDReader.

    DefDAP's text loader reshapes file rows and uses a scalar XStep. Reorder
    by coordinates here, supply absent optional quality columns, and retain
    both pixel spacings in the native archive. Phase/quality rejections become
    phase zero before detection, so they cannot bridge surviving grains.
    """
    executable = resolve_python(python)
    with tempfile.TemporaryDirectory(prefix="upxo-import-") as scratch:
        stage = Path(scratch)
        ny, nx = indexed.shape
        full_ny, full_nx = ctf.shape
        ix = np.rint(ctf["X"] / ctf.header["XStep"]).astype(int)
        iy = np.rint(ctf["Y"] / ctf.header["YStep"]).astype(int)
        inside = (ix >= 0) & (ix < full_nx) & (iy >= 0) & (iy < full_ny)
        columns = [
            "Phase",
            "X",
            "Y",
            "Bands",
            "Error",
            "Euler1",
            "Euler2",
            "Euler3",
            "MAD",
            "BC",
            "BS",
        ]
        arrays = {}
        for name in columns:
            array = np.zeros(ctf.shape)
            if ctf.has(name):
                array[iy[inside], ix[inside]] = ctf[name][inside]
            arrays[name] = array[window].copy()
        arrays["Phase"] = indexed.astype(int)
        # Invalid, rejected orientations do not participate in detection or
        # diagnostics; retain finite rejected measurements for error panels.
        for name in ("Euler1", "Euler2", "Euler3"):
            arrays[name][~np.isfinite(arrays[name])] = 0
        yy, xx = np.indices((ny, nx))
        arrays["X"], arrays["Y"] = xx * ctf.header["XStep"], yy * ctf.header["YStep"]
        lines = ctf.path.read_text().splitlines()
        phase_row = next(
            i for i, line in enumerate(lines) if line.split("\t")[0] == "Phases"
        )
        phase_line = lines[phase_row + settings.phase]
        canonical = stage / "filtered.ctf"
        with canonical.open("w") as stream:
            stream.write(
                f"Channel Text File\nXCells\t{nx}\nYCells\t{ny}\n"
                f"XStep\t{ctf.header['XStep']}\nYStep\t{ctf.header['YStep']}\n"
                f"Phases\t1\n{phase_line}\n"
            )
            stream.write("\t".join(columns) + "\n")
            np.savetxt(
                stream,
                np.column_stack([arrays[col].ravel() for col in columns]),
                fmt="%.12g",
                delimiter="\t",
            )
        request = stage / "request.json"
        _upxo._save_json(
            request,
            {"min_pixels": settings.min_pixels, "threshold": settings.threshold},
        )
        logfile = stage / "import.log"
        with logfile.open("w") as stream:
            try:
                subprocess.run(
                    _worker_command(executable, "--import", request),
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    f"UPXO/DefDAP import failed: {logfile.read_text()[-6000:]}"
                ) from exc
        with np.load(stage / "import.npz", allow_pickle=False) as imported:
            result = {
                key: imported[key]
                for key in ("source_labels", "pixel_quats", "grain_quats")
            }
        result["packages"] = json.loads((stage / "packages.json").read_text())
        if log:
            log(
                f"  importer: UPXO {result['packages']['upxo']}, "
                f"DefDAP {result['packages']['defdap']}"
            )
        return result
