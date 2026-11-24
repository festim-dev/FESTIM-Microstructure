"""Interpreter selection, worker launches, and cache handling."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from festim_microstructure.ebsd import Settings, _upxo, _worker, mesh_grains
from festim_microstructure.ebsd._archive import EbsdData, write_ebsd


def test_configured_interpreter_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("FM_EBSD_PYTHON", "/does/not/exist/python")
    assert _worker.resolve_python(required=False) is None
    with pytest.raises(FileNotFoundError, match="FM_EBSD_PYTHON"):
        _worker.resolve_python()


def test_legacy_interpreter_override_warns(monkeypatch):
    monkeypatch.setenv("FM_UPXO_PYTHON", "/retired/worker/python")
    monkeypatch.setenv("FM_EBSD_PYTHON", _worker.sys.executable)
    with pytest.warns(UserWarning, match="FM_EBSD_PYTHON.*unset FM_UPXO_PYTHON"):
        assert _worker.resolve_python() == _worker.sys.executable


@pytest.mark.parametrize(
    "version,has_upxo",
    [((3, 12), True), ((3, 13), False), ((3, 13), True)],
)
def test_active_interpreter_requires_python_313_and_upxo(
    monkeypatch, version, has_upxo
):
    monkeypatch.delenv("FM_EBSD_PYTHON", raising=False)
    monkeypatch.setattr(_worker.sys, "version_info", version)
    monkeypatch.setattr(
        _worker.importlib.util, "find_spec", lambda name: object() if has_upxo else None
    )

    if version >= (3, 13) and has_upxo:
        assert _worker.resolve_python() == _worker.sys.executable
    else:
        assert _worker.resolve_python(required=False) is None
        with pytest.raises(FileNotFoundError, match=r"environment\.yml"):
            _worker.resolve_python()


def test_worker_failure_preserves_accepted_outputs(tmp_path, monkeypatch):
    labels = np.ones((4, 4), dtype=np.int32)
    source = tmp_path / "map.npz"
    write_ebsd(source, EbsdData(labels, np.array([[1.0, 0, 0, 0]]), (1, 1)))
    mesh = tmp_path / "poly.msh4"
    mesh.write_bytes(b"previous accepted mesh")
    monkeypatch.setattr(_worker, "resolve_python", lambda: "/fake/python")
    monkeypatch.setattr(_worker, "probe_worker", lambda *a, **kw: {"python": "fake"})

    def failed_worker(command, **kwargs):
        stage = Path(command[-1]).parent
        (stage / "poly-validation.json").write_text(
            json.dumps(
                {
                    "accepted": False,
                    "error": "geometry changed",
                }
            )
        )
        (stage / "poly.msh4").write_bytes(b"rejected mesh")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(_worker.subprocess, "run", failed_worker)
    with pytest.raises(RuntimeError, match="UPXO meshing failed"):
        mesh_grains(source, workdir=tmp_path)
    assert mesh.read_bytes() == b"previous accepted mesh"
    assert (
        json.loads((tmp_path / "poly-failed-validation.json").read_text())["accepted"]
        is False
    )
    assert not list(tmp_path.glob(".poly-upxo-*"))


@pytest.mark.parametrize(
    "error",
    [
        OSError("missing"),
        subprocess.TimeoutExpired("probe", 1),
        subprocess.CalledProcessError(1, "probe", stderr="wrong version"),
    ],
)
def test_environment_probe_errors_are_actionable(monkeypatch, error):
    monkeypatch.setattr(_worker, "resolve_python", lambda: "/fake/python")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_worker.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="environment probe failed"):
        _worker.probe_worker()


def test_import_identity_observes_environment_variable_and_package_changes(
    tmp_path, ctf_writer, monkeypatch
):
    from festim_microstructure.ebsd._grains import CtfConversion

    ctf = ctf_writer(tmp_path / "map.ctf", np.zeros((4, 4, 3)))
    observed = {"version": "first"}

    def probe(**kwargs):
        return {"executable": _worker.resolve_python(), **observed}

    monkeypatch.setattr(_worker, "probe_worker", probe)
    conversion = CtfConversion(Settings(ctf=str(ctf)))
    monkeypatch.setenv("FM_EBSD_PYTHON", "/usr/bin/python3")
    first = conversion.identity()
    monkeypatch.setenv("FM_EBSD_PYTHON", "/bin/sh")
    second = conversion.identity()
    assert first != second
    observed["version"] = "replacement"
    assert second != conversion.identity()


def test_mesh_cache_checks_worker_digest_and_environment(tmp_path, monkeypatch):
    """Reuse a matching manifest and refresh old digests or environments."""
    source = tmp_path / "map.npz"
    write_ebsd(
        source, EbsdData(np.ones((4, 4), dtype=int), np.array([[1.0, 0, 0, 0]]), (1, 1))
    )
    observed = {"version": "first"}
    monkeypatch.setattr(_worker, "resolve_python", lambda: "/fake/python")
    monkeypatch.setattr(_worker, "probe_worker", lambda *a, **kw: dict(observed))
    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        stage = Path(command[-1]).parent
        (stage / "poly.msh4").write_text("accepted mesh")

    monkeypatch.setattr(_worker.subprocess, "run", worker)
    base = mesh_grains(source, workdir=tmp_path, force=False)
    mesh_grains(source, workdir=tmp_path, force=False)
    assert len(calls) == 1
    manifest = Path(f"{base}-upxo.json")
    saved = json.loads(manifest.read_text())
    worker_code = Path(_upxo.__file__).read_bytes()
    assert saved["identity"]["worker_sha256"] == hashlib.sha256(worker_code).hexdigest()

    saved["identity"]["worker_sha256"] = "old-digest"
    manifest.write_text(json.dumps(saved))
    mesh_grains(source, workdir=tmp_path, force=False)
    mesh_grains(source, workdir=tmp_path, force=False)
    assert len(calls) == 2

    observed["version"] = "replacement"
    mesh_grains(source, workdir=tmp_path, force=False)
    assert len(calls) == 3


def test_environment_probe_rejects_missing_interpreter(monkeypatch):
    monkeypatch.setenv("FM_EBSD_PYTHON", "/does/not/exist")
    with pytest.raises(RuntimeError, match="environment probe failed"):
        _worker.probe_worker()
