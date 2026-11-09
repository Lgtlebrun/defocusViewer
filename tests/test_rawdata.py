import os
import time
from types import SimpleNamespace

import pytest

from defocusviewer import __main__ as cli
from defocusviewer import rawdata

from .conftest import make_rundir

NAME = "20261017-175317_stability-test"


@pytest.fixture(autouse=True)
def nothing_in_use(monkeypatch):
    """No test asks the real hosts or Slurm what is running (test_in_use_* fake those instead)."""
    real = rawdata.in_use
    monkeypatch.setattr(rawdata, "in_use", lambda config, ignore_fetch=False: [])
    return real


@pytest.fixture
def rig(config, tmp_path, monkeypatch):
    """A fake rig: a local folder of runs, reached by plain rsync (no host)."""
    folder = tmp_path / "rig" / "Orbray6" / "Defocus"
    make_rundir(tmp_path / "rig", NAME)
    remote = SimpleNamespace(_target=None, _remote_folder=folder, _SSH_OPTS=[],
                             _remote_test=lambda path, flag: path.is_dir())
    monkeypatch.setattr(rawdata, "_remote", lambda config, device: remote)
    return folder / NAME


def test_in_use_names_jobs_processes_and_slurm(config, monkeypatch, nothing_in_use):
    from defocusviewer import jobs
    registry = jobs.Registry(config.jobs_dir)
    registry.save({"id": "1", "run": f"MPS6/Defocus/{NAME}_tophatbl-circle", "label": "fit", "state": "running"})
    registry.save({"id": "2", "run": "MPS6/Defocus/fetched", "label": "fetch (watchdog)", "state": "running"})
    registry.save({"id": "3", "run": "MPS6/Defocus/old", "label": "fit", "state": "done"})
    ps = ("python -m DefocusTest.analysis.run_stability 20261101-000000_x --device mps6\n"
          "python -m defocusviewer fetch MPS6 20261102-000000_y\n"
          "python -m defocusviewer move-data 20261103-000000_z\nbash\n")
    monkeypatch.setattr(rawdata.resources, "_shell", lambda cmd, host=None: ps)
    monkeypatch.setattr(rawdata.resources, "slurm_jobs", lambda: {"9": {"name": "20261104-000000_w"}})
    names = nothing_in_use(config)
    assert {f"{NAME}_tophatbl-circle", "fetched", "20261101-000000_x", "20261102-000000_y", "20261104-000000_w"} <= set(names)
    assert "old" not in names and "20261103-000000_z" not in names       # finished job, move-data itself
    ignoring = nothing_in_use(config, ignore_fetch=True)
    assert "fetched" not in ignoring and "20261102-000000_y" not in ignoring


def _files(folder):
    return sorted(p.name for p in folder.iterdir())


def test_fetch_new_run_puts_data_on_the_raw_root(config, rig):
    local = rawdata.fetch(config, "Orbray6", NAME)
    raw = config.raw_root / "Orbray6" / "Defocus" / NAME / "data"
    assert local == config.data_root / "Orbray6" / "Defocus" / NAME
    assert (local / "data").is_symlink() and (local / "data").resolve() == raw.resolve()
    assert _files(raw) == _files(rig / "data")
    assert (local / "meta" / "_finished").exists() and (local / "plot" / "spots_centers.pdf").exists()
    assert not list(local.parent.glob(".*.partial"))


def test_forced_fetch_keeps_the_link_and_brings_new_files(config, rig):
    local = rawdata.fetch(config, "Orbray6", NAME)
    (rig / "data" / "extra.npy").write_bytes(b"x")
    (rig / "meta" / "note.txt").write_text("late")
    rawdata.fetch(config, "Orbray6", NAME)                  # no force: a local rundir is left alone
    assert not (local / "meta" / "note.txt").exists()
    rawdata.fetch(config, "Orbray6", NAME, force=True)
    assert (local / "data").is_symlink()
    assert (local / "data" / "extra.npy").exists() and (local / "meta" / "note.txt").exists()


def test_forced_fetch_over_a_posutils_copy_switches_to_the_rig_data(config, rig):
    local = make_rundir(config.data_root, NAME)             # what posUtils' ensureRunDir leaves: a real data/
    (local / "data" / "partial.npy").write_bytes(b"half")
    rawdata.fetch(config, "Orbray6", NAME, force=True)
    assert (local / "data").is_symlink() and not (local / ".data.moved").exists()
    assert _files(local / "data") == _files(rig / "data")    # the rig's files, not the stale local copy


def test_move_switches_after_verifying(config):
    local = make_rundir(config.data_root, NAME)
    before = _files(local / "data")
    result = rawdata.move(config, local, names=[])
    assert result.startswith("moved")
    assert (local / "data").is_symlink() and _files(local / "data") == before
    assert not (local / ".data.moved").exists()
    assert rawdata.move(config, local, names=[]) == "data/ already a link"


def test_move_leaves_the_run_when_the_copy_differs(config, monkeypatch):
    local = make_rundir(config.data_root, NAME)
    monkeypatch.setattr(rawdata, "_differences", lambda source, target: [">f.st...... fa_35000_stability_0.npy"])
    assert rawdata.move(config, local, names=[]).startswith("FAILED")
    assert (local / "data").is_dir() and not (local / "data").is_symlink()


def test_move_skips_a_run_whose_variant_is_analysed(config):
    local = make_rundir(config.data_root, NAME)
    assert rawdata.move(config, local, names=[f"{NAME}_tophatbl-circle"]).startswith("skipped")
    assert not (local / "data").is_symlink()
    assert rawdata.busy(NAME, [f"{NAME}x", "other"]) == []      # only the run itself and <name>_<suffix>


def test_move_dry_run_changes_nothing(config):
    local = make_rundir(config.data_root, NAME)
    assert rawdata.move(config, local, names=[], dry_run=True).startswith("would move")
    assert not (local / "data").is_symlink() and not config.raw_root.exists()


def test_sweep_finds_only_old_real_data_dirs(config):
    old = make_rundir(config.data_root, NAME)
    young = make_rundir(config.data_root, "20261018-000000_stability-young")
    linked = make_rundir(config.data_root, "20261019-000000_stability-linked")
    rawdata.move(config, linked, names=[])
    past = time.time() - 2 * rawdata.SWEEP_MIN_AGE_S
    os.utime(old / "data", (past, past))
    assert rawdata.real_data_dirs(config, rawdata.SWEEP_MIN_AGE_S) == [old]
    assert set(rawdata.real_data_dirs(config)) == {old, young}


def test_move_data_command_moves_named_runs(config, capsys):
    local = make_rundir(config.data_root, NAME)
    cli.move_data(config, [NAME], every=False, dry_run=False)
    assert (local / "data").is_symlink()
    assert "moved" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.move_data(config, ["no-such-run"], every=False, dry_run=False)
