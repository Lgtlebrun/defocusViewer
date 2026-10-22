import subprocess
from unittest import mock

import pytest

from defocusviewer import catalog, plots


def test_local_runs_and_resolve(config, rundir):
    (config.data_root / "Orbray6" / "Defocus" / "not-a-rundir").mkdir()
    runs = catalog.local_runs(config)
    assert runs == {"Orbray6/Defocus/20261017-175317_stability-test": rundir}
    assert catalog.resolve(config, "Orbray6/Defocus/20261017-175317_stability-test") == rundir
    with pytest.raises(KeyError):
        catalog.resolve(config, "../../etc")
    assert catalog.device_of(config, "Orbray6/Defocus/x") == "Orbray6"
    assert catalog.device_of(config, "defocusApSim-b/x") is None


def test_device_ids():
    assert catalog.device_id("calibration_stand") == "calib"
    assert catalog.device_id("MPS6") == "mps6"


def test_summary_and_steps(rundir):
    summary = catalog.summary(rundir)
    assert summary["flow"] == "stability" and summary["finished"] and summary["steps_done"] == []
    (rundir / "results" / "joint_fit_diag_records.hdf5").touch()
    (rundir / "results" / "Fa_stability_timeseries_zernike.npz").touch()
    assert catalog.steps_done(rundir, "stability") == ["fit", "series"]


def test_remote_listing_parsed(config):
    listed = ("Orbray6/Defocus/20261017-175317_stability-fnum4-2\nOrbray6/Defocus/notes.txt\n"
              "MPS6/Defocus/20261001-101010_sweep\n__finished__\nOrbray6/Defocus/20261017-175317_stability-fnum4-2/meta/_finished\n")
    done = subprocess.CompletedProcess([], 0, stdout=listed, stderr="")
    with mock.patch("subprocess.run", return_value=done):
        runs = catalog.RemoteRuns(config).list_remote()
    assert runs == {"Orbray6/Defocus/20261017-175317_stability-fnum4-2": {"device": "Orbray6", "finished": True},
                    "MPS6/Defocus/20261001-101010_sweep": {"device": "MPS6", "finished": False}}


def test_plot_paths_stay_inside(rundir):
    assert [p["path"] for p in plots.list_plots(rundir)] == ["spots_centers.pdf"]
    assert plots.inside(rundir / "plot", "spots_centers.pdf").name == "spots_centers.pdf"
    for bad in ("../meta/config.yaml", "/etc/passwd", "healthchecks"):
        with pytest.raises(ValueError):
            plots.inside(rundir / "plot", bad)
