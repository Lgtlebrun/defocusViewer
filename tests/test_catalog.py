import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from defocusviewer import catalog, plots

from .conftest import make_rundir


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


def test_fnum_from_the_run_name():
    assert catalog.guess_fnum(Path("/x/20261017-175317_stability-fnum4-2")) == (4.0, "run name")
    assert catalog.guess_fnum(Path("/x/20261014-185836_stability-rawframes-apf1p8")) == (1.8, "run name")
    assert catalog.guess_fnum(Path("/x/20261015-155646_hexagonal-aperture-fit")) == (None, "unknown")
    assert catalog.fnum_from_name("20261106-121415_tracking-mount-fnummin") is None
    assert catalog.fnum_from_name("20261027-165900_stability-mount-fnummin_tophatbl-circle-f1p8_k81") == 1.8


def test_fnum_from_the_config(rundir):
    assert catalog.guess_fnum(rundir) == (4.0, "config")
    config = rundir / "meta" / "config.yaml"
    config.write_text("run: {device: Orbray6}\nfit: {fnum: 1.8}\n")
    os.utime(config, (1e9, 1e9))      # a new mtime: the cached value is not reused
    assert catalog.guess_fnum(rundir) == (1.8, "config")


def test_tags_and_analysed(config, rundir):
    original = make_rundir(config.data_root, "20261106-121415_tracking-mount-fnummin", device="MPS6")
    variant = make_rundir(config.data_root, "20261106-121415_tracking-mount-fnummin_tophatbl-circle", device="MPS6")
    (variant / "results" / "theta_shared.npy").touch()
    make_rundir(config.data_root, "20261113-163434_luminosity-dim-leds", device="MPS6")
    remote = mock.Mock(snapshot=lambda: {"runs": {"MPS6/Defocus/20261023-174520_Pointing-fnum8": {"finished": True}},
                                         "updated": 0.0, "error": None})
    data = catalog.run_list(config, remote)
    rows = {r["name"]: r for r in data["runs"]}
    assert data["tag_words"] == ["stability", "tracking", "pointing", "detection", "luminosity"]
    # the fixture's dataframe holds a stability routine: every local run here has that flow, so that tag too
    assert rows[original.name]["tags"] == ["tracking", "stability", "f/4"]
    assert rows[rundir.name]["tags"] == ["stability", "f/4"]
    assert rows["20261023-174520_Pointing-fnum8"]["tags"] == ["pointing", "f/8"]
    assert {name: r["analysed"] for name, r in rows.items()} == {
        rundir.name: False, original.name: True, variant.name: True, "20261113-163434_luminosity-dim-leds": False,
        "20261023-174520_Pointing-fnum8": False}
