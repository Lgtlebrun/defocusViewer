from pathlib import Path

import pandas as pd

from defocusviewer import jointfit

from .conftest import make_rundir


def test_fnum_from_the_run_name():
    assert jointfit.guess_fnum(Path("/x/20261017-175317_stability-fnum4-2")) == (4.0, "run name")
    assert jointfit.guess_fnum(Path("/x/20261014-185836_stability-rawframes-apf1p8")) == (1.8, "run name")
    assert jointfit.guess_fnum(Path("/x/20261015-155646_hexagonal-aperture-fit")) == (None, "unknown")


def test_fnum_from_the_config(rundir):
    assert jointfit.guess_fnum(rundir) == (4.0, "config")


def test_positioners_matched_by_centre():
    """pos_0 and pos_1 swap labels between the two aperture runs of 2026-10-15 (centres from their label maps)."""
    circular = {"Fa": (567, 1782), "pos_0": (1869, 1755), "pos_1": (2033, 1915), "pos_2": (1861, 2304)}
    hexagonal = {"Fa": (566, 1781), "pos_0": (2028, 1913), "pos_1": (1865, 1754), "pos_2": (1862, 2305)}
    assert jointfit.match_fibers(hexagonal, circular) == {"Fa": "Fa", "pos_0": "pos_1", "pos_1": "pos_0", "pos_2": "pos_2"}
    assert jointfit.match_fibers({"pos_9": (10, 10)}, circular) == {"pos_9": None}


def test_spot_table_without_positions(config):
    path = make_rundir(config.data_root, "20261015-155646_hexagonal-aperture-fit")
    df = pd.read_hdf(path / "data" / "dataframe.hdf5", key="data").drop(columns="x_position")
    df.to_hdf(path / "data" / "dataframe.hdf5", key="data", format="table")
    assert (jointfit.spot_table(path)["x_position"] == 1).all()
