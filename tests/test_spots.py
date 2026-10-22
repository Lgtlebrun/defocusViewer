import numpy as np
import pytest

from defocusviewer import spots

from .conftest import REAL_RUN


def test_table_without_fits(rundir):
    table = spots.table(rundir)
    assert len(table["rows"]) == 10 and not table["joint_fit"]
    assert all(sources == [] for sources in table["fits"].values())
    detail = spots.detail(rundir, "fa_35000_stability_1.npy", "on")
    assert detail["needs"] == "joint_fit" and detail["ghost"] == "on" and detail["ghost_box"] is None
    assert set(detail["images"]) == {"data"}


def test_ghost_inferred_from_the_last_series_run(tmp_path):
    log = tmp_path / "log.log"
    log.write_text("\n".join([
        "(t) [INFO] [Series_seq:ZernikeStabilityTimeSeries] : Fa: ghost mask applied",
        "(t) [INFO] [Main] : Analysis steps of x: ['fit', 'series']",
        "(t) [INFO] [Series_seq:ZernikeStabilityTimeSeries] : Fb: ghost mask applied",
        "(t) [INFO] [Main] : Analysis steps of x: ['model_fits']",
    ]))
    applied = spots._ghost_applied(log, log.stat().st_mtime)
    assert applied == {("ZernikeStabilityTimeSeries", "Fb")}      # Fa's line predates the last series run


def test_residual_frac_conventions():
    image = np.ones((5, 5))
    image[2, 2] = 10.0                     # the peak sits in the ghost box
    residual = np.zeros((5, 5))
    residual[0, 0], residual[2, 2] = 0.5, 4.0
    weights = np.ones((5, 5))
    weights[2, 2] = 0
    assert spots.residual_frac(residual, image, None, False) == pytest.approx(0.4)
    assert spots.residual_frac(residual, image, weights, False) == pytest.approx(0.05)    # fit_summary: image peak
    assert spots.residual_frac(residual, image, weights, True) == pytest.approx(0.5)      # fit_joint: peak outside the box


def test_seed_cut_or_padded_to_the_noll_terms():
    assert spots._seed_from([1, 2, 3, 4, 5, 6], (4, 5)) == [1, 2, 3, 4, 5]
    assert spots._seed_from([1, 2, 3, 4], (4, 5, 6)) == [1, 2, 3, 4, 0.0, 0.0]


@pytest.mark.skipif(not REAL_RUN.exists(), reason="needs the stability-fnum4-2 rundir")
@pytest.mark.parametrize("source", ["joint fit", "stability series", "train_end fit"])
def test_rendered_residual_matches_the_saved_fit(source):
    """The viewer's model of a spot gives the residual_frac the pipeline saved (joint fit unmasked excluded: its theta
    was overwritten by the refit)."""
    st = spots.state(REAL_RUN)
    images = [img for img, fits in st.fits.items() if any(f["source"] == source for f in fits)][::37][:4]
    assert images
    for img in images:
        saved = next(f for f in st.fits[img] if f["source"] == source)
        detail = spots.detail(REAL_RUN, img, saved["ghost"])
        assert detail["fit"]["source"] == source
        assert detail["residual_frac"] == pytest.approx(saved["residual_frac"], rel=1e-9)


@pytest.mark.skipif(not REAL_RUN.exists(), reason="needs the stability-fnum4-2 rundir")
def test_refit_reproduces_the_series(tmp_path):
    st = spots.state(REAL_RUN)
    img = next(img for img, fits in st.fits.items() if fits[0]["source"] == "stability series")
    saved = st.fits[img][0]
    out = spots.refit(REAL_RUN, img, saved["ghost"], tmp_path)
    assert out["r_fixed"] and out["params"][0] == pytest.approx(saved["params"][0])
    assert out["params"][3] == pytest.approx(saved["params"][3], abs=1e-3)               # c4
    assert spots.load_viewer_fit(tmp_path, img, saved["ghost"], st) == out


BASELINE = REAL_RUN.parent / "20261015-161039_stability-fnum4_A-baseline"
FNUM4 = REAL_RUN.parent / "20261015-161039_stability-fnum4"


@pytest.mark.skipif(not (BASELINE.exists() and (FNUM4 / "results" / "ghost_masks.hdf5").exists()),
                    reason="needs the fnum4 rundir, its ghost masks and its A-baseline copy")
def test_fiber_ghost_matches_the_pipeline_masks():
    """The viewer's ghost hunt on a run without masks lands on the boxes GhostMaskAnalysis found on the same data."""
    from DefocusTest.postprocessing import GhostMasks
    st = spots.state(BASELINE)
    pipeline = GhostMasks.load(spots.state(FNUM4).rundir)
    for fiber in ("Fa", "Fc", "pos_1"):
        box = spots.fiber_ghost(st, fiber, (121, 121), n_spots=5)
        assert abs(box["x0"] - pipeline.box(fiber, (121, 121))[0]) <= 1
        assert abs(box["y0"] - pipeline.box(fiber, (121, 121))[1]) <= 1
