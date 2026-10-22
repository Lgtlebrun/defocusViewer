import json

import numpy as np
import pytest

from defocusviewer import catalog, variants

RUN = "Orbray6/Defocus/20261017-175317_stability-test"


def test_variant_shares_the_data_and_starts_without_analysis(config, rundir):
    np.save(rundir / "results" / "equalizer_raw_image.npy", np.zeros((3, 3)))
    np.save(rundir / "results" / "theta_shared.npy", 4.8)                    # an analysis result: not copied
    (rundir / "log.log").write_text("(t) acquisition\n(t) [INFO] [Main] : Analysis steps of x: ['fit']\n(t) fit\n")
    key = variants.create(config, RUN, "gaussian-circle", "gaussian", "circle", "on")
    path = catalog.resolve(config, key)
    assert key == f"{RUN}_gaussian-circle" and (path / "data").is_symlink()
    assert (path / "data").resolve() == (rundir / "data").resolve()
    assert sorted(p.name for p in (path / "results").iterdir()) == ["equalizer_raw_image.npy"]
    settings = catalog.open_rundir(path).loadConfig()
    assert settings["fit"]["model"] == "gaussian@circle" and settings["fit"]["ghost_mask"] is True
    assert settings["fit"]["fnum"] == 4                                         # the rest of the config kept
    assert "Analysis steps" not in (path / "log.log").read_text()
    assert json.loads((path / "meta" / "variant.json").read_text())["parent"] == RUN
    family = variants.family(config, key)
    assert [m["suffix"] for m in family] == [None, "gaussian-circle"]
    assert [m["model"] for m in family] == ["", "gaussian@circle"]
    assert variants.parent(config, key) == RUN
    with pytest.raises(FileExistsError):
        variants.create(config, key, "gaussian-circle", "gaussian", "circle", "on")    # made from the original, exists
    with pytest.raises(ValueError):
        variants.create(config, RUN, "../x", "gaussian", "circle", "on")
    with pytest.raises(ValueError):
        variants.create(config, RUN, "x", "gaussian", "square", "on")
