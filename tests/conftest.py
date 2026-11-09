from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defocusviewer.config import Config

REAL_RUN = Path("/disk/users/llebru/Defocus/Orbray6/Defocus/20261017-175317_stability-fnum4-2")


def make_rundir(root: Path, name: str = "20261017-175317_stability-test", device: str = "Orbray6") -> Path:
    """A small rundir on disk: 2 fibers, 2 train and 3 stability spots each, config, no results."""
    path = root / device / "Defocus" / name
    for sub in ("meta", "data", "plot", "results"):
        (path / sub).mkdir(parents=True)
    (path / "meta" / "config.yaml").write_text("run: {device: Orbray6, test: Defocus}\nfit: {fnum: 4, fix_r: true}\n")
    (path / "meta" / "_finished").touch()
    rows = []
    t0 = pd.Timestamp("2026-10-17 18:00")
    for fiber in ("Fa", "Fb"):
        for i, x in enumerate((25000, 25500)):
            rows.append(dict(fiber=fiber, routine="train", x_position=x, timestamp=t0 + pd.Timedelta(minutes=i),
                             img=f"{fiber.lower()}_{x}_train.npy"))
        for i in range(3):
            rows.append(dict(fiber=fiber, routine="stability", x_position=35000, timestamp=t0 + pd.Timedelta(minutes=10 + i),
                             img=f"{fiber.lower()}_35000_stability_{i}.npy"))
    df = pd.DataFrame(rows).assign(n_frames=300, texp=1000.0, mom_center_x=100.0, mom_center_y=200.0)
    df.to_hdf(path / "data" / "dataframe.hdf5", key="data", format="table")
    for img in df["img"]:
        np.save(path / "data" / img, np.random.default_rng(0).random((21, 21)))
    (path / "plot" / "healthchecks").mkdir()
    (path / "plot" / "spots_centers.pdf").write_bytes(b"%PDF-1.4\n")
    return path


@pytest.fixture
def config(tmp_path):
    return Config(data_root=tmp_path / "data", raw_root=tmp_path / "raw", extra_roots={}, state_dir=tmp_path / "state",
                  cache_dir=tmp_path / "cache", python="python")


@pytest.fixture
def rundir(config):
    return make_rundir(config.data_root)
