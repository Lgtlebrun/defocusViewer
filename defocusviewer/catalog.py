"""The rundirs the viewer knows: local copies under the data roots, runs still only on pos-control, and what each has.

A run is named by a key, "<device>/<test>/<rundir>" under the data root and "<label>/<rundir>" under an extra root.
"""
import re
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from posUtils.data.dir import OutputFormat, RemoteCopyDataDir, RunDir, map_device

from .config import Config

RUNDIR_NAME = re.compile(r"^\d{8}-\d{6}_[\w.+-]+$")
FLOWS = ("stability", "detection", "positioner")
REMOTE_REFRESH_S = 600


def open_rundir(path: Path) -> RunDir:
    """RunDir of an existing folder, read only (RunDir.__init__ would create the subfolders it misses)."""
    rundir = RunDir.__new__(RunDir)
    rundir.entry, rundir.folder, rundir.datadir = path.name, path, None
    rundir.meta, rundir.data, rundir.plot, rundir.results = (path / d for d in ("meta", "data", "plot", "results"))
    rundir.type = rundir._output = OutputFormat.HDF5
    rundir.df_path, rundir.key, rundir.mode = rundir.data / "dataframe.hdf5", "data", "r"
    return rundir


def _is_rundir(path: Path) -> bool:
    return path.is_dir() and RUNDIR_NAME.match(path.name) is not None and (path / "data").exists()


def local_runs(config: Config) -> dict[str, Path]:
    """{key: folder} of every local rundir."""
    runs = {}
    root = config.data_root
    for path in sorted(root.glob("*/*/*")) if root.exists() else []:
        if _is_rundir(path):
            runs[f"{path.parent.parent.name}/{path.parent.name}/{path.name}"] = path
    for label, extra in config.extra_roots.items():
        for path in sorted(extra.glob("*")) if extra.exists() else []:
            if _is_rundir(path):
                runs[f"{label}/{path.name}"] = path
    return runs


def resolve(config: Config, key: str) -> Path:
    """Folder of the local run `key`; KeyError for an unknown key (never a path built from user input)."""
    return local_runs(config)[key]


def device_of(config: Config, key: str) -> str | None:
    """Device folder of a run under the data root (None under an extra root: the analyses cannot fetch those)."""
    parts = key.split("/")
    return parts[0] if len(parts) == 3 and parts[1] == config.test else None


def device_id(folder: str) -> str:
    """Identifier the entry points take for a device folder (`--device`), e.g. "calib" for calibration_stand."""
    return {name: module.value for module, name in map_device.items()}[folder]


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@lru_cache(maxsize=256)
def _routines(df_path: Path, mtime: float) -> tuple[str, ...]:
    try:
        return tuple(sorted(pd.read_hdf(df_path, key="data", columns=["routine"])["routine"].astype(str).unique()))
    except (FileNotFoundError, OSError, KeyError, ValueError, TypeError):
        try:
            return tuple(sorted(pd.read_hdf(df_path, key="data")["routine"].astype(str).unique()))
        except (FileNotFoundError, OSError, KeyError, ValueError):
            return ()


def routines(path: Path) -> tuple[str, ...]:
    """Routines of the spots of a rundir."""
    df_path = path / "data" / "dataframe.hdf5"
    return _routines(df_path, _mtime(df_path))


def flow(path: Path) -> str | None:
    """"stability", "detection" or "positioner": the hold routine of the rundir, None without one."""
    found = routines(path)
    return next((f for f in FLOWS if f in found), None)


def steps_done(path: Path, flow_name: str | None) -> list[str]:
    """Analysis steps whose results are in the rundir (see DefocusTest.analysis.remote.STEPS)."""
    results, plot = path / "results", path / "plot"
    checks = {
        "fit": (results / "joint_fit_diag_records.hdf5").exists(),
        "ghost_mask": (results / "ghost_masks.hdf5").exists(),
        "refit": (results / "joint_fit_diag_records_unmasked.hdf5").exists(),
        "joint_checks": (plot / "healthchecks" / "joint_fit").exists(),
        "series": bool(flow_name) and any(results.glob(f"*_{flow_name}_timeseries_zernike.npz")),
        "model_fits": (plot / "healthchecks" / "model_fit").exists(),
        "overlay": (plot / "healthchecks" / "train_curves_overlay.pdf").exists(),
        "relative": (plot / "relative_stability_zernike").exists(),
        "calibration": (results / "calibration_summary.hdf5").exists(),
        "temperature": (plot / "temperature").exists(),
    }
    return [step for step, done in checks.items() if done]


def summary(path: Path) -> dict:
    """What the run list shows of a local rundir."""
    flow_name = flow(path)
    return {"finished": (path / "meta" / "_finished").exists(), "flow": flow_name,
            "has_config": (path / "meta" / "config.yaml").exists(), "steps_done": steps_done(path, flow_name),
            "joint_fit": (path / "results" / "theta_shared.npy").exists(), "data_link": (path / "data").is_symlink()}


class RemoteRuns:
    """The rundirs on pos-control (Defocus test of each device), listed over SSH in the background every 10 min."""

    def __init__(self, config: Config):
        self.config, self.runs, self.updated, self.error = config, {}, None, None
        self._lock, self._wake = threading.Lock(), threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="remote-runs").start()

    def refresh(self):
        self._wake.set()

    def _loop(self):
        while True:
            self._wake.clear()
            try:
                runs = self.list_remote()
                with self._lock:
                    self.runs, self.updated, self.error = runs, time.time(), None
            except (OSError, subprocess.SubprocessError, RuntimeError) as e:
                with self._lock:
                    self.error = str(e)[:300]
            self._wake.wait(REMOTE_REFRESH_S)

    def list_remote(self) -> dict[str, dict]:
        """{key: {device, finished}} of every rundir of the remote devices; one SSH call."""
        base, test = RemoteCopyDataDir.REMOTE_BASE_DATA, self.config.test
        target = f"{RemoteCopyDataDir.USER}@{RemoteCopyDataDir.HOST}"
        folders = " ".join(f"{d}/{test}" for d in self.config.remote_devices)
        cmd = (f"cd {base} && for d in {folders}; do [ -d $d ] && ls -1 $d | sed \"s|^|$d/|\"; done; "
               f"echo __finished__; for d in {folders}; do ls -1d $d/*/meta/_finished 2>/dev/null; done; true")
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, cmd],
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError(f"ssh {target}: {out.stderr.strip() or out.returncode}")
        listed, _, finished = out.stdout.partition("__finished__")
        done = {line.removesuffix("/meta/_finished") for line in finished.split()}
        runs = {}
        for line in listed.split():
            device, _, name = line.split("/", 2)
            if RUNDIR_NAME.match(name):
                runs[line] = {"device": device, "finished": line in done}
        return runs

    def snapshot(self) -> dict:
        with self._lock:
            return {"runs": dict(self.runs), "updated": self.updated, "error": self.error}


def run_list(config: Config, remote: RemoteRuns | None) -> dict:
    """Local runs with their summary, and the runs only on pos-control, grouped by device / test or extra root."""
    local = local_runs(config)
    rows = [{"key": key, "group": key.rsplit("/", 1)[0], "name": path.name, "local": True, **summary(path)}
            for key, path in local.items()]
    remote_state = remote.snapshot() if remote else {"runs": {}, "updated": None, "error": None}
    for key, info in remote_state["runs"].items():
        if key not in local:
            rows.append({"key": key, "group": key.rsplit("/", 1)[0], "name": key.rsplit("/", 1)[1], "local": False,
                         "finished": info["finished"], "flow": None, "steps_done": [], "joint_fit": False})
    rows.sort(key=lambda r: (r["group"], r["name"]), reverse=False)
    return {"runs": rows, "remote_updated": remote_state["updated"], "remote_error": remote_state["error"]}
