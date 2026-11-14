"""The rundirs the viewer knows: local copies under the data roots, runs still only on pos-control, and what each has.

A run is named by a key, "<device>/<test>/<rundir>" under the data root and "<label>/<rundir>" under an extra root.
"""
import re
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from posUtils.data.dir import OutputFormat, RemoteCopyDataDir, RunDir, map_device

from .config import Config

RUNDIR_NAME = re.compile(r"^\d{8}-\d{6}_[\w.+-]+$")
# tracking: TrackingTest (pre_tracking + tracking rows), pointing / luminosity: PointingTest / LuminosityTest (one hold
# routine per pointing / light level, pointing_<i> / luminosity_<i>, fitted as one series)
FLOWS = ("tracking", "pointing", "luminosity", "stability", "detection", "positioner")
INDEXED_HOLD = re.compile(r"^(pointing|luminosity)_(\d+)$")
ROUTINE_ORDER = ("train_burnin", "train", "train2", "stability", "detection", "positioner", "pre_tracking", "tracking",
                 "pointing", "luminosity", "train_end")
REMOTE_REFRESH_S = 600
# f-number in a run name: fnum4, fnum2p8, apf1p8, f8 ("fnummin" has none)
FNUM_IN_NAME = (r"fnum[-_]?(\d+(?:p\d+)?)", r"apf(\d+(?:p\d+)?)", r"(?:^|[-_])f(\d+(?:p\d+)?)(?:[-_]|$)")


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


def series_of(routine: str) -> str:
    """Name of the hold series a routine's frames are fitted in: pre_tracking -> tracking, pointing_<i> -> pointing,
    luminosity_<i> -> luminosity, any other routine its own name."""
    if routine == "pre_tracking":
        return "tracking"
    indexed = INDEXED_HOLD.match(routine)
    return indexed.group(1) if indexed else routine


def routine_sort_key(routine: str) -> tuple:
    """Routines in acquisition order (ROUTINE_ORDER, the indexed holds by index), unknown ones last by name."""
    indexed = INDEXED_HOLD.match(routine)
    base, i = (indexed.group(1), int(indexed.group(2))) if indexed else (routine, 0)
    return (ROUTINE_ORDER.index(base) if base in ROUTINE_ORDER else len(ROUTINE_ORDER), base, i)


def flow(path: Path) -> str | None:
    """"tracking", "pointing", "luminosity", "stability", "detection" or "positioner": the hold routine of the rundir
    (pointing / luminosity: their first routine, pointing_0 / luminosity_0), None without one."""
    found = routines(path)
    return next((f for f in FLOWS if f in found or f"{f}_0" in found), None)


@lru_cache(maxsize=1024)
def _config_fnum(path: Path, mtime: float) -> float | None:
    try:
        fnum = (open_rundir(path).loadConfig().get("fit") or {}).get("fnum")
        return float(fnum) if fnum else None
    except Exception:      # a config posUtils cannot read (or a fnum that is no number): the next source decides
        return None


@lru_cache(maxsize=1024)
def _results_fnum(npy: Path, mtime: float) -> float | None:
    try:
        return float(np.atleast_1d(np.load(npy))[0])
    except (OSError, ValueError, IndexError):
        return None


def fnum_from_name(name: str) -> float | None:
    slug = name.split("_", 1)[-1].lower()
    for pattern in FNUM_IN_NAME:
        found = re.search(pattern, slug)
        if found:
            return float(found.group(1).replace("p", "."))
    return None


def guess_fnum(path: Path) -> tuple[float | None, str]:
    """(f-number, where it was read): config fit.fnum, results/fnum.npy (viewer joint fits), the run name (fnum4, apf1p8,
    f8), else None. Cached on the files' mtimes: the run list asks it of every run every 30 s."""
    config_mtime = _mtime(path / "meta" / "config.yaml")
    if config_mtime and (fnum := _config_fnum(path, config_mtime)):
        return fnum, "config"
    npy = path / "results" / "fnum.npy"
    if (npy_mtime := _mtime(npy)) and (fnum := _results_fnum(npy, npy_mtime)) is not None:
        return fnum, "results"
    fnum = fnum_from_name(path.name)
    return (fnum, "run name") if fnum is not None else (None, "unknown")


def name_tags(name: str, words) -> list[str]:
    """The tag words found in a run name (date stripped, any case)."""
    slug = name.split("_", 1)[-1].lower()
    return [w for w in words if w.lower() in slug]


def fnum_tag(fnum: float | None) -> str | None:
    return f"f/{fnum:g}" if fnum else None


def steps_done(path: Path, flow_name: str | None) -> list[str]:
    """Analysis steps whose results are in the rundir (see DefocusTest.analysis.remote.STEPS)."""
    results, plot = path / "results", path / "plot"
    checks = {
        "fit": (results / "joint_fit_diag_records.hdf5").exists(),
        "ghost_mask": (results / "ghost_masks.hdf5").exists(),
        "refit": (results / "joint_fit_diag_records_unmasked.hdf5").exists(),
        "joint_checks": (plot / "healthchecks" / "joint_fit").exists(),
        "series": bool(flow_name) and any(results.glob(f"*_{flow_name}_timeseries_zernike.npz")),
        "hold_overlay": (plot / "hold_overlay").exists(),
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
    fnum, fnum_source = guess_fnum(path)
    return {"finished": (path / "meta" / "_finished").exists(), "flow": flow_name, "fnum": fnum, "fnum_source": fnum_source,
            "has_config": (path / "meta" / "config.yaml").exists(), "steps_done": steps_done(path, flow_name),
            "joint_fit": (path / "results" / "theta_shared.npy").exists(), "data_link": (path / "data").is_symlink(),
            "viewer_fit": (path / "results" / "viewer_joint_fit.json").exists()}


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


def _tags(name: str, flow_name: str | None, fnum: float | None, words) -> list[str]:
    tags = name_tags(name, words)
    if flow_name in words and flow_name not in tags:
        tags.append(flow_name)
    return tags + ([fnum_tag(fnum)] if fnum else [])


def _mark_analysed(rows: list[dict]) -> None:
    """analysed: the run or one of its variants (local runs `<name>_<suffix>` of the same group) has a joint fit, so an
    original analysed through its `_tophatbl-circle` variant does not count as waiting for an analysis."""
    fitted = [r for r in rows if r["local"] and (r["joint_fit"] or "fit" in r["steps_done"])]
    for row in rows:
        row["analysed"] = row["local"] and any(
            f["group"] == row["group"] and (f["name"] == row["name"] or f["name"].startswith(f"{row['name']}_")) for f in fitted)


def run_list(config: Config, remote: RemoteRuns | None) -> dict:
    """Local runs with their summary, and the runs only on pos-control, grouped by device / test or extra root.

    Each run has tags: the words of config.run_tags in its name (or its hold routine), and "f/<fnum>" (local runs: the
    fit.fnum of meta/config.yaml first; runs on the rig: their name only)."""
    local = local_runs(config)
    rows = []
    for key, path in local.items():
        info = summary(path)
        rows.append({"key": key, "group": key.rsplit("/", 1)[0], "name": path.name, "local": True, **info,
                     "tags": _tags(path.name, info["flow"], info["fnum"], config.run_tags)})
    remote_state = remote.snapshot() if remote else {"runs": {}, "updated": None, "error": None}
    for key, info in remote_state["runs"].items():
        if key not in local:
            name = key.rsplit("/", 1)[1]
            fnum = fnum_from_name(name)
            rows.append({"key": key, "group": key.rsplit("/", 1)[0], "name": name, "local": False,
                         "finished": info["finished"], "flow": None, "steps_done": [], "joint_fit": False,
                         "fnum": fnum, "fnum_source": "run name" if fnum is not None else "unknown",
                         "tags": _tags(name, None, fnum, config.run_tags)})
    _mark_analysed(rows)
    rows.sort(key=lambda r: (r["group"], r["name"]), reverse=False)
    return {"runs": rows, "tag_words": list(config.run_tags), "remote_updated": remote_state["updated"],
            "remote_error": remote_state["error"]}
