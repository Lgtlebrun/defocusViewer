"""A readable summary of a run for the Info tab: what was acquired (test, routines, stage positions, timing, camera), how
it is fitted, and what the analyses found. Read from meta/config.yaml, the dataframe and results/; a run without a config
is summarised from its data alone."""
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from DefocusTest.analysis.SpotLabeler import drop_overwritten_images, load_labeled_dataframe, load_labeled_result_df
from DefocusTest.imgDF import STACK_SEP
from DefocusTest.psf import split_model

from . import catalog

LOG_TIME = r"^\((\d{4}-\d\d-\d\d \d\d:\d\d):\d\d\)"
SEQUENCE_LINE = re.compile(LOG_TIME + r" \[\w+\] \[(\w+)\] : Results will be saved under")
STEPS_LINE = re.compile(LOG_TIME + r".*\[Main\] : Analysis steps of \S+: \[(.*)\]")
SLURM_LINE = re.compile(LOG_TIME + r".*\[Main\] : Submitted Slurm job (\d+) \((\d+) cores, (\w+)\)")
JOINT_LINE = re.compile(r"Joint hex fit: model=([^,]+), n_rounds=\d+, device=([\w-]+), (n_workers|gpus)=([^,]+?)(?: on ([^,]+))?, k=")
DONE_LINE = re.compile(LOG_TIME + r".*\[\w+_analysis\] : \w+_analysis completed in (.+)$")
KEY_RESULTS = {"joint fit": "theta_shared.npy", "ghost masks": "ghost_masks.hdf5", "series": "*_timeseries_zernike.npz",
               "calibration": "calibration_summary.hdf5", "viewer joint fit": "viewer_joint_fit.json"}
FLOW_NAMES = {"stability": "stability (long hold at one stage position)", "detection": "detection (hold, small stage steps)",
              "positioner": "positioner motion",
              "tracking": "tracking (static pre-tracking hold, then sidereal mount tracking)"}


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours} h {rest // 60:02d} min" if hours else f"{rest // 60} min {rest % 60:02d} s"


def when(times) -> str:
    """'start to end (duration)' of timestamps, or the one time when they are within a second."""
    first, last = times.min(), times.max()
    if (last - first).total_seconds() < 1:
        return f"{first:%Y-%m-%d %H:%M}"
    return f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} ({duration((last - first).total_seconds())})"


def span(values, unit: str = "", digits: int = 0) -> str:
    """'a' when all equal, else 'a to b' (with unit)."""
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    text = f"{lo:.{digits}f}" if lo == hi else f"{lo:.{digits}f} to {hi:.{digits}f}"
    return f"{text} {unit}".strip()


def _get(config: dict, *keys, default=None):
    for key in keys:
        if not isinstance(config, dict) or key not in config:
            return default
        config = config[key]
    return config


def _camera(config: dict, section: str) -> dict:
    """Camera settings of a section over the defaults of `all` (as config_utils.create_sub_config)."""
    return {**(_get(config, "all", "camera") or {}), **(_get(config, section, "camera") or {})}


def run_summary(path: Path, jobs: list[dict] | None = None) -> list[dict]:
    """[{title, rows: [[label, value]]}] of the run at `path`; jobs: the viewer's jobs on it (registry records)."""
    rundir = catalog.open_rundir(path)
    config = rundir.loadConfig() if (path / "meta" / "config.yaml").exists() else {}
    try:
        df = drop_overwritten_images(load_labeled_dataframe(rundir))
        df = df.assign(timestamp=pd.to_datetime(df["timestamp"], errors="coerce"))
    except (FileNotFoundError, OSError, KeyError):
        df = pd.DataFrame(columns=["fiber", "routine", "timestamp"])
    flow = catalog.flow(path)
    sections = [_run(path, config, df, flow), _routines(df), _stage(config, df, flow), _camera_section(path, config, df),
                _fit_settings(path, config), _results(path, rundir), _analyses(path, jobs or []), _temperature(path)]
    return [s for s in sections if s["rows"]]


def _run(path, config, df, flow) -> dict:
    times = df["timestamp"].dropna()
    hold = "none: " + ", ".join(sorted(df["routine"].astype(str).unique())) if flow is None else FLOW_NAMES.get(flow, flow)
    rows = [
        ["device / test", f"{_get(config, 'run', 'device') or path.parent.parent.name} / {_get(config, 'run', 'test') or path.parent.name}"],
        ["type of test", hold],
        ["name", _get(config, "output", "path") or path.name.split("_", 1)[-1]],
        ["acquisition sequence", log_history(path / "log.log", _mtime(path / "log.log"))["sequence"]],
        ["acquired", when(times) if len(times) else ""],
        ["acquisition finished", "yes" if (path / "meta" / "_finished").exists() else "no (meta/_finished missing)"],
        ["fibers", f"{df['fiber'].nunique()}: " + ", ".join(sorted(df["fiber"].astype(str).unique())) if len(df) else ""],
        ["spots", f"{len(df)} stacked images"],
        ["data folder", "symlink to " + str((path / "data").resolve()) if (path / "data").is_symlink() else str(path / "data")],
    ]
    return {"title": "Run", "rows": [r for r in rows if r[1]]}


def _routines(df) -> dict:
    rows = []
    for routine, sub in df.groupby("routine", sort=False):
        per_fiber = sub.groupby("fiber").size()
        positions = sub["x_position"].dropna().unique() if "x_position" in sub else []
        times = sub["timestamp"].dropna()
        parts = [f"{len(sub)} spots ({span(per_fiber.tolist())} per fiber)"]
        if len(positions):
            parts.append(f"{len(positions)} stage position{'s' if len(positions) > 1 else ''} ({span(positions, 'um')})")
        if len(times):
            parts.append(when(times)[11:] if when(times).count(" to ") == 0 else
                         f"{times.min():%H:%M} to {times.max():%H:%M} ({duration((times.max() - times.min()).total_seconds())})")
        rows.append([str(routine), ", ".join(parts)])
    return {"title": "Routines", "rows": rows}


def _stage(config, df, flow) -> dict:
    rows = []
    if _get(config, "all", "stage", "start_coord") is not None:
        rows.append(["start position", f"{_get(config, 'all', 'stage', 'start_coord')} um"])
    train = _get(config, "train", "stage") or {}
    if train.get("limits"):
        x = train.get("x") or []
        rows.append(["train sweep", f"{span(train['limits'], 'um')}, pitch {train.get('pitch')} um"
                                    + (f", {len(x)} positions" if x else "")])
    thin = _get(config, "train2", "stage") or {}
    if thin.get("limits"):
        rows.append(["dense sweep (train2)", f"{span(thin['limits'], 'um')}, pitch {thin.get('pitch')} um"])
    if flow and flow in df["routine"].values:
        hold = df[df["routine"] == flow]
        held = _get(config, flow, "stage", "hold_position")
        positions = hold["x_position"].dropna().unique() if "x_position" in hold else []
        rows.append(["hold position", f"{held} um" if held is not None else span(positions, "um")])
        planned = _get(config, flow, "duration_s")
        times = hold["timestamp"].dropna()
        actual = duration((times.max() - times.min()).total_seconds()) if len(times) else "?"
        rows.append(["hold duration", f"{actual}" + (f" (planned {duration(planned)})" if planned else "")])
    for section in ("train", "train_end", flow):
        burn_in = _camera(config, section).get("burn_in_s") if section else None
        if burn_in:
            rows.append([f"burn-in before {section}", duration(burn_in)])
    if _get(config, "train_end", "run"):
        rows.append(["sweep repeated after the hold", "yes (train_end)"])
    return {"title": "Stage and timing", "rows": rows}


def _camera_section(path, config, df) -> dict:
    camera = _camera(config, "train")
    rows = []
    frames = df["n_frames"].dropna().unique() if "n_frames" in df else []
    if len(frames) or camera.get("n_pics"):
        rows.append(["frames averaged per spot", span(frames) if len(frames) else str(camera["n_pics"])])
    if "texp" in df and df["texp"].notna().any():
        by_fiber = df.groupby("fiber")["texp"].median()
        rows.append(["exposure (equalized)", span(by_fiber.tolist(), "us") + (" across fibers" if by_fiber.nunique() > 1 else "")
                                             + (f" (initial {camera['texp']} us)" if camera.get("texp") else "")])
    if _get(config, "equalizer", "target_intensity"):
        rows.append(["equalizer target", f"{_get(config, 'equalizer', 'target_intensity')} counts"])
    if len(df):
        image = np.load(path / "data" / df["img"].iloc[0].split(STACK_SEP)[0], mmap_mode="r")
        rows.append(["stored crop", f"{image.shape[0]} px"])
    if camera.get("work_crop_side"):
        rows.append(["working crop", f"{camera['work_crop_side']} px"])
    raw = any(_camera(config, s).get("save_raw_frames") for s in ("train", "stability", "detection", "tracking"))
    if raw:
        rows.append(["raw frames saved", "yes (data/raw_*.npz)"])
    return {"title": "Camera", "rows": rows}


def _fit_settings(path, config) -> dict:
    fit_config = _get(config, "fit") or {}
    results = path / "results"
    one = lambda name: np.atleast_1d(np.load(results / f"{name}.npy", allow_pickle=True))   # noqa: E731
    model = str(one("fit_model")[0]) if (results / "fit_model.npy").exists() else fit_config.get("model")
    source, shape = split_model(model) if model else (None, None)
    noll = one("zernike_noll").tolist() if (results / "zernike_noll.npy").exists() else fit_config.get("zernike_noll")
    fnum = float(one("fnum")[0]) if (results / "fnum.npy").exists() else fit_config.get("fnum")
    ghost = ("on (masks in results)" if (results / "ghost_masks.hdf5").exists()
             else "on in the config" if fit_config.get("ghost_mask") else "off")
    rows = [
        ["f-number", f"f/{fnum:g}" if fnum else ""],
        ["source model / aperture", f"{source} / {shape}" if source else ""],
        ["PSF engine", str(one("psf_engine")[0]) if (results / "psf_engine.npy").exists() else fit_config.get("psf_engine", "")],
        ["Zernike terms (Noll)", ", ".join(str(j) for j in noll) if noll else ""],
        ["fit crop", f"{int(one('fit_crop_side')[0])} px" if (results / "fit_crop_side.npy").exists()
         else (f"{fit_config['crop_side']} px" if fit_config.get("crop_side") else "")],
        ["source radius", "fixed per fiber in the hold fits (fix_r)" if fit_config.get("fix_r") else
         ("free" if "fix_r" in fit_config else "")],
        ["pupil scale", fit_config.get("pupil_mode", "")],
        ["wavelength search", span(fit_config["lbda_range_nm"], "nm") if fit_config.get("lbda_range_nm") else ""],
        ["joint fit rounds", str(fit_config.get("n_rounds", ""))],
        ["ghost mask", ghost],
    ]
    return {"title": "Fit settings", "rows": [r for r in rows if r[1]]}


def _results(path, rundir) -> dict:
    results = path / "results"
    rows = [["analysis steps done", ", ".join(catalog.steps_done(path, catalog.flow(path))) or "none"]]
    provenance = results / "viewer_joint_fit.json"
    if provenance.exists():
        made = json.loads(provenance.read_text())
        rows.append(["joint fit made by the viewer", f"{made.get('mode')} mode, reference {str(made.get('reference')).split('/')[-1]}, "
                                                   f"{made.get('made', '')}"])
    if (results / "theta_shared.npy").exists():
        model = str(np.atleast_1d(np.load(results / "fit_model.npy", allow_pickle=True))[0]) if (results / "fit_model.npy").exists() else ""
        rows.append(["pupil rotation theta", "not used (circular aperture)" if model.endswith("@circle")
                     else f"{float(np.atleast_1d(np.load(results / 'theta_shared.npy'))[0]):.2f} deg"])
    try:
        scales = load_labeled_result_df(rundir, "pupil_scales")
        rows.append(["wavelength per fiber", ", ".join(f"{r.fiber} {r.lbda_nm:.1f}" for r in scales.itertuples()) + " nm"])
    except (FileNotFoundError, OSError, KeyError):
        pass
    try:
        records = load_labeled_result_df(rundir, "joint_fit_diag_records")
        rows.append(["joint fit residual_frac", f"median {records['residual_frac'].median():.3f} "
                                              f"({span(records['residual_frac'].tolist(), digits=3)})"])
        rows.append(["joint fit c4", f"{span(records['c4'].tolist(), 'waves', 2)}"])
    except (FileNotFoundError, OSError, KeyError):
        pass
    try:
        masks = load_labeled_result_df(rundir, "ghost_masks")
        found = masks[masks["masked"]]
        size = lambda r: f"{int(getattr(r, 'box_w', r.side))} x {int(getattr(r, 'box_h', r.side))} px"
        rows.append(["ghost boxes", f"{len(found)} of {len(masks)} fibers"
                     + "".join(f", {r.fiber} at ({r.gx:.0f}, {r.gy:.0f}) {size(r)}" for r in found.itertuples())])
    except (FileNotFoundError, OSError, KeyError):
        pass
    try:
        calibration = load_labeled_result_df(rundir, "calibration_summary")
        for variant, sub in calibration.groupby("variant", sort=False):
            rows.append([f"delta_z scatter, {variant}", f"std {span(sub['std'].tolist(), 'um', 2)} across fibers"])
    except (FileNotFoundError, OSError, KeyError):
        pass
    return {"title": "Fit results", "rows": rows}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=32)
def log_history(log: Path, mtime: float) -> dict:
    """{sequence, analyses} read from log.log: the acquisition sequence, and every analysis run of the cluster entry
    points (start, steps, joint-fit device, ghost masks, end and duration) or Slurm submission."""
    sequence, analyses, current = "", [], None
    for line in (log.read_text(errors="replace").splitlines() if log.exists() else []):
        if not sequence and (m := SEQUENCE_LINE.search(line)):
            sequence = f"{m[2]} ({m[1]})"
        elif m := STEPS_LINE.search(line):
            current = {"started": m[1], "steps": [s.strip(" '") for s in m[2].split(",") if s.strip()], "fits": [], "ghost": False}
            analyses.append(current)
        elif m := SLURM_LINE.search(line):
            analyses.append({"started": m[1], "slurm": f"Slurm job {m[2]}, {m[3]} cores on {m[4]}"})
        elif current is not None:
            if m := JOINT_LINE.search(line):
                where = (f"GPU ({m[4]}{' on ' + m[5] if m[5] else ''})" if m[3] == "gpus" else f"CPU ({m[4]} workers)")
                current["fits"].append(f"joint fit on {where}")
            elif "Ghost masks applied" in line:
                current["ghost"] = True
            elif m := DONE_LINE.search(line):
                current.update(finished=m[1], took=m[2])
                current = None
    return {"sequence": sequence, "analyses": analyses}


def _analyses(path, jobs) -> dict:
    """The analysis history of the run: log.log, the viewer's jobs, and when the key results were written."""
    rows = []
    for entry in log_history(path / "log.log", _mtime(path / "log.log"))["analyses"]:
        if "slurm" in entry:
            rows.append([entry["started"], f"submitted to {entry['slurm']}"])
            continue
        parts = [", ".join(entry["steps"])] + entry["fits"][-1:] + (["ghost masks applied"] if entry["ghost"] else [])
        parts.append(f"took {entry['took']}" if "took" in entry else "no completion line (failed, killed or running)")
        rows.append([entry["started"], " · ".join(parts)])
    for job in sorted(jobs, key=lambda j: j["started"]):
        exit_code = f" (exit {job['exit_code']})" if job.get("exit_code") is not None else ""
        rows.append([f"viewer job {job['started'][:16].replace('T', ' ')}", f"{job['label']} · {job['mode']} · {job['state']}{exit_code}"])
    results = path / "results"
    for label, pattern in KEY_RESULTS.items():
        files = list(results.glob(pattern))
        if files:
            written = datetime.fromtimestamp(max(f.stat().st_mtime for f in files))
            rows.append([f"{label} written", f"{written:%Y-%m-%d %H:%M}"])
    return {"title": "Analyses", "rows": rows}


def _temperature(path) -> dict:
    file = path / "data" / "temperature.hdf5"
    if not file.exists():
        return {"title": "Temperature", "rows": []}
    readings = pd.read_hdf(file, key="data")
    names = {"T1": "T1", "T2": "T2", "T3": "T3", "env": "environment", "hum": "humidity"}
    rows = []
    for column, label in names.items():
        if column in readings and readings[column].notna().any():
            values = readings[column].dropna()
            unit = "%" if column == "hum" else "C"
            rows.append([label, f"mean {values.mean():.2f} {unit}, range {values.min():.2f} to {values.max():.2f} "
                                f"({len(values)} readings)"])
    return {"title": "Temperature", "rows": rows}
