"""Spots of a rundir (one stacked .npy per fiber and acquisition): their table, the saved fit of each, the model and
residual images, and the viewer's own fit of a spot whose saved fit is missing or used the other ghost setting.

ghost is "on" (weight 0 on the fiber's saved ghost box, as the masked pipeline fits) or "off".
"""
import base64
import hashlib
import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from DefocusTest import fit
from DefocusTest.analysis.SpotLabeler import drop_overwritten_images, load_labeled_dataframe, load_labeled_result_df
from DefocusTest.analysis.ZernikeStabilityAnalysis import _fit_candidate, load_crop_side, load_fnum, load_pupil_scale
from DefocusTest.devices import DESI
from DefocusTest.imgDF import load_stack
from DefocusTest.postprocessing import GhostMasks
from PIL import Image
from posUtils.define import UMperCOUNT

from .catalog import _mtime, open_rundir

HOLD_ROUTINES = {"stability": "ZernikeStabilityTimeSeries", "detection": "ZernikeDetectionTimeSeries",
                 "positioner": "ZernikePositionerTimeSeries"}
GHOST_LINE = re.compile(r"\[(?:\w+:)?(\w+)\] : (\S+): ghost mask applied")
STEPS_LINE = re.compile(r"Analysis steps of \S+: \[(.*)\]")
COLUMNS = ["img", "fiber", "routine", "x_position", "timestamp", "texp", "n_frames", "mom_center_x", "mom_center_y"]


def _one(rundir, name):
    return np.atleast_1d(rundir.loadResult(name))[0]


@lru_cache(maxsize=64)
def _ghost_applied(log_path: Path, mtime: float) -> frozenset:
    """{(analysis class, fiber)} that applied the ghost mask in the last run of the series step (log.log)."""
    if not log_path.exists():
        return frozenset()
    lines = log_path.read_text(errors="replace").splitlines()
    start = max((i for i, line in enumerate(lines) if (m := STEPS_LINE.search(line)) and "'series'" in m.group(1)),
                default=0)
    return frozenset(m.groups() for line in lines[start:] if (m := GHOST_LINE.search(line)))


class RunState:
    """What the spot views of one rundir need, read once per change of its dataframe and results."""

    def __init__(self, path: Path):
        self.path, self.rundir = path, open_rundir(path)
        df = drop_overwritten_images(load_labeled_dataframe(self.rundir).reset_index(drop=True))
        df = df[df["img"].astype(str) != ""].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        self.df = df.set_index("img", drop=False)
        self.config = self.rundir.loadConfig() if (path / "meta" / "config.yaml").exists() else {}
        fit_config = self.config.get("fit") or {}
        self.supersample = int(fit_config.get("supersample", 8))
        self.psf_oversample = int(fit_config.get("psf_oversample", fit.DEFAULT_PSF_OVERSAMPLE))
        self.fix_r = bool(fit_config.get("fix_r", True))
        self.crop_side = load_crop_side(self.rundir)
        self.joint = (path / "results" / "theta_shared.npy").exists()
        if self.joint:
            self.theta = float(_one(self.rundir, "theta_shared"))
            self.model, self.engine = str(_one(self.rundir, "fit_model")), str(_one(self.rundir, "psf_engine"))
            self.fnum = load_fnum(self.rundir)
            self.noll = tuple(int(j) for j in np.atleast_1d(self.rundir.loadResult("zernike_noll")))
        self.masks = GhostMasks.load(self.rundir)
        self.fits = self._saved_fits()

    def image(self, img: str) -> np.ndarray:
        crop = None if self.df.loc[img, "fiber"] == DESI else self.crop_side
        return load_stack(self.path / "data" / img, crop)

    def _ghost_of(self, analysis: str, fiber: str) -> str:
        log = self.path / "log.log"
        return "on" if (analysis, fiber) in _ghost_applied(log, _mtime(log)) else "off"

    def _saved_fits(self) -> dict[str, list[dict]]:
        """{img: [{source, ghost, params [r, x0, y0, c4, *cJ], noll, residual_frac, ...}]} of the pipeline's fits."""
        fits = {}
        if not self.joint:
            return fits
        results = self.path / "results"
        refitted = (results / "joint_fit_diag_records_unmasked.hdf5").exists()
        sweeps = [("joint_fit_diag_records", "on" if refitted else "off")]
        sweeps += [("joint_fit_diag_records_unmasked", "off")] if refitted else []
        for name, ghost in sweeps:
            try:
                records = load_labeled_result_df(self.rundir, name)
            except (FileNotFoundError, OSError, KeyError):
                continue
            noll = self.noll
            for _, row in records.iterrows():
                fits.setdefault(str(row["img"]), []).append({
                    "source": "joint fit" if name == "joint_fit_diag_records" else "joint fit (unmasked)",
                    "ghost": ghost, "ghost_inferred": False, "noll": list(noll),
                    "params": [float(row[c]) for c in ("r", "x0", "y0", "c4")] + [float(row.get(f"c{j}", 0.0)) for j in noll[1:]],
                    "residual_frac": float(row.get("residual_frac", np.nan)),
                    **({"note": "rendered with theta and pupil scale of the masked refit (the unmasked fit's are not "
                                "saved)"} if name.endswith("_unmasked") else {})})
        for routine, analysis in HOLD_ROUTINES.items():
            for series_path in results.glob(f"*_{routine}_timeseries_zernike.npz"):
                fiber = series_path.name.removesuffix(f"_{routine}_timeseries_zernike.npz")
                self._hold_fits(fits, fiber, routine, analysis, dict(np.load(series_path, allow_pickle=True)))
        for train_end in results.glob("*_train_end_fit.hdf5"):
            fiber = train_end.name.removesuffix("_train_end_fit.hdf5")
            self._train_end_fits(fits, fiber, pd.read_hdf(train_end, key="data"))
        return fits

    def _hold_fits(self, fits, fiber, routine, analysis, series):
        """Hold frames of a fiber, in time order, against the rows of its series (as ModelFitPlots._hold)."""
        frames = self.df[(self.df["fiber"] == fiber) & (self.df["routine"] == routine)].sort_values("timestamp")
        if len(frames) != len(series["c4"]):
            return
        noll = [int(j) for j in series["zernike_noll"]]
        ghost = self._ghost_of(analysis, fiber)
        for i, img in enumerate(frames["img"]):
            extra = {k: float(series[k][i]) for k in ("delta_z", "sigma_delta_z", "sigma_c4") if k in series}
            fits.setdefault(img, []).append({
                "source": f"{routine} series", "ghost": ghost, "ghost_inferred": True, "noll": noll,
                "params": [float(series[k][i]) for k in ("r", "x0", "y0", "c4")] + [float(series[f"c{j}"][i]) for j in noll[1:]],
                "residual_frac": float(series["residual_frac"][i]), **extra})

    def _train_end_fits(self, fits, fiber, table):
        frames = self.df[(self.df["fiber"] == fiber) & (self.df["routine"] == "train_end")]
        by_x = {int(x): img for x, img in zip(frames["x_position"], frames["img"])}
        noll = [4] + sorted(int(c[1:]) for c in table.columns if re.fullmatch(r"c\d+", c) and c != "c4")
        ghost = self._ghost_of("TrainEndFit", fiber)
        for _, row in table.iterrows():
            img = by_x.get(int(row["x_position"]))
            if img is not None:
                fits.setdefault(img, []).append({
                    "source": "train_end fit", "ghost": ghost, "ghost_inferred": True, "noll": noll,
                    "params": [float(row[c]) for c in ("r", "x0", "y0", "c4")] + [float(row[f"c{j}"]) for j in noll[1:]],
                    "residual_frac": float(row.get("residual_frac", np.nan)), "sigma_c4": float(row.get("sigma_c4", np.nan))})

    def weights(self, fiber: str, shape, ghost: str):
        return self.masks.weights(fiber, shape) if ghost == "on" and self.masks is not None else None

    def ghost_box(self, fiber: str, shape) -> dict | None:
        if self.masks is None:
            return None
        box = self.masks.box(fiber, shape)
        if box is None:
            return None
        row = self.masks.table.loc[fiber]
        return {"x0": box[0], "y0": box[1], "side": box[2], "gx": float(row["gx"]), "gy": float(row["gy"])}

    def stamp(self) -> dict:
        """Mtimes a viewer fit depends on: it is stale when one changes."""
        results = self.path / "results"
        return {name: _mtime(results / name) for name in ("theta_shared.npy", "pupil_scales.hdf5", "ghost_masks.hdf5")}


def _signature(path: Path) -> tuple:
    results = path / "results"
    files = sorted((p.name, p.stat().st_mtime) for p in results.iterdir() if p.is_file()) if results.exists() else []
    return (_mtime(path / "data" / "dataframe.hdf5"), _mtime(path / "log.log"), tuple(files))


_states: dict[Path, tuple[tuple, RunState]] = {}
_states_lock = threading.Lock()


def state(path: Path) -> RunState:
    """RunState of a rundir, rebuilt when its dataframe, log or results change."""
    signature = _signature(path)
    with _states_lock:
        cached = _states.get(path)
    if cached and cached[0] == signature:
        return cached[1]
    built = RunState(path)
    with _states_lock:
        _states[path] = (signature, built)
    return built


def table(path: Path, viewer_fits: Path | None = None) -> dict:
    """{columns, rows, fits} of the spots: one row per spot, and the fit sources of each spot."""
    st = state(path)
    df = st.df[[c for c in COLUMNS if c in st.df.columns]].copy()
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    cached = {p.name.rsplit("_", 1)[0] for p in viewer_fits.glob("*.json")} if viewer_fits and viewer_fits.exists() else set()
    fit_sources = {img: sorted({f["source"] for f in st.fits.get(img, [])}) + (["viewer"] if img in cached else [])
                   for img in df["img"]}
    rows = df.astype(object).where(df.notna(), None).values.tolist()
    return {"columns": list(df.columns), "rows": rows, "fits": fit_sources, "joint_fit": st.joint,
            "ghost_masks": st.masks is not None}


def _b64(array: np.ndarray) -> dict:
    a = np.ascontiguousarray(array, dtype=np.float32)
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def _fft_magnitude(image: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fftshift(np.fft.fft2(image)))


def viewer_fit_path(folder: Path, img: str, ghost: str) -> Path:
    return folder / f"{img}_{ghost}.json"


def load_viewer_fit(folder: Path | None, img: str, ghost: str, st: RunState) -> dict | None:
    """The cached viewer fit of a spot, None when missing or made with other joint-fit results or masks."""
    if folder is None:
        return None
    path = viewer_fit_path(folder, img, ghost)
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    return cached if cached.get("stamp") == st.stamp() else None


def detail(path: Path, img: str, ghost: str, viewer_fits: Path | None = None) -> dict:
    """Metadata, fits and images (data, model, residual = model - data, |FFT| of each) of one spot.

    `needs` says what is missing to show a model: "joint_fit" (the rundir has none) or "refit" (no saved or cached fit
    with this ghost setting)."""
    st = state(path)
    if img not in st.df.index:
        raise KeyError(img)
    row = st.df.loc[img]
    fiber = str(row["fiber"])
    image = st.image(img)
    ghost_box = st.ghost_box(fiber, image.shape) if row["fiber"] != DESI else None
    if ghost_box is None:
        ghost = "off"
    meta = {c: (None if pd.isna(row[c]) else row[c].isoformat() if c == "timestamp" else row[c].item()
                if hasattr(row[c], "item") else row[c]) for c in COLUMNS if c in row.index}
    meta["fnum"] = st.fnum if st.joint else (st.config.get("fit") or {}).get("fnum")
    meta["device"] = (st.config.get("run") or {}).get("device")
    meta["crop_side"] = int(image.shape[0])
    meta["stage_counts"] = None if meta.get("x_position") is None else round(meta["x_position"] / UMperCOUNT)
    saved = st.fits.get(img, [])
    chosen = next((f for f in saved if f["ghost"] == ghost), None)
    viewer = load_viewer_fit(viewer_fits, img, ghost, st)
    shown = chosen or (dict(viewer, source="viewer fit") if viewer else None)
    out = {"img": img, "fiber": fiber, "meta": meta, "saved": saved, "viewer_fit": viewer, "fit": shown,
           "ghost": ghost, "ghost_box": ghost_box, "ghost_masks": st.masks is not None, "joint_fit": st.joint,
           "needs": None if shown else "joint_fit" if not st.joint else "refit", "images": {"data": _b64(image),
           "fft_data": _b64(_fft_magnitude(image))}}
    if shown and st.joint and fiber != DESI:
        model = render(st, fiber, image, shown["params"], shown["noll"], ghost)
        residual = model - image
        out["images"].update(model=_b64(model), residual=_b64(residual), fft_model=_b64(_fft_magnitude(model)),
                             fft_residual=_b64(_fft_magnitude(residual)))
        out["residual_frac"] = residual_frac(residual, image, st.weights(fiber, image.shape, ghost),
                                             shown["source"].startswith("joint fit"))
        out["pupil_scale"] = load_pupil_scale(st.rundir, fiber)
        out["absolute_center"] = [meta["mom_center_x"] + shown["params"][1], meta["mom_center_y"] + shown["params"][2]]
    return out


def residual_frac(residual, image, weights, joint: bool) -> float:
    """max |residual| / max(image) over the pixels that count; the joint fit also takes the image peak outside the ghost
    box (fit.fit_joint), the per-frame fits over the whole image (fit.fit_summary)."""
    if weights is None:
        return float(np.abs(residual).max() / image.max())
    if joint:
        return float(np.abs(residual[weights > 0]).max() / image[weights > 0].max())
    return float(np.abs(residual * weights).max() / image.max())


def render(st: RunState, fiber: str, image: np.ndarray, params, noll, ghost: str) -> np.ndarray:
    """Model image of a spot from its parameters and the joint fit, amplitude and background solved as in the fit."""
    return fit.render_model_shared(
        list(params), image, st.theta, load_pupil_scale(st.rundir, fiber), model=st.model, supersample=st.supersample,
        psf_oversample=st.psf_oversample, engine=st.engine, fnum=st.fnum, zernike_noll=tuple(int(j) for j in noll),
        weights=st.weights(fiber, image.shape, ghost))


def refit(path: Path, img: str, ghost: str, folder: Path, map_function=map) -> dict:
    """Fits one spot with the joint fit's theta and pupil scale (r fixed at the series' value for hold frames when
    fit.fix_r), from its saved fits (else the nearest fit of its fiber) and a generic seed (spread over `map_function`, e.g. a process pool's map), and caches
    the result in `folder`."""
    st = state(path)
    row = st.df.loc[img]
    fiber, routine = str(row["fiber"]), str(row["routine"])
    image = st.image(img)
    series_path = path / "results" / f"{fiber}_{routine}_timeseries_zernike.npz"
    series = dict(np.load(series_path, allow_pickle=True)) if routine in HOLD_ROUTINES and series_path.exists() else None
    noll = tuple(int(j) for j in series["zernike_noll"]) if series is not None else st.noll
    r_fixed = float(series["r_fixed"][0]) if series is not None and st.fix_r and "r_fixed" in series else None
    seeds = [_seed_from(f["params"], noll) for f in st.fits.get(img, [])] or [_nearest_seed(st, fiber, row, noll)]
    seeds.append([1.0, 0.0, 0.0, seeds[0][3]] + [0.0] * (len(noll) - 1))     # generic seed, as the per-frame fits
    if r_fixed is not None:
        seeds = [s[1:] for s in seeds]
    weights = st.weights(fiber, image.shape, ghost)
    scale = load_pupil_scale(st.rundir, fiber)
    unique = [list(s) for s in dict.fromkeys(tuple(s) for s in seeds)]
    tasks = [(image, st.theta, scale, r_fixed, None, p0, st.model, st.supersample, st.psf_oversample, st.engine, st.fnum,
              noll, weights) for p0 in unique]
    best = fit.fit_summary(min(map_function(_fit_candidate, tasks), key=lambda res: res.cost), image, weights)
    x = [float(v) for v in best["x"]]
    params = [r_fixed, *x] if r_fixed is not None else x
    out = {"ghost": ghost, "noll": list(noll), "params": params, "cost": best["cost"], "residual_frac": best["residual_frac"],
           "r_fixed": r_fixed is not None, "n_seeds": len(unique), "stamp": st.stamp()}
    folder.mkdir(parents=True, exist_ok=True)
    viewer_fit_path(folder, img, ghost).write_text(json.dumps(out))
    return out


def warm_up() -> None:
    """Imports and JIT work of a fit worker, done before the first spot asks for it."""
    fit.render_model_shared([1.8, 0, 0, 0], np.ones((121, 121)), 0.0, 1.0, model="tophat_bl", engine="matrix", fnum=4)


def _seed_from(params, noll) -> list[float]:
    """[r, x0, y0, c4, *cJ] of `params`, cut or padded to the terms of `noll`."""
    extra = list(params[4:4 + len(noll) - 1])
    return list(params[:4]) + extra + [0.0] * (len(noll) - 1 - len(extra))


def _nearest_seed(st: RunState, fiber: str, row, noll) -> list[float]:
    """The saved fit of the same fiber nearest in stage position (and time), for a spot never fitted."""
    candidates = [(abs(int(st.df.loc[i, "x_position"]) - int(row["x_position"])), i) for i in st.fits
                  if i in st.df.index and st.df.loc[i, "fiber"] == fiber]
    if not candidates:
        return [1.0, 0.0, 0.0, 0.0] + [0.0] * (len(noll) - 1)
    return _seed_from(st.fits[min(candidates)[1]][0]["params"], noll)


def fov_png(path: Path, cache_dir: Path, factor: int = 4) -> Path | None:
    """Full field of view (results/equalizer_raw_image.npy) binned by `factor`, asinh-stretched to an 8-bit PNG."""
    raw = path / "results" / "equalizer_raw_image.npy"
    if not raw.exists():
        return None
    key = hashlib.sha1(f"{raw}:{raw.stat().st_mtime}:{factor}".encode()).hexdigest()
    out = cache_dir / "fov" / f"{key}.png"
    if not out.exists():
        image = np.load(raw).astype(np.float32)
        h, w = (s // factor * factor for s in image.shape)
        binned = image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        low, high = np.percentile(binned, [1, 99.9])
        stretched = np.arcsinh(np.clip(binned - low, 0, None) / max(high - low, 1) * 20) / np.arcsinh(20)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(stretched, 0, 1) * 255).astype(np.uint8)).save(out)
    return out


def fov_fibers(path: Path) -> dict:
    """Sensor shape and the median spot centre (full-sensor px) of each fiber."""
    st = state(path)
    raw = path / "results" / "equalizer_raw_image.npy"
    shape = list(np.load(raw, mmap_mode="r").shape) if raw.exists() else None
    centres = st.df.groupby("fiber")[["mom_center_x", "mom_center_y"]].median()
    return {"shape": shape, "fibers": [{"fiber": f, "x": float(r["mom_center_x"]), "y": float(r["mom_center_y"])}
                                       for f, r in centres.iterrows()]}
