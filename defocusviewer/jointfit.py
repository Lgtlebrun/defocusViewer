"""Joint fit of a run the pipeline cannot analyse (no hold routine, no config: aperture fits, image spots, scans...),
made by the viewer on every spot of the run, and saved into its results/ under the names of JointHexFitAnalysis, so the
spot panel (and notebooks) read it like a pipeline fit; `viewer_joint_fit.json` says how it was made.

Modes:
- "seeded": theta of a reference run held fixed (or searched around it, fit_theta), the wavelength of each fiber refitted
  from the reference's (fit_joint lbda_per_fiber, theta_fixed, lbda_center_init);
- "scratch": theta and the wavelengths searched over their whole range, as the pipeline does;
- "reuse": no fit, the reference's theta and the wavelengths of the matched fibers copied (the spots are then fitted one by
  one when opened).
With ghost "on" the ghost hunt of the pipeline follows: unmasked fit, ghost boxes found in its residuals
(postprocessing.fiber_ghost_table), masked fit seeded by the unmasked one.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from DefocusTest import fit
from DefocusTest.analysis.SpotLabeler import drop_overwritten_images, load_labeled_dataframe, load_labeled_result_df
from DefocusTest.analysis.ZernikeStabilityAnalysis import load_crop_side
from DefocusTest.devices import DESI, FIDUCIALS
from DefocusTest.imgDF import imgDF, load_stack
from DefocusTest.postprocessing import GHOST_MASKS, GhostMasks, detect_frames, fiber_ghost_table

from . import catalog
from .config import Config

MODES = ("seeded", "scratch", "reuse")
PROVENANCE = "viewer_joint_fit.json"
MATCH_TOLERANCE_PX = 50          # a fiber is matched to the reference's nearest spot within this distance on the sensor
C4_GRID = np.arange(-4.0, 4.01, 0.25)
FNUM_IN_NAME = (r"fnum[-_]?(\d+(?:p\d+)?)", r"apf(\d+(?:p\d+)?)", r"(?:^|[-_])f(\d+(?:p\d+)?)(?:[-_]|$)")
# fit settings when there is no reference: those of the f/4 stability runs (meta/config.yaml, fit section, 2026-10)
DEFAULT_SETTINGS = dict(model="tophat_bl", engine="matrix", noll=(4, 5, 6, 7, 8), crop_side=121, supersample=32,
                        psf_oversample=2, n_rounds=2, lbda_range_nm=(600.0, 700.0), lbda_coarse_step_nm=5.0,
                        lbda_fine_step_nm=0.5, theta_fine_step=0.1, hex_scale_fine_step=0.005)


def guess_fnum(path: Path) -> tuple[float | None, str]:
    """(f-number, where it was read): config fit.fnum, results/fnum.npy, the run name (fnum4, apf1p8, f8), else None."""
    config = path / "meta" / "config.yaml"
    if config.exists():
        fnum = (catalog.open_rundir(path).loadConfig().get("fit") or {}).get("fnum")
        if fnum:
            return float(fnum), "config"
    if (path / "results" / "fnum.npy").exists():
        return float(np.atleast_1d(np.load(path / "results" / "fnum.npy"))[0]), "results"
    slug = path.name.split("_", 1)[-1].lower()
    for pattern in FNUM_IN_NAME:
        found = re.search(pattern, slug)
        if found:
            return float(found.group(1).replace("p", ".")), "run name"
    return None, "unknown"


def _time(name: str) -> datetime | None:
    try:
        return datetime.strptime(name[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def references(config: Config, key: str) -> list[dict]:
    """Local runs with a joint fit that can seed this one: same device, same f-number first, then nearest in time."""
    path = catalog.resolve(config, key)
    fnum, _ = guess_fnum(path)
    when = _time(path.name)
    device = key.split("/")[0]
    out = []
    for other, other_path in catalog.local_runs(config).items():
        if other == key or other.split("/")[0] != device or not (other_path / "results" / "theta_shared.npy").exists():
            continue
        other_fnum, _ = guess_fnum(other_path)
        other_when = _time(other_path.name)
        gap = abs((other_when - when).total_seconds()) if when and other_when else float("inf")
        out.append({"key": other, "fnum": other_fnum, "same_fnum": fnum is not None and other_fnum == fnum,
                    "hours_apart": None if gap == float("inf") else round(gap / 3600, 1),
                    "viewer_made": (other_path / "results" / PROVENANCE).exists(), "_gap": gap})
    out.sort(key=lambda r: (not r["same_fnum"], r["viewer_made"], r["_gap"]))
    return [{k: v for k, v in r.items() if k != "_gap"} for r in out]


def spot_table(path: Path) -> pd.DataFrame:
    """The spots the joint fit takes: every spot of the run but DESI's, with an x_position (1 when the run has none:
    fit_joint's r(x) line needs x != 0)."""
    df = drop_overwritten_images(load_labeled_dataframe(catalog.open_rundir(path)).reset_index(drop=True))
    df = df[(df["fiber"] != DESI) & (df["img"].astype(str) != "")].reset_index(drop=True)
    if "x_position" not in df.columns or (df["x_position"] == 0).all():
        df["x_position"] = 1
    if "routine" not in df.columns:
        df["routine"] = ""
    return df


def centres(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Median spot centre on the sensor (px) of each fiber."""
    medians = df.groupby("fiber")[["mom_center_x", "mom_center_y"]].median()
    return {fiber: (float(row["mom_center_x"]), float(row["mom_center_y"])) for fiber, row in medians.iterrows()}


def match_fibers(run: dict, reference: dict, tolerance: float = MATCH_TOLERANCE_PX) -> dict[str, str | None]:
    """{fiber of the run: fiber of the reference}: fiducials by label (stable across runs), the others by the nearest
    centre within `tolerance` px (positioner labels are given in detection order, so they swap between runs)."""
    out = {f: f for f in run if f in FIDUCIALS and f in reference}
    pairs = sorted((np.hypot(run[f][0] - reference[r][0], run[f][1] - reference[r][1]), f, r)
                   for f in run if f not in out for r in reference if r not in out.values())
    for distance, fiber, ref in pairs:
        if fiber not in out and ref not in out.values() and distance <= tolerance:
            out[fiber] = ref
    return {f: out.get(f) for f in run}


def reference_settings(path: Path) -> dict:
    """Fit settings, theta and wavelength per fiber of a run with a joint fit."""
    rundir = catalog.open_rundir(path)
    fit_config = ((rundir.loadConfig() if (path / "meta" / "config.yaml").exists() else {}).get("fit") or {})
    one = lambda name: np.atleast_1d(rundir.loadResult(name))[0]   # noqa: E731
    settings = dict(DEFAULT_SETTINGS)
    settings.update({k: type(DEFAULT_SETTINGS[k])(fit_config[k]) if not isinstance(DEFAULT_SETTINGS[k], tuple)
                     else tuple(fit_config[k]) for k in ("supersample", "psf_oversample", "n_rounds", "lbda_range_nm",
                                                          "lbda_coarse_step_nm", "lbda_fine_step_nm", "theta_fine_step",
                                                          "hex_scale_fine_step") if k in fit_config})
    settings.update(model=str(one("fit_model")), engine=str(one("psf_engine")),
                    noll=tuple(int(j) for j in np.atleast_1d(rundir.loadResult("zernike_noll"))),
                    crop_side=load_crop_side(rundir) or DEFAULT_SETTINGS["crop_side"], theta=float(one("theta_shared")))
    try:
        scales = load_labeled_result_df(rundir, "pupil_scales")
        settings["lbda_nm"] = {str(r.fiber): float(r.lbda_nm) for r in scales.itertuples()}
    except (FileNotFoundError, OSError, KeyError):
        shared = float(one("hex_scale_shared"))
        settings["lbda_nm"] = {f: fit.DEFAULT_LBDA * 1e9 / shared for f in centres(spot_table(path))}
    settings["centres"] = centres(spot_table(path))
    return settings


def options(config: Config, key: str) -> dict:
    """What the dialog proposes for a run: f-number guess, references (the first is the default), spot count, and the
    fiber matching against the default reference."""
    path = catalog.resolve(config, key)
    df = spot_table(path)
    fnum, source = guess_fnum(path)
    refs = references(config, key)
    matched = {}
    if refs:
        matched = match_fibers(centres(df), reference_settings(catalog.resolve(config, refs[0]["key"]))["centres"])
    provenance = path / "results" / PROVENANCE
    return {"fnum": fnum if fnum is not None else (refs[0]["fnum"] if refs else None),
            "fnum_source": source if fnum is not None else ("reference" if refs else "unknown"), "references": refs,
            "n_images": len(df), "fibers": sorted(df["fiber"].unique()), "matched": matched,
            "image_side": int(np.load(path / "data" / df["img"].iloc[0], mmap_mode="r").shape[0]) if len(df) else None,
            "has_x_position": bool((df["x_position"] != 1).any()), "default_mode": "seeded" if refs else "scratch",
            "existing": json.loads(provenance.read_text()) if provenance.exists() else None,
            "pipeline_fit": (path / "results" / "theta_shared.npy").exists() and not provenance.exists()}


def _lbda_start(df: pd.DataFrame, settings: dict) -> tuple[dict, dict]:
    """({fiber: start wavelength nm}, {fiber: matched reference fiber}); unmatched fibers start at the median."""
    matched = match_fibers(centres(df), settings["centres"])
    median = float(np.median(list(settings["lbda_nm"].values())))
    return {f: settings["lbda_nm"].get(r, median) if r else median for f, r in matched.items()}, matched


def _c4_start(images: imgDF, theta: float, lbda_nm: dict, settings: dict, fnum: float, k: int, n_max: int = 15) -> float:
    """Median over (up to n_max) images of the c4 of C4_GRID whose model fits best, every other term at 0: a start for
    fit_joint's per-image fits that is not stuck at its default -0.5 for strongly defocused spots."""
    picks = images.index[np.unique(np.linspace(0, len(images) - 1, min(n_max, len(images))).round().astype(int))]
    best = []
    for i in picks:
        image = images.get_img(i)
        scale = fit.DEFAULT_LBDA * 1e9 / lbda_nm[images.loc[i, "fiber"]]
        costs = [float(((fit.render_model_shared([1.0, 0.0, 0.0, c4], image, theta, scale, model=settings["model"],
                                                  supersample=settings["supersample"], psf_oversample=settings["psf_oversample"],
                                                  engine=settings["engine"], k=k, fnum=fnum) - image) ** 2).sum())
                 for c4 in C4_GRID]
        best.append(C4_GRID[int(np.argmin(costs))])
    return float(np.median(best))


def _pupil_scales_table(scales: dict, lbda_range_nm) -> pd.DataFrame:
    """As JointHexFitAnalysis._pupil_scales_table: {fiber: hex_scale} with its wavelength, flagged at a range edge."""
    lo, hi = lbda_range_nm
    rows = []
    for fiber, scale in scales.items():
        lbda_nm = fit.DEFAULT_LBDA * 1e9 / scale
        rows.append(dict(fiber=fiber, scale=float(scale), lbda_nm=float(lbda_nm),
                         at_bound=bool(lbda_nm <= lo + 1e-6 or lbda_nm >= hi - 1e-6)))
    return pd.DataFrame(rows)


def _save_common(rundir, theta: float, scales: dict, settings: dict, fnum: float, k: int, provenance: dict) -> None:
    rundir.saveResult(theta, "theta_shared")
    rundir.saveResult(np.array("lbda_per_fiber"), "pupil_mode")
    (rundir.results / "hex_scale_shared.npy").unlink(missing_ok=True)
    rundir.saveResultDf(_pupil_scales_table(scales, settings["lbda_range_nm"]), "pupil_scales")
    rundir.saveResult(np.array(settings["noll"]), "zernike_noll")
    rundir.saveResult(np.array(settings["model"]), "fit_model")
    rundir.saveResult(np.array(settings["engine"]), "psf_engine")
    rundir.saveResult(np.array(fnum), "fnum")
    rundir.saveResult(np.array(k), "fit_crop_side")
    (rundir.results / PROVENANCE).write_text(json.dumps(provenance, indent=1, default=str))


def reuse(config: Config, key: str, reference: str, fnum: float) -> dict:
    """Copies the reference's theta and the wavelengths of the matched fibers into the run (no fit)."""
    path = catalog.resolve(config, key)
    settings = reference_settings(catalog.resolve(config, reference))
    df = spot_table(path)
    lbda_nm, matched = _lbda_start(df, settings)
    k = min(settings["crop_side"], int(np.load(path / "data" / df["img"].iloc[0], mmap_mode="r").shape[0]))
    provenance = {"mode": "reuse", "reference": reference, "matched": matched, "fnum": fnum, "theta": settings["theta"],
                  "made": datetime.now().isoformat(timespec="seconds")}
    for stale in ("joint_fit_diag_records", "joint_fit_diag_records_unmasked", "joint_fit_history", "r_lines", GHOST_MASKS):
        (path / "results" / f"{stale}.hdf5").unlink(missing_ok=True)      # they belong to an earlier fit
    _save_common(catalog.open_rundir(path), settings["theta"], {f: fit.DEFAULT_LBDA * 1e9 / l for f, l in lbda_nm.items()},
                 settings, fnum, k, provenance)
    return provenance


def _residuals(images: imgDF, diag: pd.DataFrame, theta: float, scales: dict, settings: dict, fnum: float, k: int):
    """(frames, residuals) of the records of a joint fit, model - data, unmasked (as postprocessing.sweep_residuals)."""
    row_of = {(f, i): r for r, (f, i) in enumerate(zip(images["fiber"], images["img"]))}
    columns = ["r", "x0", "y0", "c4"] + [f"c{j}" for j in settings["noll"][1:]]
    residuals = []
    for _, record in diag.iterrows():
        image = images.get_img(images.index[row_of[(record["fiber"], record["img"])]])
        model = fit.render_model_shared([record[c] for c in columns], image, theta, fit.scale_of(scales, record["fiber"]),
                                        model=settings["model"], supersample=settings["supersample"],
                                        psf_oversample=settings["psf_oversample"], engine=settings["engine"], k=k, fnum=fnum,
                                        zernike_noll=settings["noll"])
        residuals.append(model - image)
    return diag[["fiber"]].reset_index(drop=True), residuals


def run(config: Config, key: str, mode: str, reference: str | None, fnum: float, ghost: str, fit_theta: bool = False,
        device: str = "cpu", n_workers: int = 1, gpu_hosts=None, log=print) -> dict:
    """The joint fit of the module docstring on every spot of run `key`, saved into its results/."""
    path = catalog.resolve(config, key)
    rundir = catalog.open_rundir(path)
    settings = reference_settings(catalog.resolve(config, reference)) if reference else dict(DEFAULT_SETTINGS)
    df = spot_table(path)
    k = min(settings["crop_side"], int(load_stack(path / "data" / df["img"].iloc[0]).shape[0]))
    images = imgDF(df, rundir=rundir, crop_side=k)
    common = dict(model=settings["model"], n_rounds=settings["n_rounds"], n_workers=n_workers, device=device,
                  gpu_hosts=gpu_hosts, supersample=settings["supersample"], psf_oversample=settings["psf_oversample"],
                  engine=settings["engine"], theta_fine_step=settings["theta_fine_step"],
                  hex_scale_fine_step=settings["hex_scale_fine_step"], k=k, fnum=fnum, zernike_noll=settings["noll"],
                  pupil_mode="lbda_per_fiber", lbda_range_nm=settings["lbda_range_nm"],
                  lbda_coarse_step_nm=settings["lbda_coarse_step_nm"], lbda_fine_step_nm=settings["lbda_fine_step_nm"],
                  log_callback=log)
    seed, matched = {}, {}
    if mode == "seeded":
        lbda_nm, matched = _lbda_start(df, settings)
        seed = dict(lbda_center_init=lbda_nm, **({"theta_center_init": settings["theta"]} if fit_theta
                                                 else {"theta_fixed": settings["theta"]}))
        c4 = _c4_start(images, settings["theta"], lbda_nm, settings, fnum, k)
    else:
        c4 = _c4_start(images, settings.get("theta", 0.0), {f: fit.DEFAULT_LBDA * 1e9 for f in df["fiber"].unique()},
                       settings, fnum, k)
    log(f"{key}: {len(df)} spots, k={k}, fnum={fnum}, mode={mode}, reference={reference}, c4 start {c4:+.2f}, matched {matched}")
    theta, scales, diag, history, r_lines = fit.fit_joint(images, p4_0=(1.0, 0.0, 0.0, c4), **common, **seed)
    provenance = {"mode": mode, "reference": reference, "matched": matched, "fnum": fnum, "fit_theta": fit_theta,
                  "ghost": ghost, "n_images": len(df), "c4_start": c4, "made": datetime.now().isoformat(timespec="seconds")}
    if ghost == "on":
        rundir.saveResultDf(diag, "joint_fit_diag_records_unmasked")
        frames, residuals = _residuals(images, diag, theta, scales, settings, fnum, k)
        table = fiber_ghost_table(frames, detect_frames(residuals), k, fnum)
        rundir.saveResultDf(table, GHOST_MASKS)
        log("ghost boxes: " + ", ".join(f"{r.fiber} ({r.gx:.1f}, {r.gy:.1f}) found {r.n_found}/{r.n_frames}"
                                        for r in table.itertuples()))
        masked = dict(lbda_center_init={f: fit.DEFAULT_LBDA * 1e9 / s for f, s in scales.items()},
                      **({"theta_center_init": theta} if fit_theta or mode == "scratch" else {"theta_fixed": theta}))
        theta, scales, diag, history, r_lines = fit.fit_joint(images, p4_0=(1.0, 0.0, 0.0, c4), **common, **masked,
                                                              weights=GhostMasks(table).for_fibers((k, k)) or None)
    else:
        (rundir.results / "joint_fit_diag_records_unmasked.hdf5").unlink(missing_ok=True)
        (rundir.results / f"{GHOST_MASKS}.hdf5").unlink(missing_ok=True)
    rundir.saveResultDf(diag, "joint_fit_diag_records")
    rundir.saveResultDf(history, "joint_fit_history")
    rundir.saveResultDf(pd.DataFrame([{"fiber": f, "slope": s, "intercept": i, "se_slope": se, "p_value": p}
                                      for f, (s, i, se, p) in r_lines.items()]), "r_lines")
    _save_common(rundir, theta, scales, settings, fnum, k, {**provenance, "theta": theta})
    log(f"theta {theta:.3f} deg; lambda per fiber: " + ", ".join(f"{f} {fit.DEFAULT_LBDA * 1e9 / s:.1f} nm" for f, s in scales.items()))
    return provenance
