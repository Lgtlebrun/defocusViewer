"""Analysis variants of a run: a sibling rundir `<run>_<suffix>` sharing the run's data (data/ is a symlink) whose
meta/config.yaml asks for another source or aperture model, with only the acquisition-time results, plots and log. The
whole analysis then runs on the variant, so the run's own results are never overwritten; the viewer lists a run and
its variants as one family."""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from DefocusTest.fit import SOURCE_MODELS
from DefocusTest.psf import PUPIL_SHAPES, split_model
from DefocusTest.psf_batch import BATCH_MODELS

from . import catalog
from .config import Config

ACQUISITION_RESULTS = ("equalizer_", "fiber_label_map", "labeled_fiber_centers", "fiber_centers", "master_dark")
ACQUISITION_PLOTS = ("spots_centers.pdf", "example_spots")
PROVENANCE = "variant.json"
SUFFIX = re.compile(r"^[A-Za-z0-9][\w-]{0,40}$")
# fit settings a variant may change, with their allowed ranges; the pupil radius goes as 1 / (lambda * fnum), so a wide
# lambda range lets the joint fit absorb the working f-number of an iris set by hand
FIT_LIMITS = {"fnum": (0.5, 32.0), "lbda_coarse_step_nm": (0.5, 50.0)}
LBDA_LIMITS = (300.0, 1100.0)
# image oversampling of the matrix PSF engine (DefocusTest fit.image_oversample): PSF and source sampled every 1/q px;
# 2 removes the aliasing of the f/1.8 PSF, whose fringes are finer than a pixel
OVERSAMPLE_LIMITS = (1, 4)
FIT_DEFAULTS = {"fnum": 8.0, "lbda_range_nm": [600.0, 700.0], "lbda_coarse_step_nm": 5.0, "image_oversample": 1}


def family(config: Config, key: str) -> list[dict]:
    """The run of `key` (or of the variant `key`) first, then its variants: [{key, suffix, model, made}], a variant being a
    local run of the same folder named `<run>_<suffix>`."""
    runs = catalog.local_runs(config)
    root = parent(config, key) or key
    group, name = root.rsplit("/", 1)
    members = [root] + sorted(k for k in runs if k.startswith(f"{root}_") and k.rsplit("/", 1)[0] == group)
    out = []
    for member in members:
        path = runs[member]
        provenance = path / "meta" / PROVENANCE
        made = json.loads(provenance.read_text()) if provenance.exists() else {}
        out.append({"key": member, "suffix": member[len(root) + 1:] or None, "model": model_of(path), "made": made})
    return out


def parent(config: Config, key: str) -> str | None:
    """The run a variant `key` belongs to (the longest local run whose name prefixes it), None for an original run."""
    runs = catalog.local_runs(config)
    group = key.rsplit("/", 1)[0]
    candidates = [k for k in runs if k != key and key.startswith(f"{k}_") and k.rsplit("/", 1)[0] == group]
    return max(candidates, key=len) if candidates else None


def model_of(path: Path) -> str:
    """Fit model of a run: the one of its joint fit when it has one, else the one its config asks for."""
    saved = path / "results" / "fit_model.npy"
    if saved.exists():
        import numpy as np
        return str(np.atleast_1d(np.load(saved, allow_pickle=True))[0])
    config = catalog.open_rundir(path).loadConfig() if (path / "meta" / "config.yaml").exists() else {}
    return str((config.get("fit") or {}).get("model", ""))


def model_name(source: str, aperture: str) -> str:
    """Fit model name of a source model and an aperture ("tophat_bl", "tophat_bl@circle")."""
    if source not in SOURCE_MODELS or aperture not in PUPIL_SHAPES:
        raise ValueError(f"source in {sorted(SOURCE_MODELS)}, aperture in {PUPIL_SHAPES}")
    return source if aperture == "hexagon" else f"{source}@{aperture}"


def fit_settings(config: Config, key: str) -> dict:
    """The fit settings of run `key` a variant may change (FIT_DEFAULTS where its config has none)."""
    path = catalog.resolve(config, key)
    fit_config = (catalog.open_rundir(path).loadConfig().get("fit") or {}) if (path / "meta" / "config.yaml").exists() else {}
    out = {}
    for name, default in FIT_DEFAULTS.items():
        value = fit_config.get(name, default)
        if isinstance(default, list):
            out[name] = [float(v) for v in value]
        else:
            out[name] = int(value) if isinstance(default, int) else float(value)
    return out


def checked_fit(fit: dict | None) -> dict:
    """The fit settings of a variant request ({} when none), ValueError for an unknown setting or a value out of range."""
    out = {}
    for name, value in (fit or {}).items():
        if name == "lbda_range_nm":
            lo, hi = (float(v) for v in value)
            if not LBDA_LIMITS[0] <= lo < hi <= LBDA_LIMITS[1]:
                raise ValueError(f"lbda_range_nm: increasing pair within {LBDA_LIMITS[0]:g}-{LBDA_LIMITS[1]:g} nm")
            out[name] = [lo, hi]
        elif name in FIT_LIMITS:
            lo, hi = FIT_LIMITS[name]
            if not lo <= float(value) <= hi:
                raise ValueError(f"{name}: within {lo:g}-{hi:g}")
            out[name] = float(value)
        elif name == "image_oversample":
            lo, hi = OVERSAMPLE_LIMITS
            if isinstance(value, bool) or float(value) != int(float(value)) or not lo <= int(float(value)) <= hi:
                raise ValueError(f"image_oversample: an integer within {lo}-{hi}")
            out[name] = int(float(value))
        else:
            raise ValueError(f"fit settings a variant may change: {', '.join(['lbda_range_nm', *FIT_LIMITS, 'image_oversample'])}")
    return out


def create(config: Config, key: str, suffix: str, source: str, aperture: str, ghost: str, fit: dict | None = None) -> str:
    """Creates the variant `<run>_<suffix>` of the run of `key` asking for `source` and `aperture`, and the fit settings
    `fit` (see checked_fit) in place of the run's; returns its key. ValueError for a bad suffix, model or setting,
    FileExistsError when that variant exists."""
    if not SUFFIX.match(suffix):
        raise ValueError("suffix: letters, digits, '-' and '_', at most 41 characters")
    model = model_name(source, aperture)
    fit = checked_fit(fit)
    if fit.get("image_oversample", 1) > 1 and source not in BATCH_MODELS:
        raise ValueError(f"image_oversample > 1 needs a source among {', '.join(BATCH_MODELS)}")
    root = parent(config, key) or key
    path = catalog.resolve(config, root)
    target = path.parent / f"{path.name}_{suffix}"
    if target.exists():
        raise FileExistsError(f"{target.name} exists")
    target.mkdir()
    try:
        (target / "data").symlink_to((path / "data").resolve())
        (target / "meta").mkdir()
        if (path / "meta" / "_finished").exists():
            shutil.copy2(path / "meta" / "_finished", target / "meta" / "_finished")
        settings = catalog.open_rundir(path).loadConfig() if (path / "meta" / "config.yaml").exists() else {}
        settings.setdefault("fit", {})
        settings["fit"]["model"] = model
        settings["fit"]["ghost_mask"] = ghost == "on"
        settings["fit"].update(fit)
        if int(settings["fit"].get("image_oversample", 1)) > 1 and source not in BATCH_MODELS:      # from the run's config
            settings["fit"]["image_oversample"] = 1
        (target / "meta" / "config.yaml").write_text(yaml.safe_dump(_plain(settings), sort_keys=False))
        (target / "meta" / PROVENANCE).write_text(json.dumps({
            "parent": root, "model": model, "source": source, "aperture": aperture, "ghost": ghost, "fit": fit,
            "parent_model": model_of(path), "made": datetime.now().isoformat(timespec="seconds")}, indent=1))
        (target / "results").mkdir()
        for file in (path / "results").iterdir():
            if file.is_file() and file.name.startswith(ACQUISITION_RESULTS):
                shutil.copy2(file, target / "results" / file.name)
        (target / "plot").mkdir()
        for name in ACQUISITION_PLOTS:
            item = path / "plot" / name
            if item.is_dir():
                shutil.copytree(item, target / "plot" / name)
            elif item.exists():
                shutil.copy2(item, target / "plot" / name)
        _copy_acquisition_log(path / "log.log", target / "log.log", root, model, fit)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return f"{root}_{suffix}"


def _plain(value):
    """The config as plain dicts, lists and scalars: configs saved by ruamel keep its scalar types (ScalarFloat,
    ScalarBoolean...), which yaml.safe_dump refuses."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or type(value).__name__ == "ScalarBoolean":
        return bool(value)
    for kind in (int, float, str):
        if isinstance(value, kind):
            return kind(value)
    return value


def _copy_acquisition_log(source: Path, target: Path, root: str, model: str, fit: dict) -> None:
    """The acquisition part of the run's log.log (up to its first analysis), then a line saying what the variant is."""
    lines = []
    if source.exists():
        for line in source.read_text(errors="replace").splitlines(keepends=True):
            if "Analysis steps of" in line:
                break
            lines.append(line)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changed = "".join(f", fit.{k}={v}" for k, v in fit.items())
    lines.append(f"({stamp}) [INFO] [defocusViewer] : variant of {root.rsplit('/', 1)[-1]} with fit.model={model}{changed}\n")
    target.write_text("".join(lines))
