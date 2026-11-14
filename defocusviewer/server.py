"""HTTP API and page of the viewer. Every request but /login needs the login cookie (the secret of config.token)."""
import asyncio
import hmac
import json
import multiprocessing
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from DefocusTest import fit, psf, psf_batch

from . import catalog, cluster, figures, jobs, jointfit, lab, plots, spots, summary, variants
from .config import Config

STATIC = Path(__file__).parent / "static"
COOKIE = "defocusviewer"
PARTITIONS = ("express", "standard", "long", "verylong")
DETAIL_CACHE = 32           # spots whose detail is kept in memory (about 0.7 MB each)
LOGIN_HINT = ("<h3>defocusViewer: not logged in</h3><p>Open the login URL printed by <code>defocusviewer url</code> "
              "on farm-gpu1 (it sets a cookie in this browser).</p>")


class SpotRequest(BaseModel):
    run: str
    img: str
    ghost: str = "on"


class JobRequest(BaseModel):
    run: str
    analysis: str = "flow"              # "flow" (the steps of run_<flow>) or an extra entry point
    steps: list[str] = []
    ghost: str = "on"
    resources: dict = {}
    force: bool = False


class JointFitRequest(BaseModel):
    run: str
    mode: str = "seeded"
    reference: str | None = None
    fnum: float
    ghost: str = "on"
    fit_theta: bool = False
    pupil_shape: str = "hexagon"
    resources: dict = {}
    force: bool = False


class PinRequest(BaseModel):
    run: str
    pinned: bool


class VariantRequest(BaseModel):
    run: str
    suffix: str
    source: str = "tophat_bl"
    aperture: str = "hexagon"
    ghost: str = "on"
    fit: dict = {}                      # fit settings in place of the run's (variants.checked_fit)
    resources: dict = {}
    force: bool = False


class FetchRequest(BaseModel):
    run: str
    analyse: bool = False
    ghost: str = "on"


class LabLaunchRequest(BaseModel):
    routine: str
    overrides: dict = {}
    analyse_after: bool = True
    ghost: str = "on"
    confirm: bool = False               # False: only the changes and what blocks a launch (the preview)


class LabRunRequest(BaseModel):
    rundir: str


class LabJobRequest(BaseModel):
    job: str


def _ghost(value: str) -> str:
    if value not in ("on", "off"):
        raise HTTPException(400, "ghost is on or off")
    return value


def checked_resources(config: Config, res: dict) -> dict:
    """The resources of a job request, refused (HTTP 400) outside the allowed values."""
    def bad(what):
        raise HTTPException(400, f"invalid resources: {what}")

    mode = res.get("mode")
    if mode == "gpu":
        hosts = list(res.get("gpu_hosts") or [])
        n_gpus, workers = int(res.get("n_gpus", 0)), int(res.get("cpu_workers", 0))
        if not hosts or any(h not in config.gpu_hosts for h in hosts):
            bad("gpu_hosts")
        if not 1 <= n_gpus <= 8 or not 1 <= workers <= 32:
            bad("n_gpus or cpu_workers")
        if res.get("launch_host") not in config.launch_hosts:
            bad("launch_host")
        return {"mode": "gpu", "n_gpus": n_gpus, "gpu_hosts": hosts, "cpu_workers": workers,
                "launch_host": res["launch_host"]}
    if mode == "cpu":
        cpus = int(res.get("cpus", 0))
        if res.get("host") not in config.cpu_hosts:
            bad("host")
        if not 1 <= cpus <= 384:
            bad("cpus")
        return {"mode": "cpu", "host": res["host"], "cpus": cpus}
    if mode == "slurm":
        cpus, time = int(res.get("cpus", 0)), res.get("time") or None
        if not 1 <= cpus <= 128 or res.get("partition") not in PARTITIONS:
            bad("cpus or partition")
        if not re.fullmatch(r"\d+[MG]", str(res.get("mem_per_cpu", ""))):
            bad("mem_per_cpu")
        if time is not None and not re.fullmatch(r"(\d+-)?\d{1,2}:\d{2}:\d{2}", time):
            bad("time")
        return {"mode": "slurm", "cpus": cpus, "partition": res["partition"], "mem_per_cpu": res["mem_per_cpu"],
                "time": time}
    bad("mode")


def create_app(config: Config, token: str, background: bool = True) -> FastAPI:
    app = FastAPI(title="defocusViewer", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=2000)
    remote, sampler, registry = catalog.RemoteRuns(config), cluster.Sampler(config), jobs.Registry(config.jobs_dir)
    refits = {"pool": None, "inflight": {}}
    os.environ["POS_LOCAL_DATA"] = str(config.data_root)

    def fetch_job(device: str, name: str, analyse: bool, ghost: str) -> dict:
        argv = [config.python, "-m", "defocusviewer", "fetch", device, name]
        argv += ["--analyse", "--ghost", ghost] if analyse else []
        res = {"mode": "local", "launch_host": config.launch_hosts[0]}
        label = "fetch + analyse" if analyse else "fetch"
        return jobs.launch(config, registry, f"{device}/{config.test}/{name}", argv, res, label, force=True)

    def fetching(run: str) -> str | None:
        """Why the rundir is incomplete when a fetch job is still copying it (rsync fills data/ before meta/), else None."""
        for job in registry.all():
            if (job["run"] == run and job["label"].startswith("fetch") and job["state"] in ("running", "pending")
                    and f"fetched {run}" not in jobs.job_tail(job)):
                return (f"job {job['id']} ({job['label']}, started {job['started'].replace('T', ' ')}) is still copying "
                        "this run from the rig: the analyses are available once the copy ends.")
        return None

    rig = lab.Monitor(config, registry, on_finished=lambda run, job: fetch_job(run["device"], run["name"], True, job["ghost"]))
    def fit_pool() -> ProcessPoolExecutor:
        if refits["pool"] is None:
            refits["pool"] = ProcessPoolExecutor(config.refit_workers, mp_context=multiprocessing.get_context("spawn"),
                                                 initializer=os.nice, initargs=(10,))
        return refits["pool"]

    if background:
        remote.start()
        sampler.start()
        rig.start()
        registry.start_refresher()
        for _ in range(config.refit_workers):     # the first spot fit then finds warm workers
            fit_pool().submit(spots.warm_up)

    def run_path(key: str) -> Path:
        try:
            return catalog.resolve(config, key)
        except KeyError:
            raise HTTPException(404, f"no local run {key}") from None

    def viewer_fits(key: str) -> Path:
        return config.spotfits_dir / key

    @app.middleware("http")
    async def cache_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):       # the page's own modules are checked on every load
            response.headers["Cache-Control"] = ("public, max-age=31536000, immutable" if "/vendor/" in request.url.path
                                                 else "no-cache")
        return response

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        if request.url.path == "/login" or hmac.compare_digest(request.cookies.get(COOKIE, ""), token):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "not logged in"}, status_code=401)
        return HTMLResponse(LOGIN_HINT, status_code=401)

    @app.get("/login")
    def login(token: str = "", next: str = "/"):
        if not hmac.compare_digest(token, app.state.token):
            return HTMLResponse(LOGIN_HINT, status_code=401)
        response = RedirectResponse(next if next.startswith("/") and not next.startswith("//") else "/", status_code=303)
        response.set_cookie(COOKIE, token, max_age=365 * 86400, httponly=True, samesite="lax")
        return response

    app.state.token = token

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/config")
    def settings():
        return {"gpu_hosts": config.gpu_hosts, "job_gpu_hosts": config.job_gpu_hosts,
                "launch_hosts": config.launch_hosts, "cpu_hosts": config.cpu_hosts, "cpu_cores": config.cpu_cores,
                "slurm": jobs.slurm_resources(config), "partitions": PARTITIONS, "data_root": str(config.data_root)}

    @app.get("/api/runs")
    def runs():
        return catalog.run_list(config, remote)

    pins_file = config.state_dir / "pinned_runs.json"

    def pinned_runs() -> list[str]:
        return json.loads(pins_file.read_text()) if pins_file.exists() else []

    @app.get("/api/pinned")
    def get_pinned():
        return pinned_runs()

    @app.post("/api/pinned")
    def set_pinned(request: PinRequest):
        runs = [r for r in pinned_runs() if r != request.run] + ([request.run] if request.pinned else [])
        pins_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = pins_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(runs))
        tmp.replace(pins_file)
        return runs

    @app.post("/api/remote/refresh")
    def refresh_remote():
        remote.refresh()
        return {"ok": True}

    @app.get("/api/run/info")
    def run_info(run: str):
        path = run_path(run)
        config_text = (path / "meta" / "config.yaml").read_text() if (path / "meta" / "config.yaml").exists() else None
        if config_text is not None:
            try:
                config_text = yaml.safe_dump(catalog.open_rundir(path).loadConfig(), sort_keys=False)
            except Exception:  # an unreadable config is shown raw
                pass
        results = sorted((p.name, p.stat().st_size) for p in (path / "results").glob("*") if p.is_file())
        try:
            readable = summary.run_summary(path, [j for j in registry.all(500) if j["run"] == run])
        except Exception as e:  # a run in an unexpected format still shows its config and log
            readable = [{"title": "Summary", "rows": [["could not be built", f"{type(e).__name__}: {e}"]]}]
        return {"key": run, "path": str(path), **catalog.summary(path), "device": catalog.device_of(config, run),
                "summary": readable, "config": config_text, "results": results, "log": jobs.tail(path / "log.log", 20000)}

    @app.get("/api/run/plots")
    def run_plots(run: str):
        return plots.list_plots(run_path(run))

    @app.get("/api/run/file")
    def run_file(run: str, path: str):
        try:
            return FileResponse(plots.inside(run_path(run) / "plot", path))
        except ValueError as e:
            raise HTTPException(404, str(e)) from None

    @app.get("/api/run/thumb")
    def run_thumb(run: str, path: str):
        try:
            source = plots.inside(run_path(run) / "plot", path)
        except ValueError as e:
            raise HTTPException(404, str(e)) from None
        if source.suffix.lower() != ".pdf":
            return FileResponse(source)
        return FileResponse(plots.thumbnail(source, config.cache_dir), headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/run/spots")
    def run_spots(run: str):
        return spots.table(run_path(run), viewer_fits(run))

    @app.get("/api/run/fibers/options")
    def fiber_options(run: str):
        table = spots.fiber_frame(run_path(run))
        quantities = [[q, *figures.FIBER_QUANTITIES[q][:2]] for q in figures.FIBER_QUANTITIES
                      if q in table.columns and table[q].notna().any()]
        routines = sorted(set(table["routine"]), key=catalog.routine_sort_key)
        return {"fibers": sorted(table["fiber"].unique()), "routines": routines, "quantities": quantities,
                "x_axes": figures.X_AXES, "has_x_position": bool("x_position" in table and table["x_position"].notna().any())}

    @app.get("/api/run/fibers/figure")
    def fiber_figure(run: str, quantities: str, fibers: str, routines: str, x: str = "time", relative: bool = False,
                     average: int = 1, errors: bool = False, format: str = "png", dpi: int = 150):
        path = run_path(run)
        table = spots.fiber_frame(path)
        quantities_, fibers_, routines_ = (v.split(",") if v else [] for v in (quantities, fibers, routines))
        if (any(q not in figures.FIBER_QUANTITIES or q not in table.columns for q in quantities_) or x not in figures.X_AXES
                or format not in ("png", "pdf", "svg") or not 50 <= dpi <= 600 or not 1 <= average <= 501):
            raise HTTPException(400, "unknown quantity, x axis, format, or dpi / average out of range")
        if not quantities_ or not fibers_ or not routines_:
            raise HTTPException(400, "pick at least one quantity, fiber and routine")
        image = figures.fiber_figure(table, quantities_, fibers_, routines_, x, relative, average, errors,
                                     title=path.name, fmt=format, dpi=dpi)
        media = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}[format]
        headers = {} if format == "png" and dpi <= 200 else {"Content-Disposition": f'attachment; filename="{path.name}_fibers.{format}"'}
        return Response(image, media_type=media, headers=headers)

    @app.get("/api/run/fibers.csv")
    def fiber_csv(run: str):
        path = run_path(run)
        return Response(spots.fiber_frame(path).to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{path.name}_fit_results.csv"'})

    @app.get("/api/run/fov")
    def run_fov(run: str):
        return spots.fov_fibers(run_path(run))

    @app.get("/api/run/fov.png")
    def run_fov_png(run: str):
        png = spots.fov_png(run_path(run), config.cache_dir)
        if png is None:
            raise HTTPException(404, "no full-frame image in results/")
        return FileResponse(png)

    detail_slots = asyncio.Semaphore(4)       # spots computed at once (more only share the GIL)
    detail_cache, cache_lock = OrderedDict(), threading.Lock()

    def cached_detail(path: Path, run: str, img: str, ghost: str) -> dict:
        """spots.detail, kept for the last DETAIL_CACHE spots while the run's results and the viewer fit are unchanged."""
        fits = viewer_fits(run)
        key = (str(path), img, ghost, spots._signature(path), catalog._mtime(spots.viewer_fit_path(fits, img, ghost)))
        with cache_lock:
            if key in detail_cache:
                detail_cache.move_to_end(key)
                return detail_cache[key]
        out = spots.detail(path, img, ghost, fits)
        with cache_lock:
            detail_cache[key] = out
            while len(detail_cache) > DETAIL_CACHE:
                detail_cache.popitem(last=False)
        return out

    @app.get("/api/spot")
    async def spot(request: Request, run: str, img: str, ghost: str = "on"):
        path, ghost = run_path(run), _ghost(ghost)
        async with detail_slots:
            if await request.is_disconnected():      # the viewer has moved on to another spot
                return Response(status_code=204)
            try:
                return await asyncio.to_thread(cached_detail, path, run, img, ghost)
            except KeyError:
                raise HTTPException(404, f"no spot {img}") from None

    @app.post("/api/spot/refit")
    async def spot_refit(request: SpotRequest):
        path, ghost = run_path(request.run), _ghost(request.ghost)
        key = (request.run, request.img, ghost)
        if key not in refits["inflight"]:
            refits["inflight"][key] = asyncio.ensure_future(asyncio.to_thread(
                spots.refit, path, request.img, ghost, viewer_fits(request.run), fit_pool().map))
        try:
            await refits["inflight"][key]
        except Exception as e:
            raise HTTPException(500, f"refit failed: {e}") from None
        finally:
            refits["inflight"].pop(key, None)
        return await asyncio.to_thread(cached_detail, path, request.run, request.img, ghost)

    def resource_options(uses_gpu: bool) -> dict:
        """Free GPUs and cores now, the default resources of the job and those of a CPU-only one."""
        free, cores = sampler.free_gpus(), sampler.free_cores()
        return {"free_gpus": free, "free_cores": cores, "default_resources": jobs.default_resources(config, free, uses_gpu, cores),
                "cpu_resources": jobs.default_resources(config, free, False, cores)}

    @app.get("/api/analyses")
    def run_analyses(run: str, ghost: str = "on"):
        path = run_path(run)
        if reason := fetching(run):
            return {"flow": None, "reason": reason, "steps": [], "default_steps": [], "extras": [], "viewer_joint_fit": False}
        info = jobs.analyses(path, _ghost(ghost))
        uses_gpu = bool(set(info["default_steps"]) & jobs.GPU_STEPS)
        return {**info, "device": catalog.device_of(config, run), **resource_options(uses_gpu)}

    @app.post("/api/jobs")
    def start_job(request: JobRequest):
        path, ghost = run_path(request.run), _ghost(request.ghost)
        device = catalog.device_of(config, request.run)
        if device is None:
            raise HTTPException(400, "only runs under the data root can be analysed (the entry points find them there)")
        if reason := fetching(request.run):
            raise HTTPException(409, reason)
        info = jobs.analyses(path, ghost)
        if info["reason"]:
            raise HTTPException(400, info["reason"])
        if request.analysis == "flow":
            if not request.steps or any(s not in info["steps"] for s in request.steps):
                raise HTTPException(400, f"steps must be among {info['steps']}")
            module, steps, label = jobs.FLOW_MODULES[info["flow"]], request.steps, " ".join(request.steps)
        elif request.analysis in info["extras"]:
            module, steps, label = jobs.EXTRA_MODULES[request.analysis], None, request.analysis
        else:
            raise HTTPException(400, f"unknown analysis {request.analysis}")
        res = checked_resources(config, request.resources)
        try:
            argv = jobs.command(config, module, device, path.name, steps, ghost, res)
            return jobs.launch(config, registry, request.run, argv, res, label, ghost, request.force, rundir=path)
        except jobs.Conflict as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/jointfit/options")
    def jointfit_options(run: str):
        run_path(run)
        return {**jointfit.options(config, run), **resource_options(True)}

    @app.post("/api/jointfit")
    def start_jointfit(request: JointFitRequest):
        path, ghost = run_path(request.run), _ghost(request.ghost)
        if request.mode not in jointfit.MODES or not 0.5 <= request.fnum <= 32 or request.pupil_shape not in ("hexagon", "circle"):
            raise HTTPException(400, "mode is seeded, scratch or reuse; fnum within 0.5-32; pupil_shape hexagon or circle")
        if request.reference is not None and request.reference not in {r["key"] for r in jointfit.references(config, request.run)}:
            raise HTTPException(400, f"{request.reference} is not a reference for this run")
        if request.mode in ("seeded", "reuse") and request.reference is None:
            raise HTTPException(400, f"mode {request.mode} needs a reference")
        if request.mode == "reuse":
            return {"done": True, **jointfit.reuse(config, request.run, request.reference, request.fnum, request.pupil_shape)}
        res = checked_resources(config, request.resources)
        argv = jobs.jointfit_command(config, request.run, request.mode, request.reference, request.fnum, ghost,
                                     request.fit_theta, res, request.pupil_shape)
        try:
            return jobs.launch(config, registry, request.run, argv, res, f"viewer joint fit ({request.mode})", ghost,
                               request.force, rundir=path)
        except jobs.Conflict as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/variants")
    def run_variants(run: str):
        run_path(run)
        return {"family": variants.family(config, run), "sources": sorted(fit.SOURCE_MODELS), "apertures": psf.PUPIL_SHAPES,
                "fit": variants.fit_settings(config, run), "gpu_sources": psf_batch.BATCH_MODELS, **resource_options(True)}

    @app.post("/api/variants")
    def create_variant(request: VariantRequest):
        run_path(request.run)
        ghost, res = _ghost(request.ghost), checked_resources(config, request.resources)
        if res["mode"] == "gpu" and request.source not in psf_batch.BATCH_MODELS:
            raise HTTPException(400, f"the GPU fits implement {', '.join(psf_batch.BATCH_MODELS)}: run {request.source} on the CPU")
        try:
            key = variants.create(config, request.run, request.suffix, request.source, request.aperture, ghost, request.fit)
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        path, device = run_path(key), catalog.device_of(config, key)
        info = jobs.analyses(path, ghost)
        if not info["reason"] and device:
            argv = jobs.command(config, jobs.FLOW_MODULES[info["flow"]], device, path.name, info["default_steps"], ghost, res)
            label = "full analysis (" + variants.model_name(request.source, request.aperture) + ")"
        else:
            options = jointfit.options(config, key)
            reference = options["references"][0]["key"] if options["references"] else None
            if options["fnum"] is None:
                raise HTTPException(400, f"{key} created, but its f-number is unknown: fit it from its Analyses tab")
            argv = jobs.jointfit_command(config, key, "seeded" if reference else "scratch", reference, options["fnum"], ghost,
                                         False, res, request.aperture, request.source)
            label = "viewer joint fit (" + variants.model_name(request.source, request.aperture) + ")"
        job = jobs.launch(config, registry, key, argv, res, label, ghost, force=True, rundir=path)
        return {"variant": key, "job": job}

    @app.post("/api/fetch")
    def fetch(request: FetchRequest):
        parts = request.run.split("/")
        if request.run not in remote.snapshot()["runs"] or len(parts) != 3:
            raise HTTPException(404, f"{request.run} is not a run on pos-control")
        return fetch_job(parts[0], parts[2], request.analyse, _ghost(request.ghost))

    @app.get("/api/jobs")
    def list_jobs():
        return registry.all()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        try:
            return jobs.job_tail(registry.get(job_id))
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "no such job") from None

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            job = registry.get(job_id)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "no such job") from None
        jobs.cancel(job)
        return {"ok": True}

    @app.post("/api/figure.pdf")
    def figure_pdf(panel: dict):
        name = re.sub(r"[^\w.+-]", "_", str(panel.get("filename", "panel")))[:150]
        return Response(figures.panel_pdf(panel), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})

    @app.get("/api/cluster")
    def cluster_state():
        return sampler.snapshot()

    @app.get("/api/lab")
    def lab_state():
        return rig.snapshot()

    @app.get("/api/lab/routine")
    def lab_routine(name: str):
        if name not in lab.ROUTINES:
            raise HTTPException(404, f"no routine {name}")
        try:
            base = lab.base_config(config, name)
        except RuntimeError as e:
            raise HTTPException(502, str(e)) from None
        return {"routine": name, "config": f"{config.lab_root}/{lab.ROUTINES[name]['config']}",
                "fields": [{**f.describe(), "value": lab.get_key(base, f.key)} for f in lab.ROUTINES[name]["fields"]]}

    @app.post("/api/lab/launch")
    def lab_launch(request: LabLaunchRequest):
        if request.routine not in lab.ROUTINES:
            raise HTTPException(404, f"no routine {request.routine}")
        ghost = _ghost(request.ghost)
        try:
            run_config, changes = lab.merged_config(request.routine, lab.base_config(config, request.routine), request.overrides)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except RuntimeError as e:
            raise HTTPException(502, str(e)) from None
        rig.sample()                        # the guard on the rig as it is now, not 15 s ago
        snapshot = rig.snapshot()
        busy = [f"rig unreachable: {snapshot['error']}"] if snapshot["error"] else snapshot["busy"]
        if not request.confirm:
            return {"changes": changes, "busy": busy, "yaml": yaml.safe_dump(run_config, sort_keys=False)}
        if busy:
            raise HTTPException(409, "; ".join(busy))
        label = f"{request.routine} ({lab.get_key(run_config, 'output.path')})"
        try:
            job = lab.launch(config, registry, request.routine, run_config, label, request.analyse_after, ghost)
        except RuntimeError as e:
            raise HTTPException(502, str(e)) from None
        rig.refresh()
        return job

    def known_run(rundir: str) -> dict:
        try:
            return rig.known_rundir(rundir)
        except KeyError:
            raise HTTPException(404, f"{rundir} is not a run the Lab tab follows") from None

    @app.post("/api/lab/stop")
    def lab_stop(request: LabRunRequest):
        run = known_run(request.rundir)
        if not run["writers"]:
            raise HTTPException(409, f"{run['name']} is not running")
        try:
            lab.request_stop(config, run["rundir"])
        except RuntimeError as e:
            raise HTTPException(502, str(e)) from None
        rig.refresh()
        return {"ok": True, "hook": run["hook"]}

    @app.post("/api/lab/kill")
    def lab_kill(request: LabJobRequest):
        try:
            job = registry.get(request.job)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "no such job") from None
        if job.get("mode") != "lab" or job["state"] != "running":
            raise HTTPException(409, f"{request.job} is not a running lab job")
        jobs.cancel(job)
        rig.refresh()
        return {"ok": True}

    @app.get("/api/lab/log", response_class=PlainTextResponse)
    def lab_log(rundir: str):
        run = known_run(rundir)
        try:
            return lab.run_log(config, run["rundir"])
        except RuntimeError as e:
            raise HTTPException(502, str(e)) from None

    return app
