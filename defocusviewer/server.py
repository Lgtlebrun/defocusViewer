"""HTTP API and page of the viewer. Every request but /login needs the login cookie (the secret of config.token)."""
import asyncio
import hmac
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import catalog, cluster, figures, jobs, jointfit, plots, spots, summary
from .config import Config

STATIC = Path(__file__).parent / "static"
COOKIE = "defocusviewer"
PARTITIONS = ("express", "standard", "long", "verylong")
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


class FetchRequest(BaseModel):
    run: str
    analyse: bool = False
    ghost: str = "on"


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
    def fit_pool() -> ProcessPoolExecutor:
        if refits["pool"] is None:
            refits["pool"] = ProcessPoolExecutor(config.refit_workers, mp_context=multiprocessing.get_context("spawn"),
                                                 initializer=os.nice, initargs=(10,))
        return refits["pool"]

    if background:
        remote.start()
        sampler.start()
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
                "launch_hosts": config.launch_hosts, "partitions": PARTITIONS, "data_root": str(config.data_root)}

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

    @app.get("/api/run/fibers")
    def run_fibers(run: str):
        return spots.fiber_table(run_path(run))

    @app.get("/api/run/fov")
    def run_fov(run: str):
        return spots.fov_fibers(run_path(run))

    @app.get("/api/run/fov.png")
    def run_fov_png(run: str):
        png = spots.fov_png(run_path(run), config.cache_dir)
        if png is None:
            raise HTTPException(404, "no full-frame image in results/")
        return FileResponse(png)

    @app.get("/api/spot")
    def spot(run: str, img: str, ghost: str = "on"):
        try:
            return spots.detail(run_path(run), img, _ghost(ghost), viewer_fits(run))
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
        return await asyncio.to_thread(spots.detail, path, request.img, ghost, viewer_fits(request.run))

    @app.get("/api/analyses")
    def run_analyses(run: str, ghost: str = "on"):
        path = run_path(run)
        info = jobs.analyses(path, _ghost(ghost))
        free = sampler.free_gpus()
        uses_gpu = bool(set(info["default_steps"]) & jobs.GPU_STEPS)
        return {**info, "device": catalog.device_of(config, run), "free_gpus": free,
                "default_resources": jobs.default_resources(config, free, uses_gpu),
                "cpu_resources": jobs.default_resources(config, free, False)}

    @app.post("/api/jobs")
    def start_job(request: JobRequest):
        path, ghost = run_path(request.run), _ghost(request.ghost)
        device = catalog.device_of(config, request.run)
        if device is None:
            raise HTTPException(400, "only runs under the data root can be analysed (the entry points find them there)")
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
        free = sampler.free_gpus()
        return {**jointfit.options(config, run), "free_gpus": free, "default_resources": jobs.default_resources(config, free, True),
                "cpu_resources": jobs.default_resources(config, free, False)}

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

    @app.post("/api/fetch")
    def fetch(request: FetchRequest):
        parts = request.run.split("/")
        if request.run not in remote.snapshot()["runs"] or len(parts) != 3:
            raise HTTPException(404, f"{request.run} is not a run on pos-control")
        argv = [config.python, "-m", "defocusviewer", "fetch", parts[0], parts[2]]
        argv += ["--analyse", "--ghost", _ghost(request.ghost)] if request.analyse else []
        res = {"mode": "local", "launch_host": config.launch_hosts[0]}
        label = "fetch + analyse" if request.analyse else "fetch"
        return jobs.launch(config, registry, request.run, argv, res, label, force=True)

    @app.get("/api/jobs")
    def list_jobs():
        return registry.refresh()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        try:
            return jobs.tail(Path(registry.get(job_id)["log"]))
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

    return app
