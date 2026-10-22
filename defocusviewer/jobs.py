"""Analysis jobs started from the viewer: the analyses a rundir offers, default resources, the command, the launch
(detached on a GPU host, or on Slurm), the registry of jobs, and the guard against two jobs on one rundir."""
import json
import os
import re
import secrets
import shlex
import threading
from datetime import datetime
from pathlib import Path

from amkit import resources
from DefocusTest.analysis import remote

from . import catalog
from .config import Config

FLOW_MODULES = {"stability": "DefocusTest.analysis.run_stability", "detection": "DefocusTest.analysis.run_detection"}
EXTRA_MODULES = {"calibration": "DefocusTest.analysis.run_calibration",
                 "leave_one_out": "DefocusTest.analysis.run_leave_one_out",
                 "temperature": "DefocusTest.analysis.run_temperature", "desi_fit": "DefocusTest.analysis.run_desi_fit"}
GPU_STEPS = {"fit", "ghost_mask", "refit", "series"}
FETCH_MODULE = "defocusviewer"
EXIT_LINE = re.compile(r"^__exit__ (\d+)$", re.M)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Conflict(RuntimeError):
    """Another job already works on the rundir."""


class NullLog:
    """Logger stand-in: reading the settings of a rundir must not write to its logs."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def analyses(path: Path, ghost: str) -> dict:
    """The steps of the rundir's flow (remote.steps, for this ghost setting), those a full run makes, and the extra
    entry points; `reason` when the rundir has no analysis entry point."""
    flow = catalog.flow(path)
    if flow not in FLOW_MODULES:
        return {"flow": flow, "reason": f"no pipeline entry point for {flow or 'a rundir without a hold routine'}",
                "steps": [], "default_steps": [], "extras": [], "viewer_joint_fit": True}
    if not (path / "meta" / "config.yaml").exists():
        return {"flow": flow, "reason": "no meta/config.yaml", "steps": [], "default_steps": [], "extras": [],
                "viewer_joint_fit": True}
    rundir = catalog.open_rundir(path)
    settings = remote.settings_of(rundir, NullLog(), ghost_mask=ghost)
    steps = list(remote.steps(flow, rundir, NullLog(), settings))
    extras = ["calibration", "leave_one_out", "temperature"] + (["desi_fit"] if settings.with_desi else [])
    return {"flow": flow, "reason": None, "steps": steps, "extras": extras,
            "default_steps": [s for s in steps if s not in remote.OPT_IN_STEPS]}


def default_resources(config: Config, free_gpus: dict[str, list[int]], uses_gpu: bool) -> dict:
    """Up to `gpus_per_job` idle GPUs of the job GPU hosts when the job fits on GPUs and one is free, else a Slurm job of
    `slurm_cpus` cores."""
    hosts = [h for h in config.job_gpu_hosts if free_gpus.get(h)]
    n_free = sum(len(free_gpus[h]) for h in hosts)
    if uses_gpu and n_free:
        return {"mode": "gpu", "n_gpus": min(config.gpus_per_job, n_free), "gpu_hosts": hosts,
                "cpu_workers": config.gpu_job_cpu_workers, "launch_host": config.launch_hosts[0]}
    return {"mode": "slurm", "cpus": config.slurm_cpus, "partition": config.slurm_partition,
            "mem_per_cpu": config.slurm_mem_per_cpu, "time": None}


def command(config: Config, module: str, device: str, name: str, steps: list[str] | None, ghost: str | None,
            res: dict) -> list[str]:
    """argv of an analysis entry point on rundir `name` of `device` (always --now: the local copy is used as it is)."""
    argv = [config.python, "-m", module, name, "--device", catalog.device_id(device), "--now"]
    if module in FLOW_MODULES.values():
        argv += ["--only", *steps] if steps else []
    if ghost and module in (*FLOW_MODULES.values(), EXTRA_MODULES["desi_fit"]):
        argv += ["--ghost-mask", ghost]
    if res["mode"] == "gpu":
        if module not in FLOW_MODULES.values():
            raise ValueError("only the stability / detection analyses run on GPUs")
        argv += ["--gpu", str(int(res["n_gpus"])), "--gpu-hosts", ",".join(res["gpu_hosts"]),
                 "--n-workers", str(int(res["cpu_workers"]))]
    else:
        argv += ["--n-workers", str(int(res["cpus"]))]
    return argv


def jointfit_command(config: Config, key: str, mode: str, reference: str | None, fnum: float, ghost: str, fit_theta: bool,
                     res: dict) -> list[str]:
    """argv of the viewer joint fit of run `key` (defocusviewer jointfit) on the resources `res`."""
    argv = [config.python, "-m", "defocusviewer", "jointfit", key, "--mode", mode, "--fnum", repr(float(fnum)), "--ghost", ghost]
    argv += ["--reference", reference] if reference else []
    argv += ["--fit-theta"] if fit_theta else []
    if res["mode"] == "gpu":
        return argv + ["--gpu", str(int(res["n_gpus"])), "--gpu-hosts", ",".join(res["gpu_hosts"])]
    return argv + ["--n-workers", str(int(res["cpus"]))]


def _detached(argv: list[str], env: dict) -> str:
    """Shell command running `argv` in its own session (so cancel can signal the whole group), nice 10, ending with
    a line `__exit__ <code>` in its output."""
    inner = " ".join([*(f"{k}={shlex.quote(v)}" for k, v in env.items()), "nice -n 10", shlex.join(argv)])
    return f"setsid bash -c {shlex.quote(inner + '; echo __exit__ $?')}"


class Registry:
    """One JSON file per job in the jobs folder (shared by the server and the fetch command)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self._lock = threading.Lock()

    def save(self, job: dict) -> dict:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{job['id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1))
        tmp.replace(path)
        return job

    def get(self, job_id: str) -> dict:
        if not re.fullmatch(r"[\w-]+", job_id):
            raise KeyError(job_id)
        return json.loads((self.folder / f"{job_id}.json").read_text())

    def all(self, limit: int = 100) -> list[dict]:
        paths = sorted(self.folder.glob("*.json"), reverse=True)[:limit] if self.folder.exists() else []
        return [json.loads(p.read_text()) for p in paths]

    def refresh(self) -> list[dict]:
        """The jobs, newest first, with the state of the running ones updated (one ps per host, one squeue)."""
        with self._lock:
            jobs = self.all()
            running = [j for j in jobs if j["state"] in ("running", "pending")]
            if not running:
                return jobs
            slurm = resources.slurm_jobs() if any(j["mode"] == "slurm" for j in running) else {}
            by_host = {}
            for job in running:
                if job["mode"] != "slurm":
                    by_host.setdefault(job["host"], []).append(job)
            for host, host_jobs in by_host.items():
                try:
                    alive = resources.alive([j["pid"] for j in host_jobs], None if resources.is_local(host) else host)
                except RuntimeError:
                    continue
                for job in host_jobs:
                    if job["pid"] not in alive:
                        self._finish_local(job)
            for job in running:
                if job["mode"] == "slurm":
                    self._update_slurm(job, slurm.get(str(job["slurm_id"])))
            return jobs

    def _finish_local(self, job: dict):
        found = EXIT_LINE.findall(tail(Path(job["log"]), 4096))
        code = int(found[-1]) if found else None
        job.update(state="done" if code == 0 else "failed" if code is not None else "killed", exit_code=code,
                   ended=datetime.now().isoformat(timespec="seconds"))
        self.save(job)

    def _update_slurm(self, job: dict, queued: dict | None):
        if queued is not None:
            state = "running" if "RUNNING" in queued["states"] else "pending"
            if state != job["state"]:
                job["state"] = state
                self.save(job)
            return
        final = resources.slurm_state(job["slurm_id"])
        if final is None:
            return
        job.update(state="done" if final[0] == "COMPLETED" else final[0].lower(), exit_code=final[1],
                   ended=datetime.now().isoformat(timespec="seconds"))
        self.save(job)


def running_on(config: Config, registry: Registry, run_key: str, name: str) -> list[str]:
    """What is already working on rundir `name`: viewer jobs, analysis processes on the launch hosts (other sessions
    run them too), Slurm jobs named after it."""
    found = [f"viewer job {j['id']} ({j['label']})" for j in registry.refresh()
             if j["run"] == run_key and j["state"] in ("running", "pending")]
    for host in config.launch_hosts:
        target = None if resources.is_local(host) else host
        out = resources._shell("ps -eo pid=,user=,args=", target) or ""
        for line in out.splitlines():
            pid, user, args = (line.split(None, 2) + ["", ""])[:3]
            if ("DefocusTest.analysis.run_" in args or "defocusviewer jointfit" in args) and any(w.endswith(name) for w in args.split()):
                found.append(f"{host} pid {pid} ({user}): {args[:160]}")
    found += [f"Slurm job {jid} ({j['name']})" for jid, j in resources.slurm_jobs().items() if j["name"] == name]
    return found


def launch(config: Config, registry: Registry, run_key: str, argv: list[str], res: dict, label: str,
           ghost: str | None = None, force: bool = False, rundir: Path | None = None) -> dict:
    """Starts `argv` for run `run_key` (detached on the launch host, or as a Slurm job of the rundir's name) and
    registers it; Conflict when something already works on the rundir and not `force`."""
    name = run_key.rsplit("/", 1)[1]
    if not force:
        busy = running_on(config, registry, run_key, name)
        if busy:
            raise Conflict("; ".join(busy))
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    job = {"id": job_id, "run": run_key, "label": label, "argv": argv, "ghost": ghost, "resources": res,
           "mode": res["mode"], "started": datetime.now().isoformat(timespec="seconds"), "state": "running"}
    env = {"POS_LOCAL_DATA": str(config.data_root), "PYTHONUNBUFFERED": "1"}
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    if res["mode"] == "slurm":
        job_name = name if rundir is not None else f"dv-{name}"
        out = (rundir or config.jobs_dir) / "slurm-%j.out"
        os.environ.update(env)
        job["slurm_id"] = resources.submit_slurm(
            shlex.join(argv), cpus=int(res["cpus"]), partition=res["partition"], job_name=job_name, out=out,
            mem_per_cpu=res["mem_per_cpu"], time=res.get("time") or None, cwd=config.state_dir)
        job.update(state="pending", log=str(out).replace("%j", job["slurm_id"]))
    else:
        host = res.get("launch_host") or config.launch_hosts[0]
        log = config.jobs_dir / f"{job_id}.log"
        job["pid"] = resources.run_detached(_detached(argv, env), None if resources.is_local(host) else host, log,
                                            config.state_dir)
        job.update(host=host, log=str(log))
    return registry.save(job)


def cancel(job: dict) -> None:
    """Stops a job: scancel on Slurm, else SIGTERM to its whole session on its host."""
    if job["mode"] == "slurm":
        resources._shell(f"scancel {int(job['slurm_id'])}")
    else:
        host = job["host"]
        resources._shell(f"kill -TERM -- -{int(job['pid'])}", None if resources.is_local(host) else host)


def tail(path: Path, size: int = 32768) -> str:
    """Last `size` bytes of a log, without terminal colour codes ('' when it does not exist yet)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - size))
            return ANSI.sub("", f.read().decode(errors="replace"))
    except FileNotFoundError:
        return ""
