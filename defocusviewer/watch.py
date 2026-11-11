"""Watchdog of the rig: fetches every StabilityTest, TrackingTest and PointingTest run that finishes on pos-control after the watchdog starts, makes its
variant `<run>_tophatbl-circle` (tophat_bl@circle, ghost mask on, lambda scan 650-750 nm) and runs the full analysis of
the variant on the two P100s of farm-gpu1, one run at a time, as soon as no GPU analysis (ours or another session's)
runs there and both P100s are idle.

defocusviewer watch start | stop | status | run. It runs in tmux session `defocusviewer-watch`; `defocusviewer ensure`
(cron) restarts it after a reboot while the `enabled` file exists. State, log and that file: <state_dir>/watch/.
"""
import fcntl
import getpass
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from amkit import resources

from . import catalog, cluster, jobs, variants
from .config import Config

TMUX_SESSION = "defocusviewer-watch"
SUFFIX, SOURCE, APERTURE, GHOST = "tophatbl-circle", "tophat_bl", "circle", "on"
FIT = {"lbda_range_nm": [650.0, 750.0]}
GPU_HOST, N_GPUS = "farm-gpu1", 2
MATCH = re.compile(r"stability|tracking|pointing")   # rig runs the watchdog takes (named stability-..., tracking-..., pointing-...)
FLOWS = ("stability", "tracking", "pointing")        # their analysis flows, checked on the dataframe
RIG_POLL_S, TICK_S = 300, 60
FETCH_TRIES = 3


def folder(config: Config) -> Path:
    return config.state_dir / "watch"


def log(message: str) -> None:
    print(f"({datetime.now():%Y-%m-%d %H:%M:%S}) {message}", flush=True)


def load_state(config: Config) -> dict:
    path = folder(config) / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(config: Config, state: dict) -> None:
    path = folder(config) / "state.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def tmux_alive() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION], capture_output=True).returncode == 0


def start(config: Config) -> None:
    """Enables the watchdog and starts it in tmux. On the first start, the runs already finished on the rig are the
    baseline it leaves alone; a restart keeps the state."""
    folder(config).mkdir(parents=True, exist_ok=True)
    state = load_state(config)
    if not state:
        remote = catalog.RemoteRuns(config).list_remote()
        state = {"started": datetime.now().isoformat(timespec="seconds"),
                 "baseline": sorted(k for k, v in remote.items() if v["finished"]), "runs": {}}
        save_state(config, state)
        print(f"baseline: {len(state['baseline'])} runs already finished on the rig are left alone")
    (folder(config) / "enabled").write_text(datetime.now().isoformat(timespec="seconds") + "\n")
    ensure(config)
    print(f"watchdog enabled, tmux session {TMUX_SESSION}, log {folder(config) / 'watch.log'}")


def stop(config: Config) -> None:
    """Disables the watchdog and kills its tmux session; jobs it started (fetches, analyses) keep running."""
    (folder(config) / "enabled").unlink(missing_ok=True)
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
    print("watchdog stopped; the jobs it started keep running (Jobs tab or `defocusviewer watch status`)")


def ensure(config: Config) -> None:
    """Starts the tmux session when the watchdog is enabled and the session is gone (cron, after a reboot)."""
    if not (folder(config) / "enabled").exists() or tmux_alive():
        return
    command = f"{sys.executable} -m defocusviewer watch run >> {folder(config) / 'watch.log'} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-c", str(config.state_dir), "-s", TMUX_SESSION, command], check=True)


def status(config: Config) -> None:
    state = load_state(config)
    enabled = (folder(config) / "enabled").exists()
    print(f"enabled: {enabled}, tmux session {TMUX_SESSION}: {'up' if tmux_alive() else 'down'}, "
          f"started {state.get('started')}, baseline {len(state.get('baseline', []))} runs")
    registry = jobs.Registry(config.jobs_dir)
    for key, run in state.get("runs", {}).items():
        job_id = run.get("job") or run.get("fetch_job")
        job_state = ""
        if job_id:
            try:
                job_state = f" job {job_id}: {registry.get(job_id)['state']}"
            except (KeyError, FileNotFoundError):
                job_state = f" job {job_id}: ?"
        print(f"  {key}: {run['state']}{job_state} {run.get('note', '')}")


def p100s_busy(config: Config, registry: jobs.Registry) -> str | None:
    """Why the P100s cannot take the next analysis now, None when they can."""
    for job in registry.refresh():
        if (job["state"] in ("running", "pending") and job["mode"] == "gpu"
                and GPU_HOST in job["resources"].get("gpu_hosts", [])):
            return f"viewer job {job['id']} ({job['run'].rsplit('/', 1)[-1]}) uses the GPUs of {GPU_HOST}"
    out = resources._shell("ps -eo pid=,user=,args=", None if resources.is_local(GPU_HOST) else GPU_HOST) or ""
    for line in out.splitlines():
        pid, user, args = (line.split(None, 2) + ["", ""])[:3]
        if ("DefocusTest.analysis.run_" in args or "defocusviewer jointfit" in args) and " --gpu " in f"{args} ":
            return f"GPU analysis pid {pid} ({user}) on {GPU_HOST}: {args[:120]}"
    host = cluster.host_state(GPU_HOST)
    if not host.get("reachable"):
        return f"{GPU_HOST} unreachable: {host.get('hint')}"
    free = host.get("free_gpus") or []
    if len(free) < N_GPUS:
        return f"{len(free)} of the {N_GPUS} P100s of {GPU_HOST} idle"
    return None


def _fetch(config: Config, registry: jobs.Registry, key: str, run: dict) -> None:
    device, _, name = key.split("/", 2)
    argv = [config.python, "-m", "defocusviewer", "fetch", device, name]
    local = config.data_root / device / config.test / name
    if local.exists():                      # a copy made while the run was acquiring: resync it
        argv.append("--force")
    res = {"mode": "local", "launch_host": GPU_HOST}
    job = jobs.launch(config, registry, key, argv, res, "fetch (watchdog)", force=True)
    run.update(state="fetching", fetch_job=job["id"], tries=run.get("tries", 0) + 1)
    log(f"{key}: finished on the rig, fetch job {job['id']}{' (resync)' if local.exists() else ''}")


def _prepare(config: Config, key: str, run: dict) -> None:
    """After the fetch: the variant and the steps of its full analysis, or `skipped` with the reason."""
    device, _, name = key.split("/", 2)
    path = config.data_root / device / config.test / name
    if not (path / "meta" / "_finished").exists():
        run.update(state="failed", note="fetched copy has no meta/_finished")
        return
    variant_key = f"{key}_{SUFFIX}"
    try:
        variants.create(config, key, SUFFIX, SOURCE, APERTURE, GHOST, FIT)
    except FileExistsError:
        log(f"{variant_key} exists already: analysed as it is")
    variant_path = catalog.resolve(config, variant_key)
    info = jobs.analyses(variant_path, GHOST)
    if info["reason"] or info["flow"] not in FLOWS:
        run.update(state="skipped", note=info["reason"] or f"flow {info['flow']}")
        return
    run.update(state="queued", variant=variant_key, flow=info["flow"], steps=info["default_steps"],
               queued=datetime.now().isoformat(timespec="seconds"))
    log(f"{variant_key}: queued for the P100s ({' '.join(info['default_steps'])})")


def _launch(config: Config, registry: jobs.Registry, key: str, run: dict) -> None:
    device, _, name = run["variant"].split("/", 2)
    res = {"mode": "gpu", "n_gpus": N_GPUS, "gpu_hosts": [GPU_HOST], "cpu_workers": config.gpu_job_cpu_workers,
           "launch_host": GPU_HOST}
    argv = jobs.command(config, jobs.FLOW_MODULES[run.get("flow", "stability")], device, name, run["steps"], GHOST, res)
    job = jobs.launch(config, registry, run["variant"], argv, res,
                      f"watchdog: full analysis ({variants.model_name(SOURCE, APERTURE)}, lambda 650-750 nm)", GHOST,
                      rundir=catalog.resolve(config, run["variant"]))
    run.update(state="launched", job=job["id"], launched=job["started"])
    log(f"{run['variant']}: analysis job {job['id']} on the P100s: {' '.join(argv)}")


def step(config: Config, registry: jobs.Registry, state: dict, poll_rig: bool) -> None:
    runs = state["runs"]
    if poll_rig:
        remote = catalog.RemoteRuns(config).list_remote()
        for key, info in sorted(remote.items()):
            if (info["finished"] and key not in state["baseline"] and key not in runs
                    and MATCH.search(key.rsplit("/", 1)[-1])):
                runs[key] = {"state": "new"}
    for key, run in sorted(runs.items()):
        if run["state"] == "new":
            _fetch(config, registry, key, run)
        elif run["state"] == "fetching":
            job = registry.get(run["fetch_job"])
            if job["state"] == "done":
                _prepare(config, key, run)
            elif job["state"] not in ("running", "pending"):
                retry = run["tries"] < FETCH_TRIES
                run.update(state="new" if retry else "failed", note=f"fetch job {job['id']} {job['state']}")
                log(f"{key}: fetch {job['state']}{', retrying' if retry else ', giving up'}")
    queued = sorted(k for k, r in runs.items() if r["state"] == "queued")
    if queued:
        busy = p100s_busy(config, registry)
        if busy is None:
            try:
                _launch(config, registry, queued[0], runs[queued[0]])
            except jobs.Conflict as e:
                busy = f"{runs[queued[0]]['variant']} busy: {e}"
        if busy and busy != state.get("waiting"):
            log(f"waiting ({len(queued)} queued): {busy}")
        state["waiting"] = busy


def run(config: Config) -> None:
    """The loop: the rig every RIG_POLL_S, fetches, queue and P100s every TICK_S, until the `enabled` file goes."""
    os.environ.setdefault("USER", getpass.getuser())    # cron starts us without it (squeue -u $USER)
    lock = open(folder(config) / "watch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another watchdog holds the lock: exiting")
        return
    registry = jobs.Registry(config.jobs_dir)
    log(f"watchdog up (pid {os.getpid()}): {SOURCE}@{APERTURE}, ghost {GHOST}, {FIT}, {N_GPUS} GPUs of {GPU_HOST}")
    last_rig = 0.0
    while (folder(config) / "enabled").exists():
        state = load_state(config)
        poll_rig = time.time() - last_rig >= RIG_POLL_S
        try:
            step(config, registry, state, poll_rig)
            if poll_rig:
                last_rig = time.time()
        except Exception as e:                          # a failed SSH or launch is retried at the next tick
            log(f"error: {type(e).__name__}: {e}")
        save_state(config, state)
        time.sleep(TICK_S)
    log("enabled file removed: exiting")
