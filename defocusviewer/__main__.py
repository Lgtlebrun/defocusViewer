"""defocusviewer serve | url | ensure | fetch <device> <rundir> [--analyse] [--force] | move-data [runs] [--all] [--dry-run] | jointfit <run key> --fnum F [...] | watch start|stop|status|run."""
import argparse
import fcntl
import getpass
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import config as settings

TMUX_SESSION = "defocusviewer"


def serve(config):
    import uvicorn
    from .server import create_app
    os.environ.setdefault("USER", getpass.getuser())     # cron starts us without it (squeue -u $USER)
    uvicorn.run(create_app(config, settings.token()), host=config.host, port=config.port, log_level="warning",
                access_log=False)


def url(config) -> str:
    return f"http://{socket.getfqdn()}:{config.port}/login?token={settings.token()}"


def answering(config) -> bool:
    """True when a viewer answers on the port (a 401 counts: it is ours, logged out)."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{config.port}/api/config", timeout=10)
    except urllib.error.HTTPError as e:
        return e.code == 401
    except OSError:
        return False
    return True


def ensure(config):
    """Starts the server in tmux session `defocusviewer` unless one answers (for cron, every few minutes), and the
    rig watchdog when it is enabled and down."""
    config.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        from . import watch
        watch.ensure(config)
    except Exception as e:          # the viewer must come back even when the watchdog cannot
        print(f"watchdog not restarted: {e}", file=sys.stderr)
    try:
        from . import rawdata
        rawdata.ensure_sweep(config)    # a data/ copied into a rundir outside the viewer goes to the raw root
    except Exception as e:
        print(f"raw data sweep not started: {e}", file=sys.stderr)
    with open(config.state_dir / "ensure.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        running = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION], capture_output=True).returncode == 0
        for _ in range(30 if running else 1):     # a session just started needs a few seconds to answer
            if answering(config):
                return
            time.sleep(2)
        subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
        log = config.state_dir / "server.log"
        command = f"{sys.executable} -m defocusviewer serve >> {log} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-c", str(config.state_dir), "-s", TMUX_SESSION, command], check=True)
        print(f"started the viewer in tmux session {TMUX_SESSION}: {url(config)}")


def fetch(config, device: str, name: str, analyse: bool, ghost: str, force: bool = False):
    """Copies a rundir from pos-control (rsync, without waiting for meta/_finished; `force`: also when a local copy
    exists, rsync without deletes), then optionally starts its default analysis with the default resources."""
    os.environ.setdefault("POS_LOCAL_DATA", str(config.data_root))
    from . import cluster, jobs, rawdata
    rawdata.fetch(config, device, name, force=force)       # data/ onto the raw root, the rest into the rundir
    print(f"fetched {device}/{config.test}/{name}", flush=True)
    if not analyse:
        return
    path = config.data_root / device / config.test / name
    info = jobs.analyses(path, ghost)
    if info["reason"]:
        print(f"no analysis started: {info['reason']}")
        return
    free = {h: cluster.host_state(h).get("free_gpus") or [] for h in config.job_gpu_hosts}
    cores = {h: cluster.cpu_host_state(h, config.cpu_margin).get("free_cores", 0) for h in config.cpu_hosts}
    res = jobs.default_resources(config, free, bool(set(info["default_steps"]) & jobs.GPU_STEPS), cores)
    argv = jobs.command(config, jobs.FLOW_MODULES[info["flow"]], device, name, info["default_steps"], ghost, res)
    # force: the run was not local before this fetch, so the only job on it is this one (which the guard would see)
    job = jobs.launch(config, jobs.Registry(config.jobs_dir), f"{device}/{config.test}/{name}", argv, res,
                      " ".join(info["default_steps"]), ghost, force=True, rundir=path)
    print(f"started analysis job {job['id']} ({res['mode']}): {' '.join(argv)}")


def move_data(config, runs: list[str], every: bool, dry_run: bool):
    """Moves the data/ of rundirs onto the raw root (rawdata.move), one at a time; one move-data at a time."""
    from . import rawdata
    os.environ.setdefault("USER", getpass.getuser())     # cron starts us without it (squeue -u $USER)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    with open(config.state_dir / "rawdata.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(rawdata.log_line("another move-data runs: nothing done"), flush=True)
            return
        if every:
            rundirs = rawdata.real_data_dirs(config, rawdata.SWEEP_MIN_AGE_S)
        else:
            rundirs = [config.data_root / r if "/" in r else next(config.data_root.glob(f"*/{config.test}/{r}"), None)
                       for r in runs]
            if missing := [r for r, path in zip(runs, rundirs) if path is None or not path.is_dir()]:
                raise SystemExit(f"no such rundir under {config.data_root}: {', '.join(missing)}")
        for rundir in rundirs:
            key = rundir.relative_to(config.data_root)
            try:
                result = rawdata.move(config, rundir, dry_run=dry_run)
            except Exception as e:      # one failed run must not stop the others
                result = f"FAILED: {type(e).__name__}: {e}"
            print(rawdata.log_line(f"{key}: {result}"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="defocusviewer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the server in the foreground")
    sub.add_parser("url", help="print the login URL")
    sub.add_parser("ensure", help="start the server in tmux unless it answers")
    f = sub.add_parser("fetch", help="copy a rundir from pos-control, optionally analyse it")
    f.add_argument("device", help="device folder, e.g. Orbray6, MPS6, calibration_stand")
    f.add_argument("rundir")
    f.add_argument("--analyse", action="store_true")
    f.add_argument("--ghost", choices=("on", "off"), default="on")
    f.add_argument("--force", action="store_true", help="resync a local copy (rsync, no deletes)")
    m = sub.add_parser("move-data", help="move rundirs' data/ onto the raw root, linked from the rundir (rawdata.py)")
    m.add_argument("runs", nargs="*", help="rundir names or keys (device/test/name)")
    m.add_argument("--all", action="store_true", help="every rundir whose data/ is a real directory")
    m.add_argument("--dry-run", action="store_true")
    w = sub.add_parser("watch", help="rig watchdog: fetch + tophat_bl@circle analysis of new runs (watch.py)")
    w.add_argument("action", choices=("start", "stop", "status", "run"))
    j = sub.add_parser("jointfit", help="joint fit of a run the pipeline cannot analyse, into its results/ (jointfit.py)")
    j.add_argument("run", help="run key, e.g. Orbray6/Defocus/20261015-155646_hexagonal-aperture-fit")
    j.add_argument("--mode", choices=("seeded", "scratch"), default="seeded")
    j.add_argument("--reference", help="run key of the reference (seeded mode)")
    j.add_argument("--fnum", type=float, required=True)
    j.add_argument("--ghost", choices=("on", "off"), default="on")
    j.add_argument("--fit-theta", action="store_true", help="search theta around the reference's instead of holding it")
    j.add_argument("--gpu", type=int, help="number of GPUs (fit_joint device gpu)")
    j.add_argument("--gpu-hosts", help="hosts of the GPUs, in order")
    j.add_argument("--gpu-precision", choices=("float64", "float32", "mixed"), help="of the GPU renders and fits (default float64)")
    j.add_argument("--n-workers", type=int, default=1, help="CPU workers without --gpu")
    j.add_argument("--pupil-shape", choices=("hexagon", "circle"), default="hexagon", help="aperture of the model")
    j.add_argument("--source", help="source model (default: the reference's)")
    args = parser.parse_args(argv)
    config = settings.load()
    if args.command == "serve":
        serve(config)
    elif args.command == "url":
        print(url(config))
    elif args.command == "ensure":
        ensure(config)
    elif args.command == "jointfit":
        from DefocusTest import psf_batch
        from . import jointfit
        if args.gpu_precision:
            psf_batch.PRECISION = args.gpu_precision
        jointfit.run(config, args.run, args.mode, args.reference, args.fnum, args.ghost, args.fit_theta,
                     device="gpu" if args.gpu else "cpu", n_workers=args.gpu or args.n_workers,
                     gpu_hosts=args.gpu_hosts.split(",") if args.gpu_hosts else None, pupil_shape=args.pupil_shape,
                     source=args.source)
    elif args.command == "watch":
        from . import watch
        getattr(watch, args.action)(config)
    elif args.command == "move-data":
        if not args.all and not args.runs:
            parser.error("move-data needs rundirs or --all")
        move_data(config, args.runs, args.all, args.dry_run)
    else:
        fetch(config, args.device, args.rundir, args.analyse, args.ghost, args.force)


if __name__ == "__main__":
    main()
