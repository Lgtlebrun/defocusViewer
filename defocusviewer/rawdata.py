"""Raw data/ of the rundirs on `config.raw_root` (/disk/data1, farm-se1 NFS), the rest of each rundir on `data_root`.

A rundir's data/ (the rig's image stacks and acquisition tables, 90 % of its bytes and identical to the rig's copy) is a
symlink to <raw_root>/<device>/<test>/<run>/data; results/, plot/, meta/ and the logs stay in the rundir. posUtils,
DefocusTest and the viewer read through the link (variants already did).

posUtils is not changed: its fetch (ensureRunDir, `rsync -a` without -K) copies data/ into the rundir for a new run and
replaces the link by a directory on a forced resync. So the viewer fetches in two rsyncs itself (`fetch`), and `move`
puts a data/ that is a real directory again onto the raw root (the `ensure` cron sweeps for those).
"""
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from amkit import resources

from .config import Config

SWEEP_MIN_AGE_S = 600       # a real data/ younger than this may still be filling (posUtils rsync): left for later


def raw_dir(config: Config, rundir: Path) -> Path:
    """Where the data/ of `rundir` (a folder under data_root) lives on the raw root."""
    return config.raw_root / rundir.relative_to(config.data_root) / "data"


def _rsync(source: str, dest: Path, *options: str, ssh: list[str] | None = None) -> None:
    argv = ["rsync", "-a", "--partial", *options]
    if ssh is not None:
        argv += ["-e", " ".join(["ssh", *ssh])]
    result = subprocess.run([*argv, source, f"{dest}/"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rsync {source} -> {dest} failed ({result.returncode}): {result.stderr.strip()[-500:]}")


def _remote(config: Config, device: str):
    """posUtils' handle on the rig folder of `device` (only read: target, remote folder, ssh options, local folder)."""
    os.environ.setdefault("POS_LOCAL_DATA", str(config.data_root))
    from posUtils.data.dir import RemoteCopyDataDir
    from . import catalog
    return RemoteCopyDataDir.getDir(catalog.device_id(device), config.test, print=False)


def _source(remote, name: str) -> str:
    path = shlex.quote(str(remote._remote_folder / name))
    return f"{remote._target}:{path}" if remote._target else path


def fetch(config: Config, device: str, name: str, force: bool = False) -> Path:
    """Copies run `name` of `device` from the rig now (no wait for meta/_finished), data/ onto the raw root and the rest
    into the rundir; a local rundir is left as it is unless `force` (then resynced, no deletes). data/ goes first, so
    meta/_finished arrives last. Returns the rundir."""
    remote = _remote(config, device)
    local = config.data_root / device / config.test / name
    if local.is_dir() and not force:
        return local
    if not remote._remote_test(remote._remote_folder / name, "d"):
        raise FileNotFoundError(f"run-dir {name!r} not found on {remote._target}:{remote._remote_folder}")
    target = raw_dir(config, local)
    target.mkdir(parents=True, exist_ok=True)
    source = _source(remote, name)
    _rsync(f"{source}/data/", target, ssh=remote._SSH_OPTS)
    if local.is_dir():
        _rsync(f"{source}/", local, "-K", "--exclude=/data/", ssh=remote._SSH_OPTS)
        data = local / "data"
        if data.is_dir() and not data.is_symlink():  # a copy made by posUtils (maybe partial): the rig's is on target now
            if users := busy(name, in_use(config, ignore_fetch=True)):
                print(f"{device}/{config.test}/{name}: data/ left in the rundir, in use by {users[0]}", flush=True)
            else:
                _switch(local, target)
        elif not data.exists():
            data.symlink_to(target)
        return local
    partial = local.parent / f".{name}.partial"      # as posUtils: an interrupted copy is not taken for a fetched run
    local.parent.mkdir(parents=True, exist_ok=True)
    _rsync(f"{source}/", partial, "-K", "--exclude=/data/", ssh=remote._SSH_OPTS)
    link = partial / "data"
    if link.is_symlink():
        link.unlink()
    elif link.is_dir():                              # left by an interrupted posUtils fetch: the rig has it all
        shutil.rmtree(link)
    link.symlink_to(target)
    partial.rename(local)
    return local


def in_use(config: Config, ignore_fetch: bool = False) -> list[str]:
    """Names of the rundirs something works on now: viewer jobs (from disk), analysis processes on the launch and CPU
    hosts (every argument, by its last path component), our Slurm jobs. `ignore_fetch`: not the fetches."""
    from . import jobs
    names = [j["run"].rsplit("/", 1)[-1] for j in jobs.Registry(config.jobs_dir).all()
             if j["state"] in ("running", "pending") and not (ignore_fetch and j["label"].startswith("fetch"))]
    for host in dict.fromkeys([*config.launch_hosts, *getattr(config, "cpu_hosts", [])]):
        out = resources._shell("ps -eo args=", None if resources.is_local(host) else host) or ""
        for line in out.splitlines():
            if "move-data" in line or (ignore_fetch and "defocusviewer fetch" in line):
                continue
            if "DefocusTest" in line or "defocusviewer" in line:
                names += [Path(word).name for word in line.split()]
    names += [job["name"] for job in resources.slurm_jobs().values()]
    return names


def busy(name: str, names: list[str]) -> list[str]:
    """The entries of `names` that are run `name` or one of its variants/copies (`<name>_<suffix>`, they read its data/)."""
    return [n for n in names if n == name or n.startswith(f"{name}_")]


def _differences(source: Path, target: Path) -> list[str]:
    """Files of `source` missing in `target` or different there (size, mtime): `rsync --dry-run --itemize-changes`."""
    result = subprocess.run(["rsync", "-a", "--dry-run", "--itemize-changes", f"{source}/", f"{target}/"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return [f"rsync --dry-run failed ({result.returncode}): {result.stderr.strip()[-300:]}"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _count(folder: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(folder))


def move(config: Config, rundir: Path, names: list[str] | None = None, dry_run: bool = False) -> str:
    """Moves the data/ of `rundir` onto the raw root when it is a real directory: rsync, check that every file arrived
    unchanged, then data/ becomes the symlink and the copy in the rundir is removed. Skipped while something works on the
    run or one of its variants (`names`, see in_use; None: asked now). Returns what happened, for the log."""
    data = rundir / "data"
    if data.is_symlink() or not data.is_dir():
        return "data/ already a link" if data.is_symlink() else "no data/"
    names = in_use(config) if names is None else names
    if users := busy(rundir.name, names):
        return f"skipped, in use by {users[0]}"
    target = raw_dir(config, rundir)
    n = _count(data)
    if dry_run:
        return f"would move {n} files to {target}"
    t0 = time.monotonic()
    target.mkdir(parents=True, exist_ok=True)
    _rsync(f"{data}/", target)
    if diff := _differences(data, target):
        return f"FAILED, left as it was: {len(diff)} file(s) differ on {target}, e.g. {diff[0]}"
    if (m := _count(target)) < n:
        return f"FAILED, left as it was: {m} files on {target} for {n} in data/"
    if users := busy(rundir.name, in_use(config)):        # a job may have started during the copy
        return f"copied, not switched: in use by {users[0]} (the next sweep switches it)"
    _switch(rundir, target)
    return f"moved {n} files to {target} in {time.monotonic() - t0:.0f} s"


def _switch(rundir: Path, target: Path) -> None:
    """data/ of `rundir` becomes the link to `target`; the directory it was is removed (after the link exists)."""
    data, old = rundir / "data", rundir / ".data.moved"
    if old.exists():
        shutil.rmtree(old)
    data.rename(old)
    data.symlink_to(target)
    shutil.rmtree(old)


def real_data_dirs(config: Config, min_age_s: float = 0) -> list[Path]:
    """Rundirs under data_root whose data/ is a real directory not modified for `min_age_s` seconds."""
    now, found = time.time(), []
    for data in sorted(config.data_root.glob(f"*/{config.test}/*/data")):
        if data.parent.name.startswith(".") or data.is_symlink() or not data.is_dir():
            continue
        if now - data.stat().st_mtime >= min_age_s:
            found.append(data.parent)
    return found


def log_line(text: str) -> str:
    return f"({datetime.now():%Y-%m-%d %H:%M:%S}) {text}"


def ensure_sweep(config: Config) -> bool:
    """For the `ensure` cron: starts `move-data --all` detached (nice, idle I/O) when a rundir has a real data/ again
    and no move-data runs. True when one was started."""
    import fcntl
    import sys
    if not real_data_dirs(config, SWEEP_MIN_AGE_S):
        return False
    config.state_dir.mkdir(parents=True, exist_ok=True)
    with open(config.state_dir / "rawdata.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(config.cache_dir / "rawdata.log", "a") as out:
        subprocess.Popen(["nice", "-n", "10", "ionice", "-c", "3", sys.executable, "-m", "defocusviewer", "move-data",
                          "--all"], stdout=out, stderr=subprocess.STDOUT, cwd=config.state_dir, start_new_session=True)
    return True
