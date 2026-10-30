"""The lab rig (pos-control): launch acquisition routines there, follow the runs, the stage and the mount.

The viewer never opens rig hardware: everything comes from one read-only probe over SSH every 15 s (processes, which
process holds which device, the runs' log.log / meta/status.json / data file names; HDF5 files are never opened, a reader
could make the writer's append fail). Stopping a run writes <rundir>/meta/_stop_requested, which the routines check
between stacks (posUtils Operations.stop_requested); a run without that hook (no meta/status.json) can only be killed.
"""
import copy
import json
import re
import secrets
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath

import yaml

from amkit import resources

from . import jobs
from .config import Config

PERIOD_S = 15
STOP_FILE = "_stop_requested"
STALLED_S = 600             # no new log line for this long while the process lives: shown as stalled
WATCH_JOBS = 10             # recent lab jobs whose runs stay followed after they end
LOG_LINE = re.compile(r"^\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\) \[(\w+)\] \[([^\]]*)\] : (.*)$")
SAFE_NAME = re.compile(r"^\d{8}-\d{6}_[\w.-]+$")


class Field:
    """One setting of a routine's config the launch form may change."""

    def __init__(self, key: str, kind: str, label: str, lo=None, hi=None, choices=None, pattern=None):
        self.key, self.kind, self.label, self.lo, self.hi, self.choices, self.pattern = key, kind, label, lo, hi, choices, pattern

    def describe(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}

    def check(self, value):
        """`value` converted to the field's type; ValueError outside the allowed values."""
        bad = ValueError(f"{self.key}: {value!r} is not allowed ({self.label})")
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise bad
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise bad
            return value
        if self.kind == "str":
            if not isinstance(value, str) or not re.fullmatch(self.pattern, value):
                raise bad
            return value
        if self.kind == "int_list":
            if not isinstance(value, list) or not 1 <= len(value) <= 500:
                raise bad
            return [self._number(v, int, bad) for v in value]
        return self._number(value, int if self.kind == "int" else float, bad)

    def _number(self, value, cast, bad):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (cast is int and value != int(value)):
            raise bad
        value = cast(value)
        if not self.lo <= value <= self.hi:
            raise bad
        return value


STAGE_UM = (0, 50000)
STABILITY_FIELDS = [
    Field("run.device", "choice", "device: data folder and fiducial table", choices=["Orbray6", "MPS6"]),
    Field("run.serial_camera", "choice", "camera: 25193621 mount setup, 25430388 bench setup", choices=[25193621, 25430388]),
    Field("run.pwm_arrays", "int_list", "PWM LED arrays in use", lo=1, hi=9),
    Field("output.path", "str", "run name (letters, digits, _ -)", pattern=r"[A-Za-z0-9_-]{1,80}"),
    Field("output.debug", "bool", "debug lines in log.log"),
    Field("equalizer.run", "bool", "equalise the LEDs first"),
    Field("equalizer.target_intensity", "int", "counts of the brightest spot", lo=100, hi=4095),
    Field("temperature.run", "bool", "temperature acquisition (Arduino, 1 Hz)"),
    Field("train.camera.burn_in_s", "float", "burn-in before the coarse sweep (s)", lo=0, hi=86400),
    Field("train.stage.x", "int_list", "coarse sweep positions (µm)", *STAGE_UM),
    Field("stability.duration_s", "float", "hold duration (s)", lo=0, hi=14 * 86400),
    Field("stability.stage.hold_position", "int", "hold position (µm) = last sweep position", *STAGE_UM),
    Field("stability.camera.save_raw_frames", "bool", "save every frame of the hold (~21 MB per stack)"),
    Field("train_end.run", "bool", "repeat the sweep after the hold"),
    Field("all.camera.n_pics", "int", "frames per stack", lo=1, hi=5000),
    Field("all.camera.texp", "int", "initial exposure (µs)", lo=10, hi=10_000_000),
    Field("all.camera.fibers.n_spots", "int", "spots to find", lo=1, hi=50),
    Field("fit.run", "bool", "joint fit on the rig at the end (analysis normally runs on the cluster)"),
    Field("fit.fnum", "float", "f-number", lo=0.5, hi=32),
    Field("calibration.run", "bool", "fiducial calibration on the rig at the end (needs fit.run)"),
]
ROUTINES = {
    "StabilityTest_new": {"module": "DefocusTest.sequences.StabilityTest_new",
                          "config": "DefocusTest/DefocusTest/config/config_stabilitytest_new.yaml",
                          "fields": STABILITY_FIELDS},
}


class _RunConfigLoader(yaml.SafeLoader):
    """Reads the meta/config.yaml a run saves: ruamel scalars come as !!python/object/new tags, read as their value."""


def _python_object(loader, suffix, node):
    value = loader.construct_mapping(node, deep=True) if isinstance(node, yaml.MappingNode) else None
    args = (value or {}).get("args") or [None]
    return args[0]


_RunConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/object", _python_object)


def get_key(config: dict, key: str):
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_key(config: dict, key: str, value) -> None:
    *parents, last = key.split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value


def merged_config(routine: str, base: dict, overrides: dict) -> tuple[dict, list[dict]]:
    """The routine's config: `base` with the checked `overrides`, and the list of changes; ValueError on a key the form
    does not offer, a value outside its field, or a hold position that is not the sweep's last."""
    fields = {f.key: f for f in ROUTINES[routine]["fields"]}
    config, changes = copy.deepcopy(base), []
    for key, value in overrides.items():
        if key not in fields:
            raise ValueError(f"{key} cannot be set from the viewer")
        value = fields[key].check(value)
        old = get_key(config, key)
        if old != value:
            set_key(config, key, value)
            changes.append({"key": key, "old": old, "new": value})
    if routine == "StabilityTest_new":
        x, hold = get_key(config, "train.stage.x"), get_key(config, "stability.stage.hold_position")
        last = max(x) if x else (get_key(config, "train.stage.limits") or [None, None])[1]
        if get_key(config, "stability.run") and last is not None and hold != last:
            raise ValueError(f"stability.stage.hold_position ({hold}) must be the last sweep position ({last})")
    return config, changes


def ssh(config: Config, command: str, stdin: str | None = None, timeout: float = 60) -> str:
    """stdout of `command` on the rig; RuntimeError when SSH or the command fails."""
    argv = [*resources.SSH, config.lab_host, command]
    try:
        out = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{config.lab_host}: no answer in {timeout:.0f} s") from None
    if out.returncode != 0:
        raise RuntimeError(f"{config.lab_host}: {(out.stderr or out.stdout).strip()[-300:] or f'exit {out.returncode}'}")
    return out.stdout


def base_config(config: Config, routine: str) -> dict:
    return yaml.safe_load(ssh(config, f"cat {shlex.quote(f'{config.lab_root}/{ROUTINES[routine]['config']}')}"))


# Runs on the rig with its system python3, read-only. Prints one JSON document.
PROBE = r'''
import glob, json, os, re, time
A = json.loads(ARGS)
now = time.time()
hz = os.sysconf("SC_CLK_TCK")
btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
devices = dict(A["devices"])
for v in glob.glob("/sys/bus/usb/devices/*/idVendor"):
    d = os.path.dirname(v)
    try:
        if open(v).read().strip() == "2676":
            serial = open(f"{d}/serial").read().strip() if os.path.exists(f"{d}/serial") else "?"
            bus, dev = int(open(f"{d}/busnum").read()), int(open(f"{d}/devnum").read())
            devices[f"camera {serial}"] = f"/dev/bus/usb/{bus:03d}/{dev:03d}"
    except (OSError, ValueError):
        pass
wanted = {os.path.realpath(p) for p in devices.values()}
fds, cmds = {}, {}
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        links = []
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                links.append(os.readlink(f"/proc/{pid}/fd/{fd}"))
            except OSError:
                pass
    except OSError:
        continue
    cmds[int(pid)] = cmd
    for t in links:
        if t in wanted or t.endswith("/log.log"):
            fds.setdefault(t, set()).add(int(pid))

def proc(pid):
    try:
        start = btime + int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        start, cwd = None, None
    return {"pid": pid, "args": cmds.get(pid, "")[:400], "started": start, "cwd": cwd}

dev_out = {}
for name, path in devices.items():
    real = os.path.realpath(path)
    dev_out[name] = {"path": path, "real": real, "present": os.path.exists(path), "holders": sorted(fds.get(real, ()))}

watch = set(A["rundirs"])
for log in A["job_logs"]:
    try:
        with open(log, errors="replace") as f:
            head = f.read(65536)
        m = re.search(r"Results will be saved under (\S+)", head)
        if m:
            watch.add(m.group(1))
    except OSError:
        pass
data = A["data_root"].rstrip("/") + "/"
writers = {}
for t, pids in fds.items():
    if t.endswith("/log.log") and t.startswith(data):
        rundir = os.path.dirname(t)
        watch.add(rundir)
        writers[rundir] = sorted(pids)

MARK = re.compile(r"\[INFO\] \[[^\]]*\] : (Starting \S+\.\.\.|\S+ completed in |Burn-in|Estimated end|Hold at pos|Holding in place|Results will be saved|Sweep x_positions|Mount[ :])|\[Stage\] : Arrived to position|Stop requested by user|\[ERROR\]| failed !")
MOUNT = re.compile(r"metadata for key (alt|az): (\S+)")
NPY = re.compile(r"^(?P<fiber>[^_]+)_(?P<pos>-?\d+)_(?P<routine>.+?)(?:_(?P<idx>\d+))?\.npy$")
runs = []
for rundir in sorted(watch):
    if not rundir.startswith(data) or ".." in rundir or not os.path.isdir(rundir):
        continue
    info = {"rundir": rundir, "writers": writers.get(rundir, []), "marks": [], "last_line": None, "mount": {}}
    try:
        with open(f"{rundir}/log.log", errors="replace") as f:
            marked = False
            for line in f:
                if line.startswith("("):
                    info["last_line"] = line[:21]
                    marked = bool(MARK.search(line))
                    if marked:
                        info["marks"].append(line.rstrip()[:600])
                    elif "metadata for key a" in line:
                        m = MOUNT.search(line)
                        if m:
                            info["mount"][m.group(1)] = [line[1:20], m.group(2)]
                elif marked and line.startswith(">>") and len(info["marks"][-1]) < 4000:
                    info["marks"][-1] += " " + line[2:].strip()
        info["marks"] = info["marks"][:60] + info["marks"][60:][-400:]
    except OSError:
        pass
    for flag in ("_finished", "_stop_requested"):
        info[flag.strip("_")] = os.path.exists(f"{rundir}/meta/{flag}")
    try:
        info["status"] = json.load(open(f"{rundir}/meta/status.json"))
    except (OSError, ValueError):
        info["status"] = None
    try:
        info["config"] = open(f"{rundir}/meta/config.yaml").read(200000)
    except OSError:
        info["config"] = None
    stacks = {}
    try:
        for e in os.scandir(f"{rundir}/data"):
            m = NPY.match(e.name)
            if m:
                stacks.setdefault(m["routine"], set()).add((m["pos"], m["idx"]))
    except OSError:
        pass
    info["stacks"] = {r: {"stacks": len(s), "positions": len({p for p, _ in s})} for r, s in stacks.items()}
    runs.append(info)

interesting = {p for d in dev_out.values() for p in d["holders"]}
interesting |= {p for p, c in cmds.items() if re.search(A["process_pattern"], c)}
print(json.dumps({"now": now, "tz_offset": time.localtime().tm_gmtoff, "devices": dev_out,
                  "processes": [proc(p) for p in sorted(interesting)], "runs": runs}))
'''


def probe(config: Config, rundirs: list[str], job_logs: list[str]) -> dict:
    args = {"devices": config.lab_devices, "data_root": config.lab_data, "rundirs": rundirs, "job_logs": job_logs,
            "process_pattern": r"DefocusTest\.sequences\.|routines\.schedules\.jobqueue|LAB-Positioner-Daemon"}
    script = f"ARGS = {json.dumps(json.dumps(args))}\n{PROBE}"
    return json.loads(ssh(config, "python3 -", stdin=script, timeout=90))


def _epoch(stamp: str, tz_offset: int) -> float:
    """Epoch of a rig time stamp (the rig's local time, UTC offset `tz_offset` s)."""
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).timestamp() - tz_offset


def parse_run(info: dict, now: float, tz_offset: int) -> dict:
    """What the probe saw of one rundir, read: phases of the Main sequence, the current one and its progress, the stage
    position, errors, and the outcome once it ended."""
    path = PurePosixPath(info["rundir"])
    out = {"rundir": str(path), "name": path.name, "device": path.parent.parent.name, "test": path.parent.name,
           "key": f"{path.parent.parent.name}/{path.parent.name}/{path.name}", "writers": info["writers"],
           "finished": info["finished"], "stop_requested": info["stop_requested"], "hook": info["status"] is not None,
           "status": info["status"], "stacks": info["stacks"], "errors": [], "phases": [], "stage_trace": []}
    try:
        cfg = yaml.load(info["config"], Loader=_RunConfigLoader) if info["config"] else {}
    except yaml.YAMLError:
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}
    top, phases, current = None, {}, None
    hold, burn, sweeps, stopped = {}, {}, {}, False
    mount = {"alt_deg": None, "az_deg": None, "t": None, "target": None, "state": None, "since": None}
    for line in info["marks"]:
        m = LOG_LINE.match(line)
        if not m:
            continue
        stamp, level, category, msg = m.groups()
        t = _epoch(stamp, tz_offset)
        name = category.split(":")[-1]
        if level == "ERROR" or " failed !" in msg:
            out["errors"].append(line[:400])
        if msg.startswith("Starting ") and msg.endswith("..."):
            op = msg[len("Starting "):-3]
            if top is None:
                top = {"name": op, "start": t, "state": "running"}
            elif category.startswith("Main:") and category.count(":") == 1 and op == name:
                phases[op] = {"name": op, "start": t, "end": None, "state": "running"}
                current = op
        elif re.match(r"\S+ completed in ", msg):
            op = msg.split(" ", 1)[0]
            target = phases.get(op) or (top if top and top["name"] == op else None)
            if target:
                target.update(end=t, state="done")
        elif msg.endswith(" failed !"):
            op = msg[:-len(" failed !")]
            target = phases.get(op) or (top if top and top["name"] == op else None)
            if target:
                target.update(end=t, state="failed")
        elif msg.startswith("Burn-in: acquiring real data for"):
            s = re.search(r"for ([\d.]+) s", msg)
            burn[name] = {"start": t, "duration": float(s.group(1)) if s else None, "done": False}
        elif msg.startswith("Burn-in complete") and name in burn:
            burn[name]["done"] = True
        elif msg.startswith("Hold at pos="):
            s = re.search(r"duration=([\d.]+)", msg)
            hold[name] = {"start": t, "duration": float(s.group(1)) if s else None}
        elif msg.startswith("Sweep x_positions"):
            sweeps[name] = len(re.findall(r"-?\d+", msg.split(":", 1)[1]))
        elif msg.startswith("Arrived to position"):
            out["stage_trace"].append([t, float(msg.split()[-1])])
        elif msg.startswith("Holding in place: stage at"):
            out["stage_trace"].append([t, float(re.search(r"stage at (-?[\d.]+)", msg).group(1))])
        elif msg.startswith("Mount"):
            angles = re.search(r"alt (-?[\d.]+) deg, az (-?[\d.]+) deg", msg)
            if msg.startswith("Mount at") and angles:
                mount.update(alt_deg=float(angles.group(1)), az_deg=float(angles.group(2)), t=t)
            elif angles:
                mount["target"] = [float(angles.group(1)), float(angles.group(2))]
            state = {"Mount: searching home": "homing", "Mount: moving home": "going home", "Mount home": "home"}.get(msg)
            state = state or ("slewing" if msg.startswith("Mount: slewing") else "at target" if msg.startswith("Mount at") else None)
            if state:
                mount.update(state=state, since=t)
        if "Stop requested by user" in msg:
            stopped = True
    # the routine logs the mount's alt / az with every saved stack: newer than its "Mount at" line
    alt, az = info.get("mount", {}).get("alt"), info.get("mount", {}).get("az")
    if alt and az:
        try:
            t = _epoch(min(alt[0], az[0]), tz_offset)
            if mount["t"] is None or t >= mount["t"]:
                mount.update(alt_deg=float(alt[1]), az_deg=float(az[1]), t=t)
        except ValueError:
            pass
    out["mount"] = mount if mount["state"] or mount["t"] else None
    status = info["status"] or {}
    if status.get("x_readback_um") is not None and status.get("t"):
        out["stage_trace"].append([_epoch(status["t"], tz_offset), status["x_readback_um"]])
    out["stage_trace"] = sorted(out["stage_trace"])[-400:]
    out["stage"] = ({"x_um": out["stage_trace"][-1][1], "t": out["stage_trace"][-1][0]} if out["stage_trace"] else None)
    out["phases"] = list(phases.values())
    if current and phases[current]["state"] == "running":
        out["current"] = {"name": current, **_progress(current, phases[current], burn, hold, sweeps, info["stacks"], status, now)}
    else:
        out["current"] = None
    out["last_line"] = _epoch(info["last_line"][1:20], tz_offset) if info["last_line"] and info["last_line"][1:20].strip() else None
    alive = bool(info["writers"])
    out["stalled"] = bool(alive and out["last_line"] and now - out["last_line"] > STALLED_S)
    top_state = top["state"] if top else None
    out["state"] = ("running" if alive else "stopped" if stopped and top_state == "done" else top_state or "unknown")
    if not alive and out["state"] == "running":
        out["state"] = "died"           # no writer, no final line: killed or crashed hard
    out["started"] = top["start"] if top else None
    out["config"] = {k: get_key(cfg, k) for k in ("run.device", "output.path", "stability.duration_s",
                                                    "train.camera.burn_in_s", "train_end.run", "fit.run")}
    return out


def _progress(name, phase, burn, hold, sweeps, stacks, status, now) -> dict:
    """Progress of the running phase: burn-in and hold by time, a sweep by the positions saved so far."""
    stacks = stacks.get(name) or stacks.get(name.replace("_", "-")) or {}     # file names have the slug of the routine
    b = burn.get(name)
    if b and not b["done"] and b["duration"]:
        return {"kind": "burn-in", "fraction": min(1, (now - b["start"]) / b["duration"]), "end": b["start"] + b["duration"]}
    h = hold.get(name)
    if h and h["duration"]:
        return {"kind": "hold", "fraction": min(1, (now - h["start"]) / h["duration"]), "end": h["start"] + h["duration"],
                "stacks": stacks.get("stacks", 0)}
    if name in sweeps:
        done = stacks.get("positions", 0)
        if status.get("routine") == name and status.get("kind") == "sweep":
            done = status["stop"]
        return {"kind": "sweep", "done": done, "total": sweeps[name], "fraction": done / sweeps[name] if sweeps[name] else None}
    return {"kind": "step", "fraction": None}


def guard(snapshot: dict, services: str = "") -> list[str]:
    """Why a launch must wait: a routine runs, or a process holds the stage, the camera or an LED array (processes
    matching `services`, long-running monitoring that keeps a port open, are not counted)."""
    procs = {p["pid"]: p["args"] for p in snapshot.get("processes", [])}
    busy = [f"routine pid {pid}: {args[:160]}" for pid, args in procs.items() if "DefocusTest.sequences." in args]
    for name, dev in snapshot.get("devices", {}).items():
        holders = [pid for pid in dev["holders"] if not (services and re.search(services, procs.get(pid, "")))]
        if (name == "stage" or name.startswith("camera") or name.startswith("PWM")) and holders:
            busy.append(f"{name} ({dev['real']}) held by pid {', '.join(map(str, holders))}")
    return busy


class Monitor:
    """Background probe of the rig every 15 s; ends the follow-up of the viewer's lab jobs (state, analysis after)."""

    def __init__(self, config: Config, registry: jobs.Registry, on_finished=None):
        self.config, self.registry, self.on_finished = config, registry, on_finished
        self.state, self.error, self.updated = {}, None, None
        self._lock = threading.Lock()
        self._wake = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="lab-monitor").start()

    def refresh(self):
        self._wake.set()

    def lab_jobs(self) -> list[dict]:
        return [j for j in self.registry.all(200) if j.get("mode") == "lab"][:WATCH_JOBS]

    def sample(self):
        lab_jobs = self.lab_jobs()
        try:
            raw = probe(self.config, [j["lab"]["rundir"] for j in lab_jobs if j["lab"].get("rundir")],
                        [j["log"] for j in lab_jobs if not j["lab"].get("rundir")])
        except (RuntimeError, ValueError, OSError) as e:
            with self._lock:
                self.error, self.updated = str(e)[:300], time.time()
            return
        runs = [parse_run(r, raw["now"], raw["tz_offset"]) for r in raw["runs"]]
        snapshot = {"devices": raw["devices"], "processes": raw["processes"], "runs": runs, "now": raw["now"]}
        self._follow_jobs(lab_jobs, runs)
        with self._lock:
            self.state, self.error, self.updated = snapshot, None, time.time()

    def _follow_jobs(self, lab_jobs: list[dict], runs: list[dict]):
        """Records each lab job's rundir once its log names it; when the job ended, its outcome and the analysis after."""
        self.registry.refresh()
        for job in lab_jobs:
            job = self.registry.get(job["id"])
            changed = False
            if not job["lab"].get("rundir"):
                found = next((r for r in runs if job["pid"] in r["writers"]), None)
                if found is None:
                    found = self._rundir_from_log(job, runs)
                if found:
                    job["lab"]["rundir"], job["run"] = found["rundir"], found["key"]
                    changed = True
            run = next((r for r in runs if r["rundir"] == job["lab"].get("rundir")), None)
            if job["state"] not in ("running", "pending") and run and not job["lab"].get("outcome"):
                job["lab"]["outcome"] = run["state"]
                if run["state"] == "stopped" and job["state"] == "done":
                    job["state"] = "stopped"
                changed = True
                if job["lab"].get("analyse_after") and run["finished"] and self.on_finished:
                    try:
                        job["lab"]["analysis_job"] = self.on_finished(run, job)["id"]
                    except Exception as e:          # the run is done either way; show why nothing followed
                        job["lab"]["analysis_error"] = str(e)[:300]
            if changed:
                self.registry.save(job)

    def _rundir_from_log(self, job: dict, runs: list[dict]) -> dict | None:
        text = jobs.job_tail(job, 65536)
        m = re.search(r"Results will be saved under (\S+)", text)
        return next((r for r in runs if m and r["rundir"] == m.group(1)), None)

    def _loop(self):
        while True:
            try:
                self.sample()
            except Exception as e:      # keep the thread alive; show the error on the tab
                with self._lock:
                    self.error, self.updated = f"{type(e).__name__}: {e}"[:300], time.time()
            self._wake.wait(PERIOD_S)
            self._wake.clear()

    def snapshot(self) -> dict:
        with self._lock:
            state, error, updated = dict(self.state), self.error, self.updated
        return {**state, "error": error, "updated": updated, "host": self.config.lab_host,
                "busy": guard(state, self.config.lab_services) if state else ["rig not sampled yet"],
                "services": self.config.lab_services, "jobs": self.lab_jobs(),
                "routines": list(ROUTINES)}

    def known_rundir(self, rundir: str) -> dict:
        """The run of the last snapshot at `rundir` (only those may be stopped or have their log read); KeyError else."""
        with self._lock:
            runs = list(self.state.get("runs", []))
        for run in runs:
            if run["rundir"] == rundir:
                return run
        raise KeyError(rundir)


def launch(config: Config, registry: jobs.Registry, routine: str, run_config: dict, label: str,
           analyse_after: bool, ghost: str) -> dict:
    """Writes the routine's config to the rig and starts it there detached (login shell: the Basler variables of
    /etc/profile.d); registers it as a lab job."""
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    state = PurePosixPath(config.lab_state)
    yaml_path, log = state / "configs" / f"{job_id}.yaml", state / "logs" / f"{job_id}.log"
    ssh(config, f"mkdir -p {shlex.quote(str(yaml_path.parent))} {shlex.quote(str(log.parent))} && "
                f"cat > {shlex.quote(str(yaml_path))}", stdin=yaml.safe_dump(run_config, sort_keys=False))
    argv = [config.lab_python, "-u", "-m", ROUTINES[routine]["module"], str(yaml_path)]
    pid = resources.run_detached(jobs._detached(argv, {"PYTHONUNBUFFERED": "1"}, nice=None, login=True), config.lab_host,
                                 log, f"{config.lab_root}/DefocusTest")
    job = {"id": job_id, "run": f"lab:{routine}", "label": label, "argv": argv, "ghost": ghost, "mode": "lab",
           "resources": {"mode": "lab"}, "started": datetime.now().isoformat(timespec="seconds"), "state": "running",
           "host": config.lab_host, "pid": pid, "log": str(log), "log_host": config.lab_host,
           "lab": {"routine": routine, "config": str(yaml_path), "rundir": None, "analyse_after": analyse_after}}
    return registry.save(job)


def request_stop(config: Config, rundir: str) -> None:
    ssh(config, f"touch {shlex.quote(rundir + '/meta/' + STOP_FILE)}")


def run_log(config: Config, rundir: str, size: int = 65536) -> str:
    return jobs.ANSI.sub("", ssh(config, f"tail -c {int(size)} {shlex.quote(rundir + '/log.log')}"))
