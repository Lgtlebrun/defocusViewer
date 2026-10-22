"""Cluster usage for the panel and for job defaults: GPUs and cores of the GPU hosts, Slurm nodes and our jobs, sampled
in the background (one snapshot shared by every page)."""
import json
import os
import re
import socket
import subprocess
import threading
import time

from amkit import resources
from DefocusTest.psf_batch import GPU_HOST_LIMITS, GPU_MIN_FREE_MB

from .config import Config

HOST_PERIOD_S, SLURM_PERIOD_S = 15, 30
JOB_COMMAND = re.compile(r"(DefocusTest\.analysis\.run_\w+|defocusviewer jointfit)\s+(\S+)(.*)")


def master_up(host: str) -> bool:
    """True when the SSH ControlMaster to `host` is open (or the host is this machine)."""
    if resources.is_local(host):
        return True
    return subprocess.run(["ssh", "-O", "check", host], capture_output=True, timeout=10).returncode == 0


def host_state(host: str) -> dict:
    """GPUs (with the ones a job of ours may take now) and cores of one host; `reachable` False when SSH fails."""
    if not master_up(host):
        return {"host": host, "reachable": False, "hint": f"ssh master down: run `ssh -fN {host}` in a terminal"}
    cpu = resources.cpu_status(None if resources.is_local(host) else host)
    if cpu is None:
        return {"host": host, "reachable": False, "hint": "no answer over ssh"}
    target = None if resources.is_local(host) else host
    gpus = resources.gpu_status(target)
    cap = resources.host_cap(GPU_HOST_LIMITS, target)
    free = resources.free_within_cap(gpus, cap, min_free_mb=GPU_MIN_FREE_MB)
    processes = [p for p in resources.gpu_processes(target) if p["uid"] == os.getuid()]
    return {"host": host, "reachable": True, **cpu, "gpus": gpus, "gpu_error": not gpus, "free_gpus": free, "cap": cap,
            "our_processes": processes}


def process_tree(host: str) -> dict[int, tuple[int, str]]:
    """{pid: (parent pid, command)} of every process of `host`."""
    out = resources._shell("ps -eo pid=,ppid=,args=", None if resources.is_local(host) else host) or ""
    tree = {}
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            tree[int(fields[0])] = (int(fields[1]), fields[2] if len(fields) > 2 else "")
    return tree


def ancestors(pid: int, tree: dict) -> list[int]:
    """`pid` and its parents up to (not including) init."""
    chain = []
    while pid in tree and pid > 1 and pid not in chain:
        chain.append(pid)
        pid = tree[pid][0]
    return chain


def short(host: str) -> str:
    return (host or socket.gethostname()).split(".")[0]


def owner_job(host: str, process: dict, trees: dict, jobs_by_pid: dict) -> dict:
    """What a GPU process of ours belongs to: the analysis command above it (rundir, steps) and the viewer job.

    A worker started over SSH (amkit.gpupool) is matched to the ssh process on a launch host that carries its host and
    CUDA_VISIBLE_DEVICES; a local one is followed up its own process tree."""
    chains = []
    if short(host) in trees:
        chains.append((short(host), ancestors(process["pid"], trees[short(host)])))
    for launch, tree in trees.items():
        for pid, (_, args) in tree.items():
            words = args.split()
            if ("amkit.gpupool" in args and words and words[0].endswith("ssh") and host in words
                    and f"CUDA_VISIBLE_DEVICES={process['gpu_index']}" in words):
                chains.append((launch, ancestors(pid, tree)))
    for launch, chain in chains:
        job = next((jobs_by_pid[(launch, pid)] for pid in chain if (launch, pid) in jobs_by_pid), None)
        command = next((m for pid in chain if (m := JOB_COMMAND.search(trees[launch][pid][1]))), None)
        if command or job:
            only = re.search(r"--only ((?:\w+ ?)+)", command.group(3)) if command else None
            return {"run": command.group(2) if command else job["run"].rsplit("/", 1)[-1],
                    "steps": only.group(1).split() if only else [], "job_id": job and job["id"],
                    "job_label": job and job["label"], "launch_host": launch}
    return {}


def our_gpus(hosts: dict, trees: dict, jobs: list[dict]) -> list[dict]:
    """One row per GPU process of ours, with the job it belongs to."""
    jobs_by_pid = {(short(j.get("host")), j["pid"]): j for j in jobs if j.get("pid")}
    rows = []
    for host, state in hosts.items():
        for process in state.get("our_processes", []):
            rows.append({"host": host, **process, **owner_job(host, process, trees, jobs_by_pid)})
    return rows


def usable_cores(node: dict, mem_per_cpu_mb: int) -> int:
    """Idle cores a job at `mem_per_cpu_mb` per core could get now (Slurm here reserves memory, so it often binds first)."""
    if any(s in node["state"] for s in ("drain", "down", "fail", "maint", "reserved")):
        return 0
    free_memory = max(0, node["memory_mb"] - node["memory_allocated_mb"])
    return min(node["cpus_idle"], free_memory // mem_per_cpu_mb)


def _mem_mb(text: str) -> int:
    value, unit = (text[:-1], text[-1].upper()) if text[-1:].isalpha() else (text, "M")
    return int(float(value) * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[unit])


class Sampler:
    """Background sampling of the GPU hosts every 15 s and of Slurm every 30 s."""

    def __init__(self, config: Config):
        self.config = config
        self.hosts, self.slurm, self.updated, self.ours = {}, {"nodes": [], "jobs": {}}, {}, []
        self._lock = threading.Lock()

    def start(self):
        for target, name in ((self._hosts_loop, "cluster-hosts"), (self._slurm_loop, "cluster-slurm")):
            threading.Thread(target=target, daemon=True, name=name).start()

    def _hosts_loop(self):
        while True:
            trees = None
            for host in self.config.gpu_hosts:
                try:
                    state = host_state(host)
                    if state.get("our_processes") and trees is None:     # right away: a short job's workers are soon gone
                        trees = {short(h): process_tree(h) for h in self.config.launch_hosts if master_up(h)}
                except (OSError, subprocess.SubprocessError, ValueError) as e:
                    state = {"host": host, "reachable": False, "hint": str(e)[:200]}
                with self._lock:
                    self.hosts[host], self.updated["hosts"] = state, time.time()
            with self._lock:
                hosts = dict(self.hosts)
            ours = our_gpus(hosts, trees, self._jobs()) if trees else []
            with self._lock:
                self.ours = ours
            time.sleep(HOST_PERIOD_S)

    def _jobs(self) -> list[dict]:
        folder = self.config.jobs_dir
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))[-200:]] if folder.exists() else []

    def _slurm_loop(self):
        while True:
            try:
                nodes, jobs = resources.slurm_nodes(), resources.slurm_jobs()
                with self._lock:
                    self.slurm, self.updated["slurm"] = {"nodes": nodes, "jobs": jobs}, time.time()
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(SLURM_PERIOD_S)

    def snapshot(self) -> dict:
        """Hosts, Slurm nodes and partitions (cores idle and usable at the default memory per core), our Slurm jobs."""
        with self._lock:
            hosts = [self.hosts.get(h, {"host": h, "reachable": None}) for h in self.config.gpu_hosts]
            nodes, jobs, updated = list(self.slurm["nodes"]), dict(self.slurm["jobs"]), dict(self.updated)
            ours = list(self.ours)
        mem_per_cpu = _mem_mb(self.config.slurm_mem_per_cpu)
        partitions = {}
        for node in nodes:
            node["usable"] = usable_cores(node, mem_per_cpu)
            for name in node["partitions"]:
                p = partitions.setdefault(name, {"partition": name, "nodes": 0, "cpus_total": 0, "cpus_allocated": 0,
                                                 "cpus_idle": 0, "usable": 0})
                p["nodes"] += 1
                for k in ("cpus_total", "cpus_allocated", "cpus_idle", "usable"):
                    p[k] += node[k]
        return {"hosts": hosts, "nodes": nodes, "partitions": list(partitions.values()), "jobs": jobs, "our_gpus": ours,
                "updated": updated, "mem_per_cpu_mb": mem_per_cpu, "this_host": socket.gethostname(),
                "user": os.environ.get("USER", "")}

    def free_gpus(self) -> dict[str, list[int]]:
        """{host: GPU indices a job of ours may take now}, in the configured host order."""
        with self._lock:
            return {h: list(s.get("free_gpus", [])) for h in self.config.gpu_hosts if (s := self.hosts.get(h))}
