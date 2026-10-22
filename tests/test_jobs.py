import subprocess
import time

import pytest

from defocusviewer import jobs


def test_default_resources(config):
    free = {"farm-gpu5": [2], "farm-gpu1": [0, 1], "farm-gpu6": [0]}
    gpu = jobs.default_resources(config, free, uses_gpu=True)
    assert gpu == {"mode": "gpu", "n_gpus": 2, "gpu_hosts": ["farm-gpu5", "farm-gpu1"], "cpu_workers": 4,
                   "launch_host": "farm-gpu1"}
    assert jobs.default_resources(config, {"farm-gpu6": [0]}, uses_gpu=True)["mode"] == "slurm"   # gpu6 not a job host
    assert jobs.default_resources(config, free, uses_gpu=False) == {
        "mode": "slurm", "cpus": 10, "partition": "standard", "mem_per_cpu": "2000M", "time": None}


def test_command(config):
    gpu = jobs.default_resources(config, {"farm-gpu1": [0]}, True)
    argv = jobs.command(config, jobs.FLOW_MODULES["stability"], "calibration_stand", "r", ["fit", "refit"], "on", gpu)
    assert argv == ["python", "-m", "DefocusTest.analysis.run_stability", "r", "--device", "calib", "--now", "--only",
                    "fit", "refit", "--ghost-mask", "on", "--gpu", "1", "--gpu-hosts", "farm-gpu1", "--n-workers", "4"]
    cpu = jobs.default_resources(config, {}, False)
    assert jobs.command(config, jobs.EXTRA_MODULES["temperature"], "Orbray6", "r", None, "on", cpu)[-2:] == ["--n-workers", "10"]
    with pytest.raises(ValueError):
        jobs.command(config, jobs.EXTRA_MODULES["temperature"], "Orbray6", "r", None, "on", gpu)


def test_detached_command_reports_its_exit_code(tmp_path):
    command = jobs._detached(["bash", "-c", "echo 'quoted arg'; exit 3"], {"X": "a b"})
    out = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert out.stdout.splitlines() == ["quoted arg", "__exit__ 3"]


def test_registry_finishes_a_local_job(config):
    registry = jobs.Registry(config.jobs_dir)
    log = config.state_dir / "x.log"
    config.state_dir.mkdir(parents=True)
    process = subprocess.Popen(["bash", "-c", "echo hello; echo __exit__ 0"], stdout=open(log, "w"))
    process.wait()
    registry.save({"id": "20261022-000000-abcd", "run": "k", "label": "t", "mode": "gpu", "host": "localhost",
                   "pid": process.pid, "log": str(log), "state": "running", "started": "2026-10-22T00:00:00"})
    time.sleep(0.1)
    [job] = registry.refresh()
    assert job["state"] == "done" and job["exit_code"] == 0
    with pytest.raises(KeyError):
        registry.get("../x")


def test_gpu_worker_traced_to_its_job():
    from defocusviewer import cluster
    tree = {10: (1, "bash -c x"),
            11: (10, "/v/python -m DefocusTest.analysis.run_stability 20261015-161039_stability-fnum4 --device orbray6 --now "
                     "--only ghost_mask refit --gpu 2"),
            12: (11, "ssh -o BatchMode=yes farm-gpu5 cd /x && env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 /v/python -m amkit.gpupool"),
            13: (11, "ssh -o BatchMode=yes farm-gpu5 cd /x && env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 /v/python -m amkit.gpupool")}
    hosts = {"farm-gpu5": {"our_processes": [{"gpu_index": 2, "pid": 999, "used_mb": 5000}]}}
    [row] = cluster.our_gpus(hosts, {"farm-gpu1": tree}, [{"id": "J1", "host": "farm-gpu1", "pid": 10, "run": "a/b/x",
                                                          "label": "ghost_mask refit"}])
    assert (row["run"], row["steps"], row["job_id"]) == ("20261015-161039_stability-fnum4", ["ghost_mask", "refit"], "J1")
