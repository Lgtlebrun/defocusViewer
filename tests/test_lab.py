import json
import os
import subprocess
import time
from datetime import datetime, timezone

import pytest
import yaml
from fastapi.testclient import TestClient

from defocusviewer import jobs, lab
from defocusviewer.server import COOKIE, create_app

BASE = {"run": {"device": "MPS6", "serial_camera": 25193621, "pwm_arrays": [2, 3]},
        "equalizer": {"run": True}, "temperature": {"run": False},
        "train": {"run": True, "camera": {"burn_in_s": 7200}, "stage": {"x": [34000, 34500, 35000]}},
        "train_end": {"run": True}, "stability": {"run": True, "duration_s": 36000, "stage": {"hold_position": 35000}},
        "all": {"camera": {"n_pics": 300, "texp": 1200, "fibers": {"n_spots": 10}}},
        "fit": {"run": False, "fnum": 8}, "output": {"path": "stability_x", "debug": True}}


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def line(t, level, category, msg):
    return f"({stamp(t)}) [{level}] [{category}] : {msg}"


def raw_run(marks, writers=(1234,), status=None, finished=False, stop=False, stacks=None, config=BASE):
    return {"rundir": "/data/MPS6/Defocus/20261027-121028_stability-x", "writers": list(writers), "marks": marks,
            "last_line": marks[-1][:21] if marks else None, "finished": finished, "stop_requested": stop,
            "status": status, "config": yaml.safe_dump(config), "stacks": stacks or {}}


def test_merged_config_checks_every_override():
    cfg, changes = lab.merged_config("StabilityTest", BASE, {"stability.duration_s": 120, "train.stage.x": [34900, 35000],
                                                                 "train_end.run": False})
    assert cfg["stability"]["duration_s"] == 120.0 and cfg["train"]["stage"]["x"] == [34900, 35000]
    assert {c["key"] for c in changes} == {"stability.duration_s", "train.stage.x", "train_end.run"}
    assert BASE["stability"]["duration_s"] == 36000                    # the base is not modified
    for bad in ({"evil.key": 1}, {"output.path": "x; rm -rf /"}, {"run.device": "Orbray7"}, {"all.camera.n_pics": 1.5},
                {"stability.duration_s": True}, {"train.stage.x": [34000, 34500]}, {"run.pwm_arrays": "2"}):
        with pytest.raises(ValueError):
            lab.merged_config("StabilityTest", BASE, bad)


def test_run_config_with_ruamel_tags_is_read():
    text = "train:\n  camera:\n    burn_in_s: !!python/object/new:ruamel.yaml.scalarint.ScalarInt\n      args:\n      - 7200\n" \
           "      state:\n        _width: 4\n"
    assert yaml.load(text, Loader=lab._RunConfigLoader) == {"train": {"camera": {"burn_in_s": 7200}}}


def test_parse_run_in_hold():
    t0 = 1_800_000_000
    marks = [line(t0, "INFO", "StabilityTest", "Starting StabilityTest..."),
             line(t0 + 1, "INFO", "Main:PWMSpotsEqualizer", "Starting PWMSpotsEqualizer..."),
             line(t0 + 60, "INFO", "Main:PWMSpotsEqualizer", "PWMSpotsEqualizer completed in 1m"),
             line(t0 + 61, "INFO", "Main:train", "Starting train..."),
             line(t0 + 62, "INFO", "Main:train", "Sweep x_positions : [34000 34500") + " 35000]",
             line(t0 + 70, "INFO", "Stage", "Arrived to position 34999.875"),
             line(t0 + 100, "INFO", "Main:train", "train completed in 39s"),
             line(t0 + 101, "INFO", "Main:stability", "Starting stability..."),
             line(t0 + 102, "INFO", "Main:stability", "Holding in place: stage at 35000.0 µm, hold pos 35000 µm"),
             line(t0 + 103, "INFO", "Main:stability", "Hold at pos=35000 µm: (duration=1000.0 s, texp=900 µs, n_stack=300)")]
    run = lab.parse_run(raw_run(marks, stacks={"stability": {"stacks": 7, "positions": 1}}), t0 + 603, 0)
    assert [(p["name"], p["state"]) for p in run["phases"]] == [("PWMSpotsEqualizer", "done"), ("train", "done"), ("stability", "running")]
    assert run["state"] == "running" and not run["hook"] and run["key"] == "MPS6/Defocus/20261027-121028_stability-x"
    assert run["current"]["kind"] == "hold" and run["current"]["fraction"] == pytest.approx(0.5) and run["current"]["stacks"] == 7
    assert run["stage"] == {"x_um": 35000.0, "t": t0 + 102}
    assert run["config"]["stability.duration_s"] == 36000


def test_parse_run_sweep_progress_from_status_and_files():
    t0 = 1_800_000_000
    marks = [line(t0, "INFO", "StabilityTest", "Starting StabilityTest..."),
             line(t0 + 1, "INFO", "Main:train_end", "Starting train_end..."),
             line(t0 + 2, "INFO", "Main:train_end", "Sweep x_positions : [34000 34500 35000]")]
    run = lab.parse_run(raw_run(marks, stacks={"train-end": {"stacks": 2, "positions": 2}}), t0 + 10, 0)
    assert run["current"] == {"name": "train_end", "kind": "sweep", "done": 2, "total": 3, "fraction": pytest.approx(2 / 3)}
    status = {"routine": "train_end", "kind": "sweep", "stop": 3, "n_stops": 3, "x_readback_um": 35000.125, "t": stamp(t0 + 9)}
    run = lab.parse_run(raw_run(marks, status=status), t0 + 10, 0)
    assert run["current"]["done"] == 3 and run["hook"] and run["stage"] == {"x_um": 35000.125, "t": t0 + 9}


def test_parse_run_outcomes():
    t0 = 1_800_000_000
    start = [line(t0, "INFO", "StabilityTest", "Starting StabilityTest..."),
             line(t0 + 1, "INFO", "Main:stability", "Starting stability...")]
    stopped = start + [line(t0 + 5, "WARNING", "Main:stability", "Stop requested by user, ending stability early"),
                       line(t0 + 6, "INFO", "Main:stability", "stability completed in 5s"),
                       line(t0 + 7, "INFO", "StabilityTest", "StabilityTest completed in 7s")]
    assert lab.parse_run(raw_run(stopped, writers=(), finished=True, stop=True), t0 + 8, 0)["state"] == "stopped"
    failed = start + [line(t0 + 5, "ERROR", "Main:stability", "stability failed !"),
                      line(t0 + 6, "ERROR", "StabilityTest", "StabilityTest failed !")]
    run = lab.parse_run(raw_run(failed, writers=(), finished=True), t0 + 8, 0)
    assert run["state"] == "failed" and len(run["errors"]) == 2
    assert lab.parse_run(raw_run(start, writers=()), t0 + 8, 0)["state"] == "died"
    hung = lab.parse_run(raw_run(failed), t0 + 5 + lab.STALLED_S + 60, 0)       # failed, process still alive
    assert hung["state"] == "running" and hung["stalled"]


def test_guard_ignores_monitoring_services():
    snapshot = {"processes": [{"pid": 1, "args": "python -m LabUZHMonitoring.services.TCsMonitoring"},
                              {"pid": 2, "args": "python -m DefocusTest.sequences.StabilityTest cfg"}],
                "devices": {"stage": {"real": "/dev/ttyACM0", "holders": [1]}, "mount": {"real": "/dev/x", "holders": [9]},
                            "camera 1": {"real": "/dev/bus/usb/002/045", "holders": []}}}
    assert lab.guard({**snapshot, "processes": snapshot["processes"][:1]}, "LabUZHMonitoring") == []
    assert lab.guard({**snapshot, "processes": snapshot["processes"][:1]}, "") == ["stage (/dev/ttyACM0) held by pid 1"]
    assert len(lab.guard(snapshot, "LabUZHMonitoring")) == 1


def run_locally(config, command, stdin=None, timeout=60):
    out = subprocess.run(["bash", "-c", command], input=stdin, capture_output=True, text=True, timeout=timeout)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return out.stdout


def test_probe_reads_a_run_held_open(config, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "ssh", run_locally)
    rundir = tmp_path / "rig" / "MPS6" / "Defocus" / "20261027-121028_probe"
    for sub in ("meta", "data"):
        (rundir / sub).mkdir(parents=True)
    t0 = time.time() - 30
    (rundir / "meta" / "config.yaml").write_text(yaml.safe_dump(BASE))
    (rundir / "meta" / "status.json").write_text(json.dumps({"routine": "stability", "kind": "hold", "stack": 2}))
    for name in ("fa_35000_stability_0.npy", "fb_35000_stability_0.npy", "fa_35000_stability_1.npy", "fa_34000_train.npy"):
        (rundir / "data" / name).touch()
    device = tmp_path / "ttyFAKE"
    device.touch()
    config.lab_data, config.lab_devices = str(tmp_path / "rig"), {"stage": str(device)}
    with open(rundir / "log.log", "w") as log, open(device):
        log.write("\n".join([line(t0, "INFO", "StabilityTest", "Starting StabilityTest..."),
                             line(t0, "DEBUG", "Main", "noise"),
                             line(t0 + 1, "INFO", "Main:train", "Sweep x_positions : [34000 34500"),
                             ">>  35000]", ""]))
        log.flush()
        raw = lab.probe(config, [], [])
    run = next(r for r in raw["runs"] if r["rundir"] == str(rundir))
    assert os.getpid() in run["writers"] and run["status"]["stack"] == 2
    assert run["stacks"] == {"stability": {"stacks": 2, "positions": 1}, "train": {"stacks": 1, "positions": 1}}
    assert run["marks"][-1].endswith("[34000 34500 35000]") and len(run["marks"]) == 2
    assert raw["devices"]["stage"]["present"] and os.getpid() in raw["devices"]["stage"]["holders"]
    parsed = lab.parse_run(run, raw["now"], raw["tz_offset"])
    assert parsed["state"] == "running" and parsed["hook"]


def test_launch_writes_the_config_and_starts_in_a_login_shell(config, monkeypatch):
    calls = {}
    monkeypatch.setattr(lab, "ssh", lambda c, command, stdin=None, timeout=60: calls.setdefault("ssh", (command, stdin)) and "")
    monkeypatch.setattr(lab.resources, "run_detached", lambda cmd, host, out, cwd: calls.setdefault("run", (cmd, host, str(out), cwd)) and 4242)
    registry = jobs.Registry(config.jobs_dir)
    job = lab.launch(config, registry, "StabilityTest", BASE, "label", True, "on")
    command, stdin = calls["ssh"]
    assert command.startswith("mkdir -p ") and f"configs/{job['id']}.yaml" in command and yaml.safe_load(stdin) == BASE
    cmd, host, out, cwd = calls["run"]
    assert cmd.startswith("setsid bash -lc ") and "nice" not in cmd and "DefocusTest.sequences.StabilityTest" in cmd
    assert host == config.lab_host and cwd == f"{config.lab_root}/DefocusTest" and out == job["log"]
    assert registry.get(job["id"])["pid"] == 4242 and job["log_host"] == config.lab_host and job["mode"] == "lab"


def test_job_tail_reads_a_remote_log(monkeypatch):
    seen = []
    monkeypatch.setattr(jobs.resources, "_shell", lambda cmd, host=None: seen.append((cmd, host)) or "\x1b[1mok\n__exit__ 0\n")
    assert jobs.job_tail({"log": "/r/x.log", "log_host": "rig"}, 100) == "ok\n__exit__ 0\n"
    assert seen == [("tail -c 100 /r/x.log", "rig")]


def test_follow_marks_a_stopped_job_and_starts_the_analysis(config, monkeypatch):
    registry = jobs.Registry(config.jobs_dir)
    job = registry.save({"id": "20261027-150000-abcd", "run": "lab:StabilityTest", "label": "x", "argv": [], "mode": "lab",
                         "state": "done", "host": "rig", "pid": 77, "log": "/r/x.log", "log_host": "rig", "ghost": "on",
                         "started": "2026-10-27T15:00:00", "lab": {"routine": "StabilityTest", "rundir": None, "analyse_after": True}})
    monkeypatch.setattr(registry, "refresh", lambda: [])
    started = []
    monitor = lab.Monitor(config, registry, on_finished=lambda run, job: started.append(run["key"]) or {"id": "fetch-1"})
    monkeypatch.setattr(jobs, "job_tail", lambda job, size=0: "Results will be saved under /data/MPS6/Defocus/20261027-121028_stability-x\n")
    run = {"rundir": "/data/MPS6/Defocus/20261027-121028_stability-x", "key": "MPS6/Defocus/20261027-121028_stability-x",
           "writers": [], "state": "stopped", "finished": True}
    monitor._follow_jobs([job], [run])
    saved = registry.get(job["id"])
    assert saved["state"] == "stopped" and saved["run"] == run["key"] and saved["lab"]["analysis_job"] == "fetch-1"
    monitor._follow_jobs([saved], [run])
    assert started == [run["key"]]                      # once


@pytest.fixture
def client(config, monkeypatch):
    raw = {"now": time.time(), "tz_offset": 0, "processes": [], "runs": [],
           "devices": {"stage": {"path": "/dev/s", "real": "/dev/ttyACM0", "present": True, "holders": []}}}
    monkeypatch.setattr(lab, "probe", lambda c, rundirs, logs: raw)
    monkeypatch.setattr(lab, "base_config", lambda c, routine: BASE)
    client = TestClient(create_app(config, "secret", background=False))
    client.cookies.set(COOKIE, "secret")
    client.raw = raw
    return client


def test_launch_preview_and_guard(client, monkeypatch):
    launched = []
    monkeypatch.setattr(lab, "launch", lambda *a: launched.append(a) or {"id": "j", "pid": 1, "label": "x"})
    preview = client.post("/api/lab/launch", json={"routine": "StabilityTest", "overrides": {"stability.duration_s": 60}}).json()
    assert preview["busy"] == [] and preview["changes"][0]["new"] == 60.0 and not launched
    assert client.post("/api/lab/launch", json={"routine": "StabilityTest", "overrides": {"evil": 1}}).status_code == 400
    assert client.post("/api/lab/launch", json={"routine": "rm -rf", "confirm": True}).status_code == 404
    client.raw["devices"]["stage"]["holders"] = [99]
    response = client.post("/api/lab/launch", json={"routine": "StabilityTest", "confirm": True})
    assert response.status_code == 409 and "held by pid 99" in response.json()["detail"] and not launched
    client.raw["devices"]["stage"]["holders"] = []
    assert client.post("/api/lab/launch", json={"routine": "StabilityTest", "confirm": True}).status_code == 200 and launched


def test_stop_only_a_followed_running_run(client, monkeypatch):
    touched = []
    monkeypatch.setattr(lab, "request_stop", lambda c, rundir: touched.append(rundir))
    assert client.post("/api/lab/stop", json={"rundir": "/etc"}).status_code == 404
    client.raw["runs"] = [raw_run([line(1_800_000_000, "INFO", "StabilityTest", "Starting StabilityTest...")])]
    assert client.post("/api/lab/stop", json={"rundir": client.raw["runs"][0]["rundir"]}).status_code == 404   # not sampled yet
    assert client.post("/api/lab/launch", json={"routine": "StabilityTest"}).status_code == 200      # the preview samples
    assert client.post("/api/lab/stop", json={"rundir": client.raw["runs"][0]["rundir"]}).json()["ok"]
    assert touched == [client.raw["runs"][0]["rundir"]]
    assert client.post("/api/lab/kill", json={"job": "nope"}).status_code == 404
