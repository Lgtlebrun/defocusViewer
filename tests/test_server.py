import json
import os
import socket

import pytest
from fastapi.testclient import TestClient

from defocusviewer.server import COOKIE, checked_resources, create_app

RUN = "Orbray6/Defocus/20261017-175317_stability-test"


@pytest.fixture
def client(config, rundir):
    return TestClient(create_app(config, "secret", background=False))


def test_login_required(client):
    assert client.get("/api/runs").status_code == 401
    assert client.get("/").status_code == 401
    assert client.get("/login", params={"token": "wrong"}).status_code == 401
    response = client.get("/login", params={"token": "secret", "next": "//evil.example"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert COOKIE in response.cookies
    assert client.get("/api/runs").status_code == 200


def test_paths_and_runs_checked(client):
    client.cookies.set(COOKIE, "secret")
    assert [r["key"] for r in client.get("/api/runs").json()["runs"]] == [RUN]
    assert client.get("/api/run/file", params={"run": RUN, "path": "../meta/config.yaml"}).status_code == 404
    assert client.get("/api/run/info", params={"run": "../../etc"}).status_code == 404
    assert client.get("/api/spot", params={"run": RUN, "img": "nope.npy"}).status_code == 404
    assert client.get("/api/spot", params={"run": RUN, "img": "fa_25000_train.npy", "ghost": "maybe"}).status_code == 400
    spots = client.get("/api/run/spots", params={"run": RUN}).json()
    assert len(spots["rows"]) == 10


def test_job_requests_validated(client, config):
    client.cookies.set(COOKIE, "secret")
    bad_step = {"run": RUN, "analysis": "flow", "steps": ["rm -rf"], "resources": {"mode": "slurm"}}
    assert client.post("/api/jobs", json=bad_step).status_code == 400
    with pytest.raises(Exception):
        checked_resources(config, {"mode": "gpu", "n_gpus": 2, "gpu_hosts": ["evil-host"], "cpu_workers": 4,
                                   "launch_host": "farm-gpu1"})
    with pytest.raises(Exception):
        checked_resources(config, {"mode": "slurm", "cpus": 4, "partition": "standard", "mem_per_cpu": "1G; rm"})
    ok = checked_resources(config, {"mode": "slurm", "cpus": 4, "partition": "express", "mem_per_cpu": "2000M", "time": "1:00:00"})
    assert ok["cpus"] == 4


def test_pinned_runs_kept_on_the_server(client, config):
    client.cookies.set(COOKIE, "secret")
    assert client.get("/api/pinned").json() == []
    assert client.post("/api/pinned", json={"run": RUN, "pinned": True}).json() == [RUN]
    assert client.post("/api/pinned", json={"run": RUN, "pinned": True}).json() == [RUN]      # once
    assert (config.state_dir / "pinned_runs.json").exists()
    assert client.post("/api/pinned", json={"run": RUN, "pinned": False}).json() == []


def test_info_has_a_readable_summary(client):
    client.cookies.set(COOKIE, "secret")
    info = client.get("/api/run/info", params={"run": RUN}).json()
    titles = [s["title"] for s in info["summary"]]
    assert titles[:2] == ["Run", "Routines"]
    run = dict(info["summary"][0]["rows"])
    assert run["type of test"].startswith("stability") and run["fibers"] == "2: Fa, Fb"


def test_fiber_figure_drawn_by_matplotlib(client):
    client.cookies.set(COOKIE, "secret")
    options = client.get("/api/run/fibers/options", params={"run": RUN}).json()
    assert options["fibers"] == ["Fa", "Fb"] and options["routines"] == ["train", "stability"]
    assert [q[0] for q in options["quantities"]] == ["texp"]       # no fit yet: only the metadata
    query = {"run": RUN, "quantities": "texp", "fibers": "Fa,Fb", "routines": "train,stability", "x": "x_position"}
    png = client.get("/api/run/fibers/figure", params=query)
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    pdf = client.get("/api/run/fibers/figure", params={**query, "format": "pdf", "dpi": 400})
    assert pdf.content[:4] == b"%PDF" and "attachment" in pdf.headers["content-disposition"]
    assert client.get("/api/run/fibers/figure", params={**query, "quantities": "nope"}).status_code == 400
    assert client.get("/api/run/fibers.csv", params={"run": RUN}).text.startswith("img,fiber,routine")


def test_analyses_wait_for_a_running_fetch(client, config):
    # rsync fills data/ before meta/: while a fetch copies the run, the analyses say so instead of a misleading reason
    client.cookies.set(COOKIE, "secret")
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    log = config.jobs_dir / "j.log"
    log.write_text("Starting rsync\n")
    job = {"id": "j", "run": RUN, "label": "fetch", "mode": "local", "host": socket.gethostname(), "pid": os.getpid(),
           "started": "2026-10-30T11:31:49", "state": "running", "log": str(log)}
    (config.jobs_dir / "j.json").write_text(json.dumps(job))
    data = client.get("/api/analyses", params={"run": RUN}).json()
    assert "still copying" in data["reason"] and not data["viewer_joint_fit"]
    assert client.post("/api/jobs", json={"run": RUN, "analysis": "flow", "steps": ["fit"], "resources": {"mode": "slurm"}}).status_code == 409
    log.write_text(f"Starting rsync\nfetched {RUN}\nanalysing\n")      # fetch + analyse past its copy
    assert "still copying" not in (client.get("/api/analyses", params={"run": RUN}).json()["reason"] or "")
