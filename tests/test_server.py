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
