"""Settings of the viewer: data roots, port, state folders, GPU hosts and the default resources of a job.

Defaults below; ~/.config/defocusViewer/config.toml overrides any of them (same names).
"""
import os
import secrets
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "defocusViewer"


@dataclass
class Config:
    data_root: Path = Path(os.environ.get("POS_LOCAL_DATA", "/disk/users/llebru/Defocus"))
    extra_roots: dict = field(default_factory=lambda: {
        "defocusApSim-b": Path.home() / "projects" / "defocusApSim-b" / "data"})
    test: str = "Defocus"
    remote_devices: list = field(default_factory=lambda: ["Orbray6", "MPS6", "calibration_stand"])
    host: str = "0.0.0.0"
    port: int = 8765
    state_dir: Path = Path("/disk/users/llebru/defocusViewer")
    cache_dir: Path = Path("/scratch/llebru/defocusViewer")
    python: str = sys.executable
    gpu_hosts: list = field(default_factory=lambda: ["farm-gpu5", "farm-gpu1", "farm-gpu6"])
    # hosts a job takes GPUs from by default, in order; farm-gpu6 left out while its CUDA fails (cudaErrorUnknown, 2026-10-22)
    job_gpu_hosts: list = field(default_factory=lambda: ["farm-gpu5", "farm-gpu1"])
    launch_hosts: list = field(default_factory=lambda: ["farm-gpu1", "farm-gpu5"])
    gpus_per_job: int = 2
    gpu_job_cpu_workers: int = 4
    slurm_cpus: int = 10
    slurm_partition: str = "standard"
    slurm_mem_per_cpu: str = "2000M"
    refit_workers: int = 4

    @property
    def jobs_dir(self) -> Path:
        return self.state_dir / "jobs"

    @property
    def spotfits_dir(self) -> Path:
        return self.state_dir / "spotfits"


def load(path: Path = CONFIG_DIR / "config.toml") -> Config:
    """Config with the values of `path` (when it exists) over the defaults."""
    values = tomllib.loads(path.read_text()) if path.exists() else {}
    config = Config()
    for f in fields(Config):
        if f.name in values:
            value = values[f.name]
            if f.name == "extra_roots":
                value = {label: Path(p).expanduser() for label, p in value.items()}
            elif isinstance(getattr(config, f.name), Path):
                value = Path(value).expanduser()
            setattr(config, f.name, value)
    return config


def token(path: Path = CONFIG_DIR / "token") -> str:
    """The secret of the login cookie, created on first use (readable by this user only)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(24))
    return path.read_text().strip()
