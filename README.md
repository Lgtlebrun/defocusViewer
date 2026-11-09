# defocusViewer

A browser GUI for the Defocus test. It runs as one service on farm-gpu1 and does five things:
- browses the rundirs and their figures;
- opens single spots with their metadata and fit (model, residual, FFT views, with or without the ghost mask);
- compares spots side by side;
- launches the DefocusTest analyses on GPUs or Slurm;
- fetches runs from pos-control and shows cluster usage.

## Use

- Open the login URL once per browser. Print it with `defocusviewer url` (from `~/Defocus/.venv`). It sets a cookie; every other request needs it.
- The service runs in tmux session `defocusviewer`. A crontab line restarts it when it stops answering: `defocusviewer ensure`, every 5 min.
- The server log is in `/disk/users/llebru/defocusViewer/server.log`.

## What it reads and writes

| | Location |
|---|---|
| Runs | `$POS_LOCAL_DATA/<device>/Defocus/<rundir>` (default `/disk/users/llebru/Defocus`), plus the extra roots of the config. Browsing never writes into a rundir. |
| Raw data | A rundir's `data/` (the rig's stacks and acquisition tables, 90 % of its bytes) is a symlink to `/disk/data1/des/Leonard/Defocus/<device>/Defocus/<rundir>/data` (`raw_root`, farm-se1 NFS, seen by farm-gpu1 and farm-gpu2). results/, plot/, meta/ and the logs stay in the rundir. See [Raw data](#raw-data). |
| Runs on the rig | `pos-control:/mnt/data1/positioners/data/<device>/Defocus`, listed over SSH every 10 min. Fetch copies with rsync and never deletes. |
| Viewer state | `/disk/users/llebru/defocusViewer/`: `jobs/` (one JSON and log per job) and `spotfits/` (spots the viewer fitted). |
| Thumbnails | `/scratch/llebru/defocusViewer/` |
| Settings | `~/.config/defocusViewer/config.toml` overrides the defaults of `defocusviewer/config.py`: hosts, roots, port, job resources. The login secret is `~/.config/defocusViewer/token`. |

## How it models a spot

- The model is `fit.render_model_shared` with the run's theta and pupil scale, as `ModelFitPlots` does.
- The parameters are those of the pipeline's saved fit for that spot:
  - train spots: `joint_fit_diag_records` (and `_unmasked`);
  - hold frames: `<fiber>_<routine>_timeseries_zernike.npz`;
  - train_end spots: `<fiber>_train_end_fit.hdf5`.
- Whether a saved series used the ghost mask is read from `log.log` ("ghost mask applied" after the last series run).
- When no saved fit has the requested ghost setting, the viewer fits that one spot itself (`spots.refit`).
  - It holds theta and the pupil scale fixed, with r fixed for hold frames when `fit.fix_r` is set.
  - It starts from the saved fit and a generic seed, and keeps the result in `spotfits/`.
- residual_frac follows each source's own convention:
  - the joint fit takes the image peak outside the ghost box;
  - the per-frame fits take the peak of the whole image.

## Jobs

- **GPU steps** (fit, ghost_mask, refit, series) default to up to 2 idle GPUs of `job_gpu_hosts` (farm-gpu1, farm-gpu2). Such a job runs detached on farm-gpu1 at nice 10.
- **Other steps, or no idle GPU:** a Slurm job of 10 cores.
- **Guard:** a launch is refused while a viewer job, an analysis process (farm-gpu1, farm-gpu5) or a Slurm job of the same name works on the rundir.
- **Rig watchdog:** `defocusviewer watch start` fetches each run that finishes on the rig and analyses its `_tophatbl-circle` variant on the P100s. [WATCHDOG.md](WATCHDOG.md) covers how to stop, check and restart it.

## Raw data

The raw `data/` of a rundir is a copy of the rig's, so it lives on `raw_root` (`/disk/data1`), not on `/disk/users`; the rundir holds a link to it (`defocusviewer/rawdata.py`). posUtils, DefocusTest and the viewer read through the link unchanged, as they already did for variants.

- **Fetch** (`defocusviewer fetch`, the Fetch button, the watchdog) copies `data/` onto `raw_root` first, then the rest of the run into the rundir, and links the two. `--force` resyncs both and keeps the link.
- **posUtils' own fetch** (`ensureRunDir`, so DefocusTest `--now` on a run that is not local yet, or `--force`) copies `data/` into the rundir: its `rsync -a` replaces the link by a directory. Prefer `defocusviewer fetch [--force] <device> <rundir>`.
- **Sweep:** the `ensure` cron (every 5 min) starts `defocusviewer move-data --all` (nice, idle I/O) when a rundir has a real `data/` untouched for 10 min. It copies it onto `raw_root`, checks every file arrived unchanged (`rsync --dry-run`), then switches to the link and removes the copy. A run is skipped while a job, an analysis process or a Slurm job works on it or on one of its variants (`<rundir>_<suffix>`). Log: `/scratch/llebru/defocusViewer/rawdata.log`.
- **By hand:** `defocusviewer move-data [--dry-run] <rundir>...` or `--all`.

## Tests

```
~/Defocus/.venv/bin/python -m pytest tests
```

The numerical tests use the real `20261017-175317_stability-fnum4-2` rundir, and are skipped without it.
