# defocusViewer

A browser GUI for inspecting, re-fitting and launching the analyses of **Defocus test** runs: a laboratory test that measures,
fibre by fibre, how far each optical-fibre end sits from the focus of a telescope-like beam, and how that offset drifts
with time, temperature and orientation.

The viewer is a small web service (FastAPI backend, plain ES-module frontend, vendored Plotly) that sits on top of a run
directory tree and a fitting library. It is meant for a team that acquires runs on a lab rig and analyses them on a GPU
cluster, and wants one place to look at the data, compare spots, start analyses and see what the cluster is doing.

> **Scope.** This repository is only the viewer. The acquisition code, the PSF fitting library (`DefocusTest`), the run
> directory layer (`posUtils`) and the resource helpers (`amkit`) are separate packages that must be installed in the same
> environment. They are not public, so the viewer cannot be run from this repository alone; it is published so the design
> can be read and reused.

## The physics in short

**What is measured.** Light from each fibre of a bundle (the test uses a DESI-style fibre positioner layout) is imaged by a
camera that sits on a motorised stage moving along the optical axis. The beam has a known focal ratio (f-number, from f/1.8
to f/8 in the runs so far). A stack of images is taken as the stage moves through focus; each image of one fibre is a
*spot*.

**Why defocus matters.** For a given beam, the spot size and shape change quickly with the distance to focus. A fibre end
that sits a few micrometres too far in or out loses light and degrades the image quality of the instrument. The test
therefore needs focus offsets at the micrometre level and stable over hours.

**How it is measured: fitting the spot.** Every spot is fitted with a physical point-spread-function model rather than a
Gaussian or a threshold:
- the telescope pupil is a flat-top (top-hat) aperture, hexagonal or circular, whose size follows from the f-number and the
  wavelength;
- the wavefront is a sum of Zernike aberrations. The defocus term (called *c4*, in waves) sets the spot size; the other
  terms (astigmatism, coma, ...) describe what remains of the optics;
- the fibre's own near-field is a source model convolved with that PSF;
- a faint secondary reflection (the *ghost*) appears next to some spots. It is detected from the fit residual and masked, so
  it does not bias the fit. The viewer shows a spot with the ghost mask on or off.

**From c4 to micrometres.** For a known f-number and wavelength, the fitted defocus c4 predicts where the focus is along the
axis. The *focus offset* (`delta_z`, in micrometres) is that predicted position minus where the stage actually is. It is
tracked over a hold of the stage and compared between fibres, orientations and light levels. Because fibres are fitted
jointly, the pupil shape and wavelength are shared by the whole run and each fibre only adds its own offset.
A *calibration* step then fits the rigid motion (shift and tilt) of the four fiducial fibres at each stack and removes it, giving
calibrated focus offsets that are not affected by the tilt or drift of the whole field.

**What the run types probe** (each a different routine on the rig):

| Test | Question |
|---|---|
| stability | does the focus offset drift over hours with the stage held still (temperature, mechanics)? |
| tracking | what changes while the mount follows the sky, i.e. as gravity changes direction? |
| pointing | how does the offset depend on mount altitude and azimuth? |
| luminosity | does the fitted focus depend on the light level (camera non-linearity, flux)? |
| detection | how well are the offsets measured from a hold at a single stage position, without a dense scan through focus? |

## Features

**Runs tab**
- Lists the run directories on disk and the runs still on the rig, as one family with their analysis variants.
- Filters by test type and f-number as chips, by *fetched* / *only on the rig*, and by *analysed* / *not analysed*.
  Filters are remembered per browser.
- An Info page summarises each run: what was acquired, with which routines, stage positions and timing, how it is fitted
  and what the analyses found.
- Per-run figures and the series of every fibre (fit quantities and focus offset against time, position or temperature).

**Spot panel**
- Opens a single spot with its acquisition metadata and its fit.
- Shows a matrix of data, model, residual, PSF, source, pupil and wavefront, each as an image or as the magnitude of its
  Fourier transform, with adjustable scales. The transforms are computed in the browser.
- Right-click an image to export it as PNG, PDF (in the project's plot style) or `.npy`.
- Re-renders the model with or without the ghost mask. When no saved fit matches the requested setting, the viewer fits that
  one spot itself and caches the result; it never overwrites the run's results.

**Compare tab**: pin spots from any runs and view them side by side.

**Analyses**
- Launches the analysis flows of a run (fit, ghost mask, series, fiducial calibration with its leave-one-out test, temperature, ...) on GPUs or
  through Slurm, from the browser.
- **Variants** let you re-analyse a run with another source model, aperture shape (hexagon or circle), wavelength range,
  f-number or image oversampling, without touching the run's own results. A variant is a sibling directory that shares the
  run's raw data through a symlink.
- A **viewer joint fit** handles runs the standard pipeline cannot analyse (aperture scans, image spots, ...) by seeding
  from a reference run.

**Jobs and Cluster tabs**
- Every job has a log, state and the resources it took.
- Guards refuse a second job on a run that is already being analysed, locally or on Slurm.
- GPU jobs go to idle GPUs of the configured hosts; other steps go to the CPU host with the most idle cores, or to Slurm.
  Hosts can be marked as restricted.
- The cluster page shows the GPUs and cores in use, and which of them belong to viewer jobs.

**Lab tab (optional, needs access to the rig)**
- Launches acquisition routines on the rig computer and follows the runs, the stage and the telescope mount.
- Everything is read through one read-only probe over SSH every 15 s. The viewer never opens rig hardware, and never opens
  an HDF5 file that is still being written.
- A running routine can be stopped by a stop file that the routines check between stacks.

**Rig watchdog (optional)**
- `defocusviewer watch start` fetches every run that finishes on the rig, creates a standard variant (circular aperture,
  ghost mask on) and analyses it on the GPUs, one run at a time, as soon as they are free. See [WATCHDOG.md](WATCHDOG.md).

## Install and run

```
pip install -e .                      # FastAPI, uvicorn, numpy, pandas, tables, pillow, pyyaml
# plus DefocusTest, posUtils and amkit in the same environment
defocusviewer serve                   # port 8765 by default
defocusviewer url                     # login URL for your browser
```

Open the login URL once per browser. It sets a cookie; every other request needs it, so the service can sit on a trusted
network without a separate login system. For long-running use, start `serve` under tmux or systemd and run
`defocusviewer ensure` from cron every few minutes: it restarts the service if it stops answering.

### Configuration

Defaults are in [defocusviewer/config.py](defocusviewer/config.py). Put overrides, using the same names, in
`~/.config/defocusViewer/config.toml`. The login secret is kept next to it, in `token`.

| Setting | Meaning |
|---|---|
| `data_root` | root of the local run directories (`<root>/<device>/Defocus/<run>`); also read from `POS_LOCAL_DATA` |
| `raw_root` | where the raw image stacks live, if different from the run directories |
| `extra_roots` | further named trees of runs to browse |
| `remote_devices` | rig devices whose runs are listed over SSH |
| `run_tags` | words in a run name or hold routine that become filter chips |
| `job_gpu_hosts`, CPU hosts | where analyses may run, and the default resources of a job |
| port | where the service listens |

Browsing never writes into a run directory. The viewer's own state (job records and logs, spots it has fitted) and its
thumbnails go in separate folders that the configuration sets.

## Data layout

A run directory holds `meta/` (configuration, status), `data/` (raw stacks and acquisition tables, about 90 % of the
bytes), `results/` (fits and series), `plot/` and the logs. Runs still on the rig are listed every ten minutes and fetched
with rsync, which never deletes.

Because the raw `data/` is large, the viewer can keep it on separate storage: it is replaced in the run directory by a
symlink, so the other tools read it unchanged.
- `defocusviewer fetch` copies `data/` to the raw storage first, then the rest of the run, and links the two.
- `defocusviewer move-data [--dry-run] <run>...` (or `--all`) moves an existing run: it copies the data, checks every file
  arrived unchanged, switches to the link and removes the copy. It skips runs that a job is using.
- `defocusviewer ensure` also sweeps for runs whose `data/` was copied in full again by another tool, once it is untouched
  for ten minutes.

## How the viewer models a spot

- The model is rendered with the run's shared parameters (shape, pupil scale), as the pipeline's own diagnostic plots do.
- The per-spot parameters come from the pipeline's saved fit:
  - spots of the training and detection sweeps: the joint fit records, masked and unmasked;
  - frames of a hold: the per-fibre time series of Zernike coefficients;
  - end-of-training spots: the fibre's end-of-training fit.
- Whether a saved series used the ghost mask is read from the run's log.
- The residual fraction follows each source's own convention: the joint fit divides by the image peak outside the ghost box,
  the per-frame fits by the peak of the whole image.

## Tests

```
python -m pytest tests
```

Some numerical tests use a real run directory and are skipped when it is absent.

## Layout

| Path | Role |
|---|---|
| `defocusviewer/server.py` | HTTP API and static files |
| `defocusviewer/catalog.py` | the runs, their tags, f-numbers and what each has |
| `defocusviewer/spots.py`, `jointfit.py`, `variants.py` | spot rendering and refits, viewer joint fit, analysis variants |
| `defocusviewer/jobs.py`, `cluster.py`, `watch.py` | launching analyses, cluster usage, rig watchdog |
| `defocusviewer/lab.py`, `rawdata.py` | rig probe and routines, raw-data storage |
| `defocusviewer/static/` | frontend: one ES module per tab |
