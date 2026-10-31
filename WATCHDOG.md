# Rig watchdog

`defocusviewer watch` follows pos-control. For every StabilityTest run that finishes on the rig after the watchdog
started:

1. It fetches the run, with a "fetch (watchdog)" job in the Jobs tab.
2. It makes the variant `<run>_tophatbl-circle`, with these settings:
   - model `tophat_bl@circle` (circular aperture)
   - ghost mask on
   - `fit.lbda_range_nm` [650, 750]
   - every other setting from the run's own config
3. It runs the full analysis of the variant on the two P100s of farm-gpu1, in float64. This shows as a "watchdog: full
   analysis" job.

The analyses run one at a time, oldest run first. A queued run starts only when both of the following are true:
- No GPU analysis runs on farm-gpu1. This counts viewer jobs, and also `run_stability` or `jointfit` processes started
  by other sessions.
- Both P100s are idle.

When the watchdog is first started, it records every run already finished on the rig as a baseline and never touches
those runs.

The code is in `defocusviewer/watch.py`. Its state, log and on/off switch are in `/disk/users/llebru/defocusViewer/watch/`:

| file | what |
|---|---|
| `enabled` | the switch: while it exists, the watchdog runs, and cron restarts it after a reboot |
| `watch.log` | what it did and what it is waiting for |
| `state.json` | the baseline and each run it handles (`fetching`, `queued`, `launched`, `skipped`, `failed`) |

It runs in tmux session `defocusviewer-watch` on farm-gpu1. Every 5 min, the cron line `defocusviewer ensure`, which
also keeps the viewer up, restarts it if `enabled` exists and the session is gone.

## Stop it

On farm-gpu1:

```bash
~/Defocus/.venv/bin/defocusviewer watch stop
```

This deletes `enabled` and kills the tmux session, so cron does not bring the watchdog back.

If the command is not available, do the same by hand:

```bash
rm /disk/users/llebru/defocusViewer/watch/enabled
tmux kill-session -t defocusviewer-watch
```

Deleting `enabled` alone is enough as well: the watchdog exits within a minute.

**Stopping the watchdog does not stop the jobs it already started.** A fetch or an analysis that is running keeps
going. To stop one:
- cancel it in the viewer's Jobs tab, or
- run `kill -TERM -- -<pid>` on farm-gpu1, where `<pid>` is the `pid` in `/disk/users/llebru/defocusViewer/jobs/<job id>.json`.

## Check it

```bash
~/Defocus/.venv/bin/defocusviewer watch status        # switch, tmux session, each run and its job
tail -f /disk/users/llebru/defocusViewer/watch/watch.log
```

## Start it again

```bash
~/Defocus/.venv/bin/defocusviewer watch start
```

A restart keeps `state.json`. The baseline and the runs it already handled stay as they were, so nothing is fetched or
analysed twice.

To start over with a new baseline (every run finished on the rig at that moment), stop the watchdog, delete
`state.json`, then start it.

To have one run handled again, delete that run's entry from `state.json` while the watchdog is stopped.
