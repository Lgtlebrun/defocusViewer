"""PDF of one image panel of the viewer (the values as shown, with its colour map and range), in the house style."""
import base64
import io
import threading

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from amkit import plot
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 400          # resolution of the image (and of anything rasterised) inside the PDF
PYPLOT = threading.Lock()     # pyplot and the style are global: one figure at a time across the server's threads
FIBER_QUANTITIES = {      # column: (label, unit, column of its sigma)
    "cx": ("spot centre x on the sensor", "px", None), "cy": ("spot centre y on the sensor", "px", None),
    "x0": ("centre x in the crop", "px", None), "y0": ("centre y in the crop", "px", None),
    "c4": ("c4 (defocus)", "waves", "sigma_c4"), "delta_z": ("delta_z, raw", "um", "sigma_delta_z"),
    "dz_pooled": ("delta_z calibrated (pooled)", "um", None), "dz_free": ("delta_z calibrated (free)", "um", None),
    "x_obj_pooled": ("object x (pooled calibration)", "um", None), "y_obj_pooled": ("object y (pooled calibration)", "um", None),
    "x_obj_free": ("object x (free calibration)", "um", None), "y_obj_free": ("object y (free calibration)", "um", None),
    "r": ("source radius r", "px", None), "c5": ("c5", "waves", None), "c6": ("c6", "waves", None),
    "c7": ("c7", "waves", None), "c8": ("c8", "waves", None), "residual_frac": ("residual_frac", "", None),
    "texp": ("exposure", "us", None),
}
X_AXES = {"time": "time", "minutes": "minutes since the first spot", "x_position": "stage position (um)", "index": "spot index"}
ROUTINE_MARKERS = {"train_burnin": "x", "train": "o", "train2": "s", "stability": ".", "detection": ".", "positioner": ".",
                   "train_end": "D"}
COLOUR_MAPS = {"viridis": "viridis", "cividis": "cividis", "inferno": "inferno", "magma": "magma", "gray": "gray",
               "hot": "hot", "blues": "am_seq", "blue-red": "am_div", "purple-green": "PRGn", "brown-teal": "BrBG"}


def panel_pdf(panel: dict) -> bytes:
    with PYPLOT:
        return _panel_pdf(panel)


def _panel_pdf(panel: dict) -> bytes:
    """PDF bytes of a panel: {shape, data (base64 float32), coords, zrange, colour_map, reverse, title, colorbar_label,
    xlabel, ylabel, ghost_box}."""
    plot.setup()
    values = np.frombuffer(base64.b64decode(panel["data"]), dtype=np.float32).reshape(panel["shape"])
    colour_map = matplotlib.colormaps[COLOUR_MAPS.get(panel.get("colour_map"), "viridis")]
    if panel.get("reverse"):
        colour_map = colour_map.reversed()
    coords = panel.get("coords")
    extent = None
    if coords:
        half = (coords[1] - coords[0]) / 2
        extent = (coords[0] - half, coords[-1] + half, coords[-1] + half, coords[0] - half)
    fig, ax = plot.figure("single", ratio=0.85)
    zmin, zmax = panel.get("zrange") or (None, None)
    image = ax.imshow(np.ma.masked_invalid(values), origin="upper", cmap=colour_map, vmin=zmin, vmax=zmax, extent=extent,
                      interpolation="nearest")
    fig.colorbar(image, ax=ax, label=panel.get("colorbar_label", ""))
    box = panel.get("ghost_box")
    if box:
        ax.add_patch(Rectangle((box["x0"] - 0.5, box["y0"] - 0.5), box["side"], box["side"], fill=False, ec="#d03b3b", lw=0.8,
                               ls=":"))
    ax.set_title(panel.get("title", ""), fontsize=6)
    ax.set_xlabel(panel.get("xlabel", "x (px)"))
    ax.set_ylabel(panel.get("ylabel", "y (px)"))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", dpi=DPI)
    plt.close(fig)
    return buffer.getvalue()


def fiber_figure(table: pd.DataFrame, quantities, fibers, routines, x_axis: str = "time", relative: bool = False,
                 average: int = 1, errors: bool = False, title: str = "", fmt: str = "png", dpi: int = 150) -> bytes:
    """Figure (png, pdf or svg bytes) of fitted quantities of a run's spots, one row per quantity sharing the x axis: one
    colour per fiber (fixed palette order), one marker per routine; relative subtracts each fiber's median, average is a
    centred moving average over that many spots, errors draws +-1 sigma where the fit gives one (of the moving average)."""
    with PYPLOT:
        plot.setup(grid=True)
        all_fibers = sorted(table["fiber"].unique())
        colour = {f: plot.CATEGORICAL[i % len(plot.CATEGORICAL)] for i, f in enumerate(all_fibers)}
        start = table["timestamp"].min()
        fig, axes = plot.figure("double", ratio=0.24, nrows=len(quantities), sharex=True, squeeze=False, layout="constrained")
        for ax, quantity in zip(axes[:, 0], quantities):
            label, unit, sigma = FIBER_QUANTITIES[quantity]
            for fiber in fibers:
                of_fiber = table[(table["fiber"] == fiber) & table["routine"].isin(routines) & table[quantity].notna()]
                offset = of_fiber[quantity].median() if relative and len(of_fiber) else 0.0
                for routine in routines:
                    rows = of_fiber[of_fiber["routine"] == routine]
                    if rows.empty:
                        continue
                    x = {"time": rows["timestamp"], "minutes": (rows["timestamp"] - start).dt.total_seconds() / 60,
                         "x_position": rows.get("x_position"), "index": rows.index.to_series()}[x_axis]
                    order = np.argsort(x.to_numpy(), kind="stable")
                    x = x.iloc[order]
                    y = (rows[quantity].iloc[order] - offset).rolling(max(1, average), center=True, min_periods=1).mean()
                    marker = ROUTINE_MARKERS.get(routine, "v")
                    ax.plot(x, y, marker=marker, ms=2.2 if marker == "." else 2.8, lw=0.5, color=colour[fiber],
                            markeredgewidth=0.6)
                    if errors and sigma and sigma in rows and rows[sigma].notna().any():
                        window = max(1, average)      # error of the moving average: sigma / sqrt(n), the n spots independent
                        yerr = rows[sigma].iloc[order].rolling(window, center=True, min_periods=1).mean() / np.sqrt(window)
                        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=colour[fiber], elinewidth=0.4, alpha=0.25)
            ax.set_ylabel(label + (f"\n({unit})" if unit else ""), fontsize=7)
            ax.tick_params(labelsize=7)
        bottom = axes[-1, 0]
        bottom.set_xlabel(X_AXES[x_axis], fontsize=8)
        if x_axis == "time":
            bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        handles = [Line2D([], [], color=colour[f], lw=1.2, label=f) for f in fibers]
        handles += [Line2D([], [], color=plot.MUTED, marker=ROUTINE_MARKERS.get(r, "v"), lw=0, label=r) for r in routines]
        axes[0, 0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(handles), 10), fontsize=6,
                          frameon=False, columnspacing=1.0, handlelength=1.5)
        notes = ["each fiber minus its median"] * relative + [f"moving average over {average} spots"] * (average > 1)
        if title or notes:
            fig.suptitle(" · ".join([title, *notes]).strip(" ·"), fontsize=7)
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()
