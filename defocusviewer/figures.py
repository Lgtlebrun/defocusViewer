"""PDF of one image panel of the viewer (the values as shown, with its colour map and range), in the house style."""
import base64
import io

import matplotlib
import numpy as np
from amkit import plot
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 400          # resolution of the image (and of anything rasterised) inside the PDF
COLOUR_MAPS = {"viridis": "viridis", "cividis": "cividis", "inferno": "inferno", "magma": "magma", "gray": "gray",
               "hot": "hot", "blues": "am_seq", "blue-red": "am_div", "purple-green": "PRGn", "brown-teal": "BrBG"}


def panel_pdf(panel: dict) -> bytes:
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
