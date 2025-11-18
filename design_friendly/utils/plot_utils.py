import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase, HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import warnings
import xarray as xr

BASE_FONTSIZE = 14  # or 16, 18, tweak to taste

mpl.rcParams.update(
    {
        "font.size": BASE_FONTSIZE,  # default text
        "axes.titlesize": BASE_FONTSIZE * 1.2,  # axes titles
        "axes.labelsize": BASE_FONTSIZE,  # x/y labels
        "xtick.labelsize": BASE_FONTSIZE * 0.9,
        "ytick.labelsize": BASE_FONTSIZE * 0.9,
        "legend.fontsize": BASE_FONTSIZE * 0.9,
        "figure.titlesize": BASE_FONTSIZE * 1.3,
    }
)


def pretty_flowmap(
    fm,
    x,
    y,
    *pts,
    yaw_deg=None,
    wd_deg=None,
    D=284.0,
    cmap="RdBu",
    levels=121,
    fsize=(6, 6),
    title=None,
    show=True,
    tick_step_D=5,
    ax=None,
    clim=None,
    add_colorbar=True,
):
    if not isinstance(fm, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"Expected xarray.DataArray or xarray.Dataset, got {type(fm).__name__}"
        )
    if ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=fsize)

    Z = np.asarray(getattr(fm, "values", fm))
    # A = float(np.max(np.abs(Z)))
    # norm = TwoSlopeNorm(vmin=-A, vcenter=0.0, vmax=A)
    if clim is None:
        A = float(np.max(np.abs(Z))) if Z.size else 0.0
        vmin, vmax = -A, A
    else:
        vmin, vmax = float(clim[0]), float(clim[1])
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    Xg = getattr(fm, "x", np.arange(Z.shape[1]))
    Yg = getattr(fm, "y", np.arange(Z.shape[0]))
    cs = ax.contourf(
        Xg, Yg, Z, levels=np.linspace(vmin, vmax, levels), cmap=cmap, norm=norm
    )
    ax.scatter(x, y, c="k", s=D/50)

    if add_colorbar:
        cbar = fig.colorbar(cs, ax=ax, pad=0.005, shrink=0.5, aspect=20)
        cbar.set_label(r"$\Delta$ Wind speed [m/s]")
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    segs = []
    if yaw_deg is None:
        for xi, yi in zip(x, y):
            segs.append([[xi, yi - D / 2], [xi, yi + D / 2]])
    else:
        # wd_term = 0.0 if wd_deg - 90 is None else wd_deg - 90
        # wd_term = 0.0 if wd_deg is None else (wd_deg - 90.0)
        # wd_term = wd_deg-90
        for xi, yi, ya in zip(x, y, yaw_deg):
            th = np.deg2rad(-ya) + np.deg2rad(wd_deg - 90)
            dx, dy = (D / 2) * np.sin(th), (D / 2) * np.cos(th)
            segs.append([[xi - dx, yi - dy], [xi + dx, yi + dy]])
    lc = LineCollection(
        segs, colors="black", linewidths=2.0, capstyle="round", zorder=4
    )
    ax.add_collection(lc)


    class HandlerVerticalLine(HandlerLine2D):
        def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
            xmid = xd + w / 2
            ln = Line2D(
                [xmid, xmid],
                [yd, yd + h],
                color=orig.get_color(),
                linewidth=orig.get_linewidth(),
            )
            ln.set_transform(trans)
            return [ln]

    turb_proxy = Line2D([0], [0], color="black", linewidth=2.0)

    # Per-turbine labels
    if pts:
        pe_outline = [pe.withStroke(linewidth=2, foreground="white")]
        R = 0.7 * D
        K = len(pts)
        offsets = [
            (R * np.cos(2 * np.pi * k / K), R * np.sin(2 * np.pi * k / K))
            for k in range(K)
        ]
        pts = [np.asarray(p) for p in pts]
        for i, (xi, yi) in enumerate(zip(x, y)):
            for k, arr in enumerate(pts):
                if i < len(arr) and arr[i] is not None:
                    dx, dy = offsets[k]
                    ax.text(
                        xi + dx,
                        yi + dy,
                        int(arr[i]),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        zorder=10,
                        path_effects=pe_outline,
                    )

    # Wind arrow with circle badge
    arrow_proxy = None
    if wd_deg is not None:
        theta = np.deg2rad(270 - wd_deg)
        # r = D / 3  # badge radius and arrow length
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        span = min(xmax - xmin, ymax - ymin)  # shorter axis in data units
        r = 0.07 * span  # ~7% of plot size; tweak factor as you like

        # place badge inside current axes (top-left), using a small margin = r
        x0 = xmin + 1.2 * r
        y0 = ymin + 1.2 * r

        # badge
        ax.add_patch(
            Circle(
                (x0, y0), radius=r, fc="white", ec="0.6", lw=1.0, alpha=0.95, zorder=18
            )
        )

        # main arrow (outside badge)
        x1, y1 = x0 + r * np.cos(theta), y0 + r * np.sin(theta)
        p0, p1 = ax.transData.transform((x0, y0)), ax.transData.transform((x1, y1))
        Lpx = float(np.hypot(*(p1 - p0)))
        ms = float(np.clip(Lpx * 0.10, 10, 60))
        lw = float(np.clip(Lpx * 0.012, 1.5, 4.0))
        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            fc="black",
            ec="black",
            zorder=20,
            joinstyle="round",
            capstyle="round",
            label="Wind direction (from)",
        )
        arr.set_path_effects([pe.withStroke(linewidth=lw + 1.5, foreground="white")])
        ax.add_patch(arr)

        # legend arrow rotated to same direction
        arrow_proxy = FancyArrowPatch((0, 0), (1, 0), fc="none", ec="black")

        class HandlerArrow(HandlerBase):
            def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
                x_a, x_b = xd + 0.15 * w, xd + 0.85 * w
                y = yd + 0.5 * h
                patch = FancyArrowPatch(
                    (x_a, y),
                    (x_b, y),
                    arrowstyle="-|>",
                    mutation_scale=fs * 1.2,
                    linewidth=1.8,
                    fc="black",
                    ec="black",
                    joinstyle="round",
                    capstyle="round",
                )
                from matplotlib.transforms import Affine2D

                ang = np.rad2deg(theta)
                patch.set_transform(
                    Affine2D().rotate_deg_around((x_a + x_b) / 2.0, y, ang) + trans
                )
                return [patch]

    # legend with pointy wind direction
    handles, labels, hmap = (
        [turb_proxy],
        ["Turbines"],
        {turb_proxy: HandlerVerticalLine()},
    )
    if arrow_proxy is not None:
        handles += [arrow_proxy]
        labels += ["Wind direction"]
        hmap[arrow_proxy] = HandlerArrow()
    leg = ax.legend(
        handles=handles,
        labels=labels,
        handler_map=hmap,
        loc="upper right",
        handlelength=2.0,
        handleheight=1.0,
        # zorder=100,
    )
    leg.set_zorder(100)
    # uniform axis D's
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_step_D * D))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step_D * D))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / D:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / D:.0f}"))
    ax.set_xlabel("X [D]")
    ax.set_ylabel("Y [D]")
    ax.set_aspect("equal", "box")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def lut_heatmap(rose_ilk):
    """
    rose_ilk: array-like with columns [WD_bin (degrees, 0-359), WS_bin (e.g., 3-25), yaw_0]
    """
    wd = rose_ilk[:, 0]
    ws = rose_ilk[:, 1]
    yaw = rose_ilk[:, 2]
    ws_step = 1
    wd_step = 1

    wsin = 1
    wsout = 14
    wds = np.arange(0, 360, wd_step)  # wind direction centers (degrees)
    wss = np.arange(wsin, wsout, ws_step)  # wind speed centers
    theta_edges = np.deg2rad(np.arange(0, 361, wd_step))  # 0..360 deg edges in radians
    r_edges = np.arange(wsin - 0.5, wsout, wd_step)  # wind speed edges (3±0.5)
    warnings.warn("lut_heatmap assumes rose_ilk ws columns starts at 3m/s????")
    sum_grid = np.zeros((len(wss), len(wds)), dtype=float)
    count_grid = np.zeros_like(sum_grid)

    wd_idx = (np.floor(wd) % 360).astype(int)
    ws_idx = (np.floor(ws) - wsin).astype(int)  # since wss starts at 3

    valid = (ws_idx >= 0) & (ws_idx < len(wss))
    for wi, di, val in zip(ws_idx[valid], wd_idx[valid], yaw[valid]):
        sum_grid[wi, di] += val
        count_grid[wi, di] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        C = sum_grid / count_grid  # shape (23, 360); missing bins are NaN

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection="polar"))
    Theta, R = np.meshgrid(theta_edges, r_edges)
    pcm = ax.pcolormesh(Theta, R, C, cmap="plasma", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Yaw Offset [%]")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_ylim(wss[0], wss[-1] + 0.5)
    ax.set_yticks(wss)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    plt.tight_layout()
    plt.show()
    return fig, ax


def lut_3d_heatmap(rose_ilk, wds=None, wss=None, vmin=-15, vmax=15):
    plt.ion()  # Enable interactive mode

    if wds is None or wss is None:
        wds = np.arange(0, 360, 1)
        wss = np.arange(3, 12, 1)
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection="polar"))
    theta, r = np.meshgrid(np.deg2rad(wds), wss)
    C = np.asarray(rose_ilk)
    if C.shape == (len(wds), len(wss)):  # (n_wds, n_wss) -> transpose to (n_wss, n_wds)
        C = C.T

    c = ax.pcolormesh(
        theta, r, C, cmap="coolwarm", shading="nearest", vmin=vmin, vmax=vmax
    )  # 'BrBG'
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Yaw Offset[%]")
    ax.set_theta_zero_location("N")  # Set 0 degrees to point north
    ax.set_theta_direction(-1)  # Set clockwise direction
    # Adjust radial label position for better readability
    ax.set_rlabel_position(135)
    plt.tight_layout()
    ax.set_ylim(0, 16)
    plt.show()
    plt.ioff()  # Disable interactive mode


# lut_3d_heatmap(hkn_yaw[0])
