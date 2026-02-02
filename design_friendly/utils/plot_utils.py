import warnings
from functools import wraps

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
from autograd import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase, HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter


def deprecated(reason=None, *, stacklevel=2):
    def _decorator(func):
        name = func.__qualname__
        msg = f"{name}() is deprecated."
        if reason:
            msg += f" {reason}"

        @wraps(func)
        def _wrapper(*args, **kwargs):
            warnings.warn(msg, category=FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


@deprecated("Use plot_ws_diff_field() instead. Supports both the fm and its diffs")
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
    yaw_deg = yaw_deg.squeeze()
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
        vcenter = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    Xg = getattr(fm, "x", np.arange(Z.shape[1]))
    Yg = getattr(fm, "y", np.arange(Z.shape[0]))
    cs = ax.contourf(
        Xg, Yg, Z, levels=np.linspace(vmin, vmax, levels), cmap=cmap, norm=norm
    )
    ax.scatter(x, y, c="k", s=D / 50)
    if add_colorbar:
        cbar = fig.colorbar(cs, ax=ax, pad=0.005, shrink=0.5, aspect=20)
        cbar.set_label(r"$\Delta$ Wind speed [m/s]")
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    segs = []
    if yaw_deg is None:
        for xi, yi in zip(x, y):
            segs.append([[xi, yi - D / 2], [xi, yi + D / 2]])
    else:
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
        r = 0.07 * span  # ~7% of plot size
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


def lut_heatmap(rose_jm):
    """
    this is only dataframe-like look-up-table
    rose_jm: table-like with columns [WD_bin (degrees, 0-359), WS_bin, yaw_0]
    """
    wd = rose_jm[:, 0] % 360.0  # WD-major (assumed from np.arange)
    ws = rose_jm[:, 1]  # WS-minor (assumed from np.arange)
    yaw = rose_jm[:, 2]

    wd0 = wd.min()
    ws0 = ws.min()
    wd_step = wd[1] - wd[0] if len(wd) > 1 else 360.0
    ws_step = ws[1] - ws[0] if len(ws) > 1 else 1.0

    n_wd = int(round((wd.max() - wd0) / wd_step)) + 1
    n_ws = int(round((ws.max() - ws0) / ws_step)) + 1
    wss = ws0 + ws_step * np.arange(n_ws)
    wd_idx = ((wd - wd0) / wd_step).astype(int)
    ws_idx = ((ws - ws0) / ws_step).astype(int)

    # 2D LUT: rows = WS bins, cols = WD bins
    C = np.full((n_ws, n_wd), np.nan)
    C[ws_idx, wd_idx] = yaw

    theta_edges = np.deg2rad(wd0 - wd_step / 2.0 + wd_step * np.arange(n_wd + 1))
    r_edges = ws0 - ws_step / 2.0 + ws_step * np.arange(n_ws + 1)
    Theta, R = np.meshgrid(theta_edges, r_edges)

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection="polar"))
    pcm = ax.pcolormesh(Theta, R, C, cmap="plasma", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Yaw Offset($^o$)")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_ylim(r_edges[0], r_edges[-1])
    ax.set_yticks(wss)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    plt.tight_layout()
    plt.show()
    return fig, ax


def lut_3d_heatmap(rose_ilk, wds=None, wss=None, vmin=-15, vmax=15, ymax=16):
    """
    PyWake style heatmap (actually this is lk and a single turbine yaw offset (wd,ws) should be provided)
    if in doubt, provide the wss and wds
    """
    if wds is None or wss is None:
        wds = np.arange(0, 360, 1)
        wss = np.arange(3, 12, 1)
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection="polar"))
    theta, r = np.meshgrid(np.deg2rad(wds), wss)
    C = np.asarray(rose_ilk)
    if C.shape == (len(wds), len(wss)):  # (n_wds, n_wss) -> transpose to (n_wss, n_wds)
        C = C.T

    c = ax.pcolormesh(
        theta, r, C, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax
    )  # 'BrBG'
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Yaw Offset($^o$)")
    ax.set_theta_zero_location("N")  # Set 0 degrees to point north
    ax.set_theta_direction(-1)  # Set clockwise direction
    ax.set_rlabel_position(135)
    plt.tight_layout()
    ax.set_ylim(0, ymax)
    plt.show()


def plot_turbine_graph(
    data,
    ax=None,
    annotate_nodes=False,
    annotate_edges=True,
    edge_attr_fmt="({:.0f})",
    curvature=0.18,
    node_size=60,
    arrow_lw=1.0,
    arrow_alpha=0.8,
    label_offset_frac=0.02,
    title_from_meta=True,
):
    """Plot a directed turbine graph.

    Parameters
    ----------
    data : object
        pos : (N, 2)
        edge_index : (2, E)
        edge_attr : (E, F) required if annotate_edges=True
        meta : dict required if title_from_meta=True
    """
    pos = data.pos  # (N, 2)
    edge_index = data.edge_index  # (2, E)
    edge_attr = data.edge_attr if annotate_edges else None  # (E, F)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size)
    if annotate_nodes:
        for i in range(pos.shape[0]):
            ax.text(pos[i, 0], pos[i, 1], f"{i}", fontsize=9, ha="left", va="bottom")
    src = edge_index[0, :]
    dst = edge_index[1, :]
    for k, (i, j) in enumerate(zip(src, dst)):
        p0 = pos[i]
        p1 = pos[j]
        rad = curvature if i < j else -curvature
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=arrow_lw,
                alpha=arrow_alpha,
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        if annotate_edges:
            mid = 0.5 * (p0 + p1)
            d = p1 - p0
            L = float(np.hypot(d[0], d[1]))
            if L > 0.0:
                n = np.array([-d[1], d[0]]) / L
                mid = mid + np.sign(rad) * (label_offset_frac * L) * n
            ax.text(
                mid[0],
                mid[1],
                edge_attr_fmt.format(edge_attr[k, 0]),
                fontsize=8,
                ha="center",
                va="center",
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    if title_from_meta:
        ax.set_title(" | ".join(f"{k}={v}" for k, v in data.meta.items()))
    return fig, ax


def plot_ws_diff_field(
    ws_eff_fm,
    X_fix=None,
    Y_fix=None,
    yaw_deg=None,
    *,
    ax=None,
    D=284.0,
    bar_length_m=None,
    levels=121,
    cmap="Blues_r",
    cbar_label=r"Wind speed (m/s)",  # $\Delta$
    tick_step_D=5,
    yaw_text_offset_m=(200.0, 150.0),
    figsize=(10, 10),
    save_path=None,
    dpi=300,
    show=True,
    # sizing controls
    label_size=15,
    tick_label_size=13,
    cbar_label_size=15,
    cbar_tick_label_size=13,
    yaw_text_size=14,
    lw_axes=0.8,
    lw_rotor=2.0,
    scatter_s=15,
    use_normslop=False,
):
    """Should work well with PyWake flow"""
    bar_length_m = D if bar_length_m is None else bar_length_m

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    field = ws_eff_fm.squeeze()
    x = field["x"].to_numpy()
    y = field["y"].to_numpy()
    z = field["WS_eff"].to_numpy()  # fixes `.values()` / ValuesView issues
    cf = dict(extend="both", cmap=cmap)

    if use_normslop:
        max_abs = np.nanmax(np.abs(z))
        cf["levels"] = np.linspace(-max_abs, max_abs, int(levels))
        cf["norm"] = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    else:
        cf["levels"] = int(levels)

    cs = ax.contourf(x, y, z, **cf)

    cbar = fig.colorbar(cs, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_tick_label_size)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    if (X_fix is not None) and (Y_fix is not None) and (yaw_deg is not None):
        segs = []
        for xi, yi, g in zip(X_fix, Y_fix, yaw_deg):
            th = np.deg2rad(g)
            dx = (bar_length_m / 2.0) * np.sin(th)
            dy = (bar_length_m / 2.0) * np.cos(th)
            segs.append([[xi - dx, yi - dy], [xi + dx, yi + dy]])

        ax.add_collection(
            LineCollection(
                segs,
                colors="black",
                linewidths=lw_rotor,
                capstyle="round",
                zorder=4,
            )
        )
        ax.scatter(X_fix, Y_fix, s=scatter_s, c="k", zorder=5)

        ox, oy = yaw_text_offset_m
        pe_outline = [pe.withStroke(linewidth=3, foreground="white")]
        for xi, yi, g in zip(X_fix, Y_fix, yaw_deg):
            ax.text(
                xi + ox,
                yi + oy,
                f"{int(np.round(g, 0)):.0f}" + r"$^\circ$",
                ha="center",
                va="bottom",
                zorder=10,
                fontsize=yaw_text_size,
                path_effects=pe_outline,
            )

    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_step_D * D))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_step_D * D))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / D:.0f}D"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / D:.0f}D"))

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (D)", fontsize=label_size)
    ax.set_ylabel("Y (D)", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size, direction="out")
    for sp in ax.spines.values():
        sp.set_linewidth(lw_axes)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax, cbar
