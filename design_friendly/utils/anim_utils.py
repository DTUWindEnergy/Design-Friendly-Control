import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image
from design_friendly.utils.plot_utils import pretty_flowmap


def save_mp4_from_figs(
    figs_or_axes,
    out_path="anim.mp4",
    fps=2,
    dpi=120,
    target_size=None,
):
    frames = []
    for item in figs_or_axes:
        fig = item.figure if hasattr(item, "figure") else item
        if dpi is not None:
            fig.set_dpi(dpi)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[..., :3]  # drop alpha
        if target_size is not None and (frame.shape[1], frame.shape[0]) != (
            target_size[0],
            target_size[1],
        ):
            frame = np.array(Image.fromarray(frame).resize(target_size, Image.BILINEAR))
        frames.append(frame)

    if not frames:
        raise ValueError("No frames provided.")

    H, W = frames[0].shape[:2]
    # Create a single canvas to display frames and grab them
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.set_axis_off()
    im = ax.imshow(frames[0], animated=True)

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt",
            "yuv420p",  # ppt compatible
            "-movflags",
            "+faststart",
        ],
        metadata=None,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with writer.saving(fig, out_path, dpi):
        for f in frames:
            im.set_data(f)
            writer.grab_frame()

    plt.close(fig)
    return out_path


def animate_flowmap_over_windrose(
    sim_base,  # report uplift, x, y
    sim_gnn,  # yaw setpoints
    fmap_diff,
    fix_wd=None,
    fix_ws=None,
    TI=0.6,
    *,
    D=284,
    clim=(-1.5, 1.5),
    filename="FLOWS.mp4",
    fps=5,
    dpi=200,
    # target_size=(1280, 720),
    display_inline=False,
    resolution=100,
):
    plt.close("all")
    wds = sim_gnn.wd.values
    wss = sim_gnn.ws.values
    x_ = sim_gnn.isel(wd=0, ws=0).squeeze().x.values
    y_ = sim_gnn.isel(wd=0, ws=0).squeeze().y.values

    if fix_wd:
        wds = np.atleast_1d(fix_wd)
        assert fix_wd in wds
    elif fix_ws:
        wss = np.atleast_1d(fix_ws)
        assert fix_ws in wss
    else:
        raise ValueError(f"fix wd or ws to animate")

    yaws = sim_gnn.yaw
    figs = []
    for i, wd_ in enumerate(wds):
        for j, ws_ in enumerate(wss):
            yaw_subset = yaws.sel(wd=wd_, ws=ws_)
            base_power = float(sim_base.Power.sum("wt").sel(wd=wd_, ws=ws_)) / 1e6
            gnn_power = float(sim_gnn.Power.sum("wt").sel(wd=wd_, ws=ws_)) / 1e6
            uplift = round(100 * (gnn_power - base_power) / base_power, 2)
            # ws_base = fmap_base.WS_eff.sel(wd=wd_, ws=ws_).squeeze()
            # ws_gnn = fmap_gnn.WS_eff.sel(wd=wd_, ws=ws_).squeeze()
            ws_diff = fmap_diff.WS_eff.sel(wd=wd_, ws=ws_).squeeze()
            ws_rotor_eff = sim_gnn.WS_eff.sel(wd=wd_, ws=ws_).squeeze().round(1)
            fig, _ = pretty_flowmap(
                ws_diff,
                x_,
                y_,
                yaw_subset,
                yaw_deg=yaw_subset,
                wd_deg=wd_,
                D=D,
                title=f"WS:{ws_}, WD:{wd_}, Uplift:{uplift:.2f}% ({base_power:.2f}MW->{gnn_power:.2f}MW)",
                clim=clim,
                show=False,
                fsize=(8, 6),
            )
            figs.append(fig)

    video_path = save_mp4_from_figs(
        figs,
        filename,
        fps=fps,
        dpi=dpi,  # , target_size=target_size
    )
    print("saved:", video_path)

    if display_inline:
        try:
            from IPython.display import Video, display

            display(Video(video_path, embed=True))
        except Exception:
            pass

    return video_path, figs