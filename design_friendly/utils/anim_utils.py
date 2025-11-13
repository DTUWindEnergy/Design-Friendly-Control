import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image


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