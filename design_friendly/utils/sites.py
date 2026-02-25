from py_wake import numpy as np


def polygon_area(vertices):  # for hydesign area
    # vertices: (n, 2)
    x, y = vertices[:, 0], vertices[:, 1]
    # Shoelace'; A = 0.5 * |sum_i (x_i * y_{i+1} - y_i * x_{i+1})|
    signed_area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return 0.5 * np.abs(signed_area)


def scale_by_D(D_origin, x_origin, y_origin, D_new, center=None):
    """
    Scale wind farm coordinates when changing rotor diameter, using the
    median (per-axis) of the coordinates as the scale center.

    The scale factor is s = D_new / D_origin. If coordinates are in meters,
    this keeps spacing measured in rotor diameters (D) constant: e.g., a 7D
    spacing for D_origin becomes a 7D spacing for D_new after scaling.

    Parameters
    ----------
    D_origin : float
        Original rotor diameter.
    x_origin : array_like of shape (n,) or (...,)
        X-coordinates of turbines (1D or any shape; must match y_origin).
    y_origin : array_like of shape (n,) or (...,)
        Y-coordinates of turbines (same shape as x_origin).
    D_new : float
        New rotor diameter to scale to.

    Returns
    -------
    x_scaled : ndarray
        X-coordinates after scaling (same shape as x_origin).
    y_scaled : ndarray
        Y-coordinates after scaling (same shape as y_origin).
    """

    x_arr = np.asarray(x_origin, dtype=float)
    y_arr = np.asarray(y_origin, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x_origin and y_origin must have the same shape.")

    s = float(D_new) / float(D_origin)
    # Scale about the per-axis median
    if center is None:
        cx, cy = geometric_median(x_arr, y_arr)
    else:  # center provided as (cx, cy)
        cx, cy = float(center[0]), float(center[1])
    x_scaled = (x_arr - cx) * s + cx
    y_scaled = (y_arr - cy) * s + cy
    return x_scaled, y_scaled


def geometric_median(x, y):
    """
    Compute the medoid (discrete geometric median) of 2D points. (O(n^2))

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Coordinates of points.

    Returns
    -------
    center : ndarray, shape (2,)
        Coordinates of the central point (argmin distance to all points).
    """
    P = np.column_stack([x, y])  # (n, 2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)  # (n, n)
    idx = np.argmin(D.sum(axis=1))  # index of central turbine
    center = P[idx]
    return center


def Hornsrev1Site(scale_D=None, move_mediod=True):
    from py_wake.examples.data.hornsrev1 import V80
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site
    from py_wake.examples.data.hornsrev1 import wt_x, wt_y

    wt = V80()
    site = Hornsrev1Site()
    if move_mediod:
        # move center turbine to origin
        cx, cy = geometric_median(wt_x, wt_y)
        wt_x -= cx
        wt_y -= cy
    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, wt_x, wt_y, scale_D)
        return wt_x, wt_y, site
    return wt_x, wt_y, site, wt


def iea37(scale_D=None, n_wt=36):
    from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines

    assert n_wt in [9, 16, 36, 64]
    site = IEA37Site(n_wt)
    wt = IEA37_WindTurbines()
    layout = site.initial_position.T  # (wt_x, wt_y)
    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, layout[0], layout[1], scale_D)
        return wt_x, wt_y, site
    return layout, site, wt


def lillgrund(scale_D=None, move_mediod=True):
    from py_wake.examples.data.lillgrund import wt_x, wt_y, SWT23, LillgrundSite

    site = LillgrundSite()
    wt = SWT23()
    if move_mediod:
        # move center turbine to origin
        cx, cy = geometric_median(wt_x, wt_y)
        wt_x -= cx
        wt_y -= cy
    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, wt_x, wt_y, scale_D)
        return wt_x, wt_y, site
    return wt_x, wt_y, site, wt


def hkn(
    scale_D=284.0,
    move_mediod=True,
    return_boundary=False,
    ti=0.06,
    global_wind_atlas=True,
):
    from design_friendly.utils.sites_data import HKN_x, HKN_y, HKN_wgsx, HKN_wgsy
    from design_friendly.utils.sites_data import HKN_boundaries
    from design_friendly.utils.sites import geometric_median

    if global_wind_atlas:
        from py_wake.site.xrsite import GlobalWindAtlasSite, XRSite

        x_wgscenter, y_wgscenter = geometric_median(HKN_wgsx, HKN_wgsy)  # wgs center
        site = GlobalWindAtlasSite(
            lat=float(y_wgscenter),
            long=float(x_wgscenter),
            roughness=0.0002,
            height=184.0,
            ti=ti,
            interp_method="linear",
        )
        site = XRSite(
            site.ds.interp(wd=np.arange(0, 361), method="linear"),
            interp_method="linear",
        )  # allow sampling from wind rose at 1 deg resolution (instead of 30 deg default)
    else:
        from py_wake.site import UniformSite

        site = UniformSite()
    bound_x, bound_y = HKN_boundaries[:, 0], HKN_boundaries[:, 1]
    if move_mediod:
        # move center turbine to origin
        cx, cy = geometric_median(HKN_x, HKN_y)  # utm center
        wt_x = HKN_x - cx
        wt_y = HKN_y - cy
        bound_x = bound_x - cx
        bound_y = bound_y - cy
    else:
        wt_x, wt_y = HKN_x, HKN_y
        cx, cy = 0.0, 0.0
    if scale_D:
        diam_original = 200.0  # SG11MW
        wt_x, wt_y = scale_by_D(diam_original, wt_x, wt_y, scale_D, center=(cx, cy))
        if return_boundary:
            bound_x, bound_y = scale_by_D(
                diam_original, bound_x, bound_y, scale_D, center=(cx, cy)
            )
    if return_boundary:
        return wt_x, wt_y, site, np.column_stack([bound_x, bound_y])
    return wt_x, wt_y, site


def plot_site(x, y, bounds=None, center=True, save_fig=None):
    import matplotlib.pyplot as plt

    if center:
        cx, cy = geometric_median(x, y)
    else:
        cx, cy = 0.0, 0.0
    x -= cx
    y -= cy
    fig, ax = plt.subplots(figsize=(4, 5))
    n_wt = len(x)
    ax.scatter(x, y, zorder=5, marker="2", label=f"Turbines ({n_wt})", s=80)
    for i, n in enumerate(zip(x, y)):
        ax.text(n[0], n[1], i)
    if bounds is not None:
        bounds[:, 0] -= cx
        bounds[:, 1] -= cy
        ax.plot(
            bounds[:, 0],
            bounds[:, 1],
            lw=2,
            zorder=3,
            label="Boundary",
            ls="--",
            c="gray",
        )
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_fig is not None:
        fig.savefig(f"{save_fig}.pdf")
    plt.show()


if __name__ == "__main__":
    x, y, site, b = hkn(return_boundary=True)
    plot_site(x, y, b)
    x, y, site, b = hkn(return_boundary=True, scale_D=None)
    plot_site(x, y, b)
