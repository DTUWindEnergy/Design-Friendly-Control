import numpy as np


def scale_by_D(D_origin, x_origin, y_origin, D_new):
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
    cx, cy = geometric_median(x_arr, y_arr)

    x_scaled = (x_arr - cx) * s + cx
    y_scaled = (y_arr - cy) * s + cy
    return x_scaled, y_scaled


def geometric_median(x, y):
    """
    Compute the medoid (discrete geometric median) of 2D points.

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Coordinates of points.

    Returns
    -------
    center : ndarray, shape (2,)
        Coordinates of the central point (existing point minimizing
        the sum of Euclidean distances to all points).

    Notes
    -----
    O(n^2)
    """
    P = np.column_stack([x, y])  # (n, 2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)  # (n, n)
    idx = np.argmin(D.sum(axis=1))  # index of central turbine
    center = P[idx]
    return center


def hkn(scale_D=None, subset=True):  # wip
    from py_wake.site import UniformSite
    from py_wake.examples.data.dtu10mw import DTU10MW
    from py_wake.site import UniformWeibullSite

    # originally 11MW SG turbines
    wt = DTU10MW()

    def hkn_frequencies():
        """
        return array: a_weib,
                array: k_weib,
                array: sector_probabilities,

        """
        import os

        hkn_freq_path = os.path.join(DATA_PATH_HKN, "HKNB_frequencies.csv")
        freqs = np.loadtxt(
            hkn_freq_path, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4)
        )
        centered_wd_bin = freqs[:, 0]
        a_weib = freqs[:, 1]
        k_weib = freqs[:, 2]
        n_samples = freqs[:, 3]
        total_samples = n_samples.sum()  # 2012 has 366 days!
        sector_probs = n_samples / total_samples

        return a_weib, k_weib, sector_probs, centered_wd_bin

    def hkn_site(n_sectors=360, ti=0.08, seed=99, options_str=""):
        """
        hkn site object compatible with price rose
        ti: sector-wise ti
        """
        a_weib, k_weib, sector_probs, centered_wd_bin = hkn_frequencies()
        if len(sector_probs) != n_sectors:
            grid = np.linspace(0, 360, n_sectors, endpoint=True)
            a_weib = np.interp(grid, centered_wd_bin, a_weib, period=360)
            k_weib = np.interp(grid, centered_wd_bin, k_weib, period=360)
            sector_probs = np.interp(grid, centered_wd_bin, sector_probs, period=360)
            sector_probs = sector_probs / sector_probs.sum()
            # interpolate values of sector_probs
        site = UniformWeibullSite(p_wd=sector_probs, a=a_weib, k=k_weib, ti=ti)
        site.interp_method = "linear"
        site.name = "grid_site"

        # extend sector_frequencies (wd) to (wd, ws)
        def weibull_pdf(ws, A, k):
            return (k / A) * (ws / A) ** (k - 1) * np.exp(-((ws / A) ** k))

        A = site.ds["Weibull_A"]
        k = site.ds["Weibull_k"]
        ws_broadcasted = site.ds["ws"].values
        probs = np.empty((len(site.ds["wd"]), len(site.ds["ws"])))
        for i, (a, k_val, freq) in enumerate(
            zip(A.values, k.values, site.ds["Sector_frequency"].values)
        ):
            pdf = weibull_pdf(ws_broadcasted, a, k_val)
            probs[i, :] = pdf * freq
        site.ds["probs"] = (("wd", "ws"), probs)

        return site

    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, layout[0], layout[1], scale_D)
        layout_scaled = wt_x, wt_y
        return layout_scaled, site
    return layout, site, wt


def Hornsrev1Site(scale_D=None, move_mediod=True):
    from py_wake.examples.data.hornsrev1 import V80
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site
    from py_wake.examples.data.hornsrev1 import wt_x, wt_y

    wt = V80()
    site = Hornsrev1Site()
    layout = wt_x, wt_y
    if move_mediod:
        cx, cy = geometric_median(wt_x, wt_y)
        wt_x -= cx
        wt_y -= cy
        layout = wt_x, wt_y
    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, layout[0], layout[1], scale_D)
        layout_scaled = wt_x, wt_y
        return layout_scaled, site

    return layout, site, wt


def iea37(scale_D=None, n_wt=36):
    from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines

    assert n_wt in [9, 16, 36, 64]
    site = IEA37Site(n_wt)
    wt = IEA37_WindTurbines()
    layout = site.initial_position.T
    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, layout[0], layout[1], scale_D)
        layout_scaled = wt_x, wt_y
        return layout_scaled, site
    return layout, site, wt


def lillgrund(scale_D=None, move_mediod=True):
    from py_wake.examples.data.lillgrund import wt_x, wt_y, SWT23, LillgrundSite

    site = LillgrundSite()
    wt = SWT23()
    layout = wt_x, wt_y
    if move_mediod:
        cx, cy = geometric_median(wt_x, wt_y)
        wt_x -= cx
        wt_y -= cy
        layout = wt_x, wt_y
    if scale_D:
        D_origin = wt.diameter()
        wt_x, wt_y = scale_by_D(D_origin, layout[0], layout[1], scale_D)
        layout_scaled = wt_x, wt_y
        return layout_scaled, site
    return layout, site, wt
