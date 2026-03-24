from design_friendly.models import models_filepath
from py_wake import numpy as np

from design_friendly.utils.misc import log_execution_time
from design_friendly.utils.pred import predict_torchscript, torchscript_to_lut
from design_friendly.utils.graph import graph_maker
from design_friendly.utils.vjp import prepare_from_graphs, make_dP_dz_inflowgrid, gradP_vjp_xy_inflowgrid_prepared


@log_execution_time
def easy(
    x,
    y,
    wd,
    ws,
    TI,
    model_path=models_filepath + "torchscript26a.pt",
    num_threads=0,
    batch_size=512,
    lut=True,
):
    """Predict yaw angles using the unified graph_maker().

    Parameters
    ----------
    x, y : array-like
        ''(n_wt,)'' fixed layout, or ''(n_cases, n_wt_max)'' NaN-padded for sequential mode.
    wd, ws : array-like
        ''(n_wd,)'' / ''(n_ws,)'' grid axes when ''lut=True''.
        ''(n_cases,)'' paired values when ''lut=False''.
    TI : float
        Turbulence intensity (scalar).
    lut : bool, default True
        ''True'' - wdxws cartesian product (LUT).
        ''False'' - 1:1 pairing; sequential mode auto-detected when x is 2D.
    """
    graphs = graph_maker(
        x=x,
        y=y,
        wd=wd,
        ws=ws,
        TI=TI,
        lut=lut,
        num_threads=num_threads,
        connectivity="wake_aware",
    )
    results = predict_torchscript(model_path, graphs, batch_size, "array")

    if lut:
        results = torchscript_to_lut(results, wd, ws)
        # results = results[:, :, :]
    elif np.asarray(x).ndim == 1:
        # time-series
        results = results.T.squeeze()
    else:
        # sequential
        results = results.squeeze()

    return results


@log_execution_time
def easy_grad(
    *,
    wf_model,
    coords,
    ts_path=models_filepath + "torchscript26a.pt",
    connectivity="wake_aware",
    edge_uv_cols=(0, 1),
    rotor_diameter=284.0,
    k_wake=0.40,
    gamma_col=-1,
    uv_scale=1.0,
    time=False,
):
    """
    returns an easy function computing full chain-rule gradients dP/d(x,y)
    for a FIXED layout, supporting either:
      - time series (paired wd_t, ws_t), or
      - LUT axes (wd[:], ws[:]) where a (wd,ws) grid is formed.
    PyWake partials (make_dP_dz_inflowgrid) are cached.

    Parameters
    ----------
    wf_model : object
        PyWake wind farm model used to construct inflow-grid partials.
    ts_path : str
        TorchScript model path used in the VJP.
    coords : array_like or tuple[array_like, array_like]
        Fixed turbine coordinates. (n_wt, 2)
    connectivity : str
        Graph connectivity mode (e.g. "wake_aware").
    edge_uv_cols : tuple[int, int]
        Edge_attr columns holding (u, v) deltas for VJP.
    rotor_diameter : float
        Rotor diameter used by "wake_aware".
    k_wake : float
        Wake expansion slope used by "wake_aware".
    gamma_col : int
        Output column index in the TorchScript model corresponding to yaw (gamma).
    uv_scale : float
        Scale factor applied to (u,v) deltas inside the VJP pipeline.

    Returns
    -------
    grad_fn : callable
        grad_fn(wd, ws, TI, return_gamma=False)

        Modes inferred from wd/ws:
          - scalar/scalar -> single case
          - 1D/1D same length + time=True -> time series (paired)
          - 1D/1D -> LUT grid (cross-product)

        Outputs:
          - single: dP_dxy (n_wt, 2)
          - time:   dP_dxy (n_wt, T, 2)
          - LUT:    dP_dxy (n_wt, n_wd, n_ws, 2)
        If return_gamma=True, also returns gamma with the same shape minus the last axis.
    """
    if isinstance(coords, tuple) and len(coords) == 2:
        x = np.asarray(coords[0])  # (n_wt,)
        y = np.asarray(coords[1])  # (n_wt,)
    else:
        xy = np.asarray(coords)  # (n_wt, 2)
        x = xy[:, 0]
        y = xy[:, 1]

    # cache packed PyWake partials for reuse across calls
    dP_dz = make_dP_dz_inflowgrid(wf_model, time=time)

    def grad_fn(wd, ws, TI, *, return_gamma=False, time=time):
        """
        Compute full chain-rule dP/d(x,y) for given inflow.

        Parameters
        ----------
        wd, ws : float or array_like
            Wind direction(s) and speed(s).
        TI : float
            Turbulence intensity (scalar).
        return_gamma : bool
            If True, also return gamma.

        Returns
        -------
        dP_dxy : ndarray
            See factory docstring for shapes.
            Last axis is [dP/dx, dP/dy].
        gamma : ndarray, optional
            Same shape as dP_dxy without the final axis.
        """
        wds = np.asarray(wd)
        wss = np.asarray(ws)

        if wds.ndim == 0 and wss.ndim == 0:
            mode = "single"
        elif wds.ndim == 1 or wss.ndim == 1:
            mode = "lut"
        else:
            raise ValueError("unknown inflow shape")
        if time:  # overwrites 'lut'
            assert wds.ndim == 1 and wss.ndim == 1
            mode = "time"

        wds_1d = np.atleast_1d(wds)
        wss_1d = np.atleast_1d(wss)

        if mode == "lut":
            n_wd, n_ws = len(wds_1d), len(wss_1d)
            graphs = graph_maker(
                x,
                y,
                wd=wds_1d,
                ws=wss_1d,
                TI=TI,
                lut=True,
                connectivity=connectivity,
                rotor_diameter=rotor_diameter,
                k_wake=k_wake,
            )
            prepared = prepare_from_graphs(
                graphs,
                lut=True,
                n_wd=n_wd,
                edge_uv_cols=edge_uv_cols,
            )
        else:
            graphs = graph_maker(
                x,
                y,
                wd=wds_1d,
                ws=wss_1d,
                TI=TI,
                lut=False,
                connectivity=connectivity,
                rotor_diameter=rotor_diameter,
                k_wake=k_wake,
            )
            prepared = prepare_from_graphs(
                graphs,
                lut=False,
                edge_uv_cols=edge_uv_cols,
            )

        out = gradP_vjp_xy_inflowgrid_prepared(
            ts_path=ts_path,
            prepared=prepared,
            dP_dz=dP_dz,
            gamma_col=gamma_col,
            uv_scale=uv_scale,
            return_gamma=return_gamma,
        )

        if return_gamma:
            dP_dxy_list, gamma_list = out
        else:
            dP_dxy_list = out

        # stack list of (n_wt,2) -> (n_cases, n_wt, 2) -> (n_wt, n_cases, 2)
        # will break for arbitrary number of turbines
        dP = np.stack(dP_dxy_list, axis=0).transpose(1, 0, 2)

        if mode == "single":
            dP = dP[:, 0, :]  # (n_wt, 2)
        elif mode == "time":
            # (n_wt, T, 2)
            pass
        else:
            # (n_wt, n_wd, n_ws, 2)
            dP = dP.reshape(dP.shape[0], n_wd, n_ws, 2)

        if not return_gamma:
            return dP

        g = np.stack(gamma_list, axis=0).T  # (n_wt, n_cases)
        if mode == "single":
            g = g[:, 0]  # (n_wt,)
        elif mode == "time":
            # (n_wt, T)
            pass
        else:
            g = g.reshape(g.shape[0], n_wd, n_ws)

        return dP, g

    return grad_fn


if __name__ == "__main__":
    # case
    from design_friendly.utils.iea22s import IEA22s
    from design_friendly.utils.sites import Hornsrev1Site

    # this is a smoother version of PyWake IEA22 that works better with wake steering optimization
    wt = IEA22s()
    wds = np.arange(0, 360, 4)
    wss = np.arange(3, 25, 4)
    x, y, site = Hornsrev1Site(
        scale_D=wt.diameter()  # scale up the layout based on turbine diameter ratio
    )
    TI = 0.04  # site.local_wind().TI_ilk.ravel()s
    n_threads = 4

    # predict LUT
    # from design_friendly.utils.easy import easy_yaw_gnn

    # work with PyWake-style inputs
    yaws = easy(x, y, wd=wds, ws=wss, TI=TI, num_threads=n_threads, batch_size=512)
