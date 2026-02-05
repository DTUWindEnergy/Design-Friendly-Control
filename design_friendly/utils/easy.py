from design_friendly.models import models_filepath

from .misc import log_execution_time
from .pred import predict_torchscript, torchscript_to_lut
from .to_graph import graph_maker_lut, graph_maker_sequential, graph_maker_time

from .vjp import (
    gradP_vjp_xy_inflowgrid_prepared,
    make_dP_dz_inflowgrid,
    prepare_inflowgrid_from_layout,  # mega-graph
    prepare_inflowseq_from_layout,  # mega-graph sequence
)


# TODO: generalize vjp's graph structure to to_graph
@log_execution_time
def easy_yaw_gnn(
    x,
    y,
    wd,
    ws,
    TI,
    model_path=models_filepath + "best.ptnox.torchscript.pt",
    num_threads=0,
    batch_size=512,
    time=False,
    output_yaw_idx=-1,  # multivariate preds
    sequential=False,
):
    if sequential:
        assert time is False, "sequential only for steady state"
        assert len(wd) == len(ws) == len(TI)
        graphs = graph_maker_sequential(
            xs=x,
            ys=y,
            wds=wd,
            wss=ws,
            TIs=TI,
            connectivity="wake_aware",
        )
        results = predict_torchscript(
            model_path,
            graphs,
            batch_size,
            "array",
        )  # wt, ts, out
        results = results[:, :, output_yaw_idx]
    elif not time:
        graphs = graph_maker_lut(
            x=x,
            y=y,
            wds=wd,
            wss=ws,
            TI=TI,
            num_threads=num_threads,
            connectivity="wake_aware",
        )
        results = predict_torchscript(
            model_path,
            graphs,
            batch_size,
            "array",
        )  # wt, wd, ws, out
        results = torchscript_to_lut(results, wd, ws)
        results = results[:, :, :, output_yaw_idx]
    elif time:
        assert len(wd) == len(ws), "provide time series"
        graphs = graph_maker_time(
            x=x,
            y=y,
            wd_t=wd,
            ws_t=ws,
            TI_t=TI,
            num_threads=num_threads,
            connectivity="wake_aware",
        )
        results = predict_torchscript(
            model_path,
            graphs,
            batch_size,
            "array",
        )  # ts, wt, out
        results = results[:, :, output_yaw_idx]
        results = results.T  # wt, ts
    else:
        raise ValueError("invalid combination of time, sequential or lut")
    return results


@log_execution_time
def easy_grad(
    *,
    wf_model,
    coords,
    ts_path=models_filepath + "best.ptnox.torchscript.pt",
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
    easy : callable
        easy(wd, ws, TI, return_gamma=False)

        Modes inferred from wd/ws:
          - scalar/scalar -> single case
          - 1D/1D with same length -> time series (paired)
          - 1D/1D with different lengths -> LUT grid (cross-product)

        Outputs:
          - single: dP_dxy (n_wt, 2)
          - time:   dP_dxy (n_wt, T, 2)
          - LUT:    dP_dxy (n_wt, n_wd, n_ws, 2)
        If return_gamma=True, returns matching gamma arrays without the last dim.
    """
    from py_wake import numpy as np

    if isinstance(coords, tuple) and len(coords) == 2:
        x = np.asarray(coords[0])  # (n_wt,)
        y = np.asarray(coords[1])  # (n_wt,)
    else:
        xy = np.asarray(coords)  # (n_wt, 2)
        x = xy[:, 0]
        y = xy[:, 1]

    # cache packed PyWake partials for reuse across calls
    dP_dz = make_dP_dz_inflowgrid(wf_model, time=time)

    def easy(wd, ws, TI, *, return_gamma=False, time=time):
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

        if mode == "lut":
            # cross-product grid: (n_wd, n_ws)
            wd_grid, ws_grid = np.meshgrid(wds, wss, indexing="ij")
            wds_flat = wd_grid.ravel()
            wss_flat = ws_grid.ravel()
            n_wd, n_ws = wd_grid.shape
        else:
            wds_flat = np.atleast_1d(wds)
            wss_flat = np.atleast_1d(wss)

        if time:
            prepared = prepare_inflowseq_from_layout(
                x=x,
                y=y,
                wd_t=wds_flat,
                ws_t=wss_flat,
                TI=TI,
                connectivity=connectivity,
                edge_uv_cols=edge_uv_cols,
                rotor_diameter=rotor_diameter,
                k_wake=k_wake,
            )
        else:
            prepared = prepare_inflowgrid_from_layout(
                x=x,
                y=y,
                wds=wds_flat,
                wss=wss_flat,
                TI=TI,
                connectivity=connectivity,
                edge_uv_cols=edge_uv_cols,
                rotor_diameter=rotor_diameter,
                k_wake=k_wake,
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

    return easy
