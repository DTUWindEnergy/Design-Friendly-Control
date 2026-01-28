from design_friendly.models import models_filepath
from py_wake import numpy as np

from .misc import log_execution_time
from .pred import predict_torchscript, torchscript_to_lut
from .to_graph import graph_maker_lut, graph_maker_time


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
):
    n_wt = len(x)
    if not time:
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
        results = results[:, :, -1]
        results = results.T  # wt, ts
    return results


def main():
    from design_friendly.utils.sites import Hornsrev1Site

    wds = np.arange(0, 360, 2)
    wss = np.arange(3, 25, 1)
    TI = 0.06
    (x, y), _, _ = Hornsrev1Site()
    yaws = easy_yaw_gnn(x, y, wd=wds, ws=wss, TI=TI)
