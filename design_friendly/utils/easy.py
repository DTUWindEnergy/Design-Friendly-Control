from itertools import product

import numpy as np
from design_friendly.models import models_filepath
from design_friendly.utils.misc import log_execution_time
from design_friendly.utils.pred import predict
from design_friendly.utils.to_graph import graph_maker_lut, graph_maker_time


@log_execution_time
def easy_yaw_gnn(
    x,
    y,
    wd,
    ws,
    TI,
    model_path=models_filepath + "best.pt",
    num_threads=0,
    batch_size=256,
    time=False,
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
        )
        results = predict(
            model_path=model_path,
            graphfarms=graphs,
            batch_size=batch_size,  # int(len(wd) * len(ws)),
            reshape=(n_wt, len(wd), len(ws)),
        )
    elif time:
        n_t = len(wd)
        assert n_t == len(ws), "provide time series"
        graphs = graph_maker_time(
            x=x,
            y=y,
            wd_t=wd,
            ws_t=ws,
            TI_t=TI,
            num_threads=num_threads,
        )
        results = predict(
            model_path=model_path,
            graphfarms=graphs,
            batch_size=batch_size,
            reshape=(n_wt, n_t),
        )
    return results


def main():
    from design_friendly.utils.sites import Hornsrev1Site

    wds = np.arange(0, 360, 2)
    wss = np.arange(3, 25, 1)
    TI = 0.06
    (x, y), _, _ = Hornsrev1Site()
    yaws = easy_yaw_gnn(x, y, wd=wds, ws=wss, TI=TI)