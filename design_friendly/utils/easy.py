from itertools import product

import numpy as np
from design_friendly.utils.pred import predict
from design_friendly.utils.to_graph import graph_maker_lut
from design_friendly.models import models_filepath

def easy_yaw_gnn(
    x,
    y,
    wd,
    ws,
    TI,
    model_path=models_filepath + "best.pt",
    num_threads=0,
):
    n_wt = len(x)
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
        test_graphs=graphs,
        batch_size=int(len(wd) * len(ws)),
        reshape=(n_wt, len(wd), len(ws)),
    )
    return results


def main():
    from design_friendly.utils.sites import Hornsrev1Site

    wds = np.arange(0, 360, 2)
    wss = np.arange(3, 25, 1)
    TI = 0.06
    (x, y), _, _ = Hornsrev1Site()
    yaws = easy_yaw_gnn(x, y, wd=wds, ws=wss, TI=TI)
