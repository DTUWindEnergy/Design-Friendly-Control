from design_friendly.models import models_filepath

from .misc import log_execution_time
from .pred import predict_torchscript, torchscript_to_lut
from .to_graph import graph_maker_lut, graph_maker_sequential, graph_maker_time


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
