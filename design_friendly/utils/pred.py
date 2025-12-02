import os

import numpy as np
import torch
import yaml
from design_friendly.gnn import WindFarmGNN  # GNN model class
from design_friendly.utils.misc import log_execution_time
from torch_geometric.loader import DataLoader


@log_execution_time
def predict(
    model_path,
    graphfarms,
    batch_size=1,
    reshape=None,
    append_latent=True,
):
    """
    Predict with a trained WindFarmGNN on an in-memory test set.
    """
    model_dir = os.path.dirname(model_path)
    search_roots = [model_dir, os.path.dirname(model_dir)]
    cfg_fns = ["config.yml"]
    cfg_path = None
    for root_dir in search_roots:
        if not root_dir:
            continue
        for root, _, files in os.walk(root_dir):
            for name in cfg_fns:
                if name in files:
                    cfg_path = os.path.join(root, name)
                    break
            if cfg_path is not None:
                break
        if cfg_path is not None:
            break
    if cfg_path is None:
        raise FileNotFoundError(f"trained_config.yml not found undesr {model_dir!r}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(
        graphfarms,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    model = WindFarmGNN(**config["hyperparameters"], **config["model_settings"])
    checkpoint = torch.load(
        os.path.join(model_path),
        map_location=device,
        weights_only=False,
    )
    model.trainset_stats = checkpoint["trainset_stats"]
    model.load_state_dict(checkpoint["model_state_dict"])

    # num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of trainable parameters:", num_t_params)
    # model = torch.compile(model)
    model.to(device)
    model.eval()

    y = []
    y_pred = []

    with torch.inference_mode():
        print(f"Evaluating model {model_path}")
        for i_batch, data in enumerate(test_loader):
            data = data.to(device)
            if append_latent:
                data, latent_x = model(data, denorm_output=True, return_latent=True)
                data.latent_x = latent_x
            else:
                data = model(data, denorm_output=True, return_latent=False)
            y += [data.y.squeeze().cpu().numpy()]
            y_pred += [data.x.squeeze().cpu().numpy()]
    if reshape is not None:
        if len(reshape) == 3:  # reshape ilk: (len(wt), len(wd), len(ws))
            flat = np.concatenate(y_pred)
            aILK = flat.reshape(
                reshape[1],
                reshape[2],
                reshape[0],
            ).transpose(2, 0, 1)  # TODO: hardcoded for windrose LUT
            return aILK
        elif len(reshape) == 2:  # reshape ilk: (len(wt), len(t_s))
            flat = np.concatenate(y_pred)
            aILK = flat.reshape(
                reshape[1],
                reshape[0],
            ).transpose(1, 0)  # TODO: hardcoded for timeseries
            return aILK
        else:
            raise ValueError("not implemented")

    return y_pred


def main():
    from utils.to_graph import test_cases_graph

    graphs = test_cases_graph()
    model_path = "/home/dgodi/ActiveProjects/eth/windfarm-gnn/gnn_framework/runs/GEN_4_layers_0.0_dropout_0.0001_lr_100_epochs_256_latent_dim_07_30_21_11/"
    results = predict(
        model_path=model_path,
        test_graphs=graphs,  # in-memory graphs
    )
    # visualize
    from py_wake import HorizontalGrid
    from utils.get_flowmodel import get_flowmodel
    from utils.iea22s import IEA22s

    wt = IEA22s()
    wf_model = get_flowmodel(wt=wt)

    from utils.plot_utils import pretty_flowmap

    for result, grph in zip(results, graphs):
        n_row1 = (len(result) + 1) // 2
        n_row2 = len(result) - n_row1
        print(result[:n_row1].astype("int"))
        print(result[n_row1:].astype("int"))
        ws_ = grph.globals[0]
        ti_ = grph.globals[1]
        wd_ = 270
        x_ = grph.pos[:, 0]
        y_ = grph.pos[:, 1]

        sim_base = wf_model(x=x_, y=y_, wd=wd_, ws=ws_, TI=ti_, yaw=0, tilt=0)
        fmap_base = sim_base.flow_map(grid=HorizontalGrid(resolution=150, extend=0.2))

        sim_gnn = wf_model(x=x_, y=y_, wd=wd_, ws=ws_, TI=ti_, yaw=result, tilt=0)
        fmap_gnn = sim_gnn.flow_map(grid=HorizontalGrid(resolution=150, extend=0.2))

        ws_base = fmap_base.WS_eff.squeeze()
        ws_opt = fmap_gnn.WS_eff.squeeze()
        ws_diff = ws_opt - ws_base
        pretty_flowmap(ws_diff, x_, y_, result, yaw_deg=result, wd_deg=wd_, D=284)
        base_power = (sim_base.Power.sum() / 1e6).item()
        gnn_power = (sim_gnn.Power.sum() / 1e6).item()
        print(f"{base_power:.2f} \n {gnn_power:.2f}")


if __name__ == "__main__":
    main()
