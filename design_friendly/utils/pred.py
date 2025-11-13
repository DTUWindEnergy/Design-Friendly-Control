import os
import numpy as np
import torch
import yaml
from gnn import WindFarmGNN  # GNN model class
from torch_geometric.loader import DataLoader


def predict(
    model_path,
    test_graphs,
    batch_size=1,
    reshape=None,
    temp_ret_all=None,
):
    """
    Predict with a trained WindFarmGNN on an in-memory test set.
    """
    model_dir = os.path.dirname(model_path)
    cfg_path = os.path.join(model_dir, "config.yml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    test_dataset = test_graphs
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # to recover order
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    model = WindFarmGNN(**config["hyperparameters"], **config["model_settings"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        os.path.join(model_path),
        map_location=device,
    )
    model.trainset_stats = checkpoint["trainset_stats"]
    model.load_state_dict(checkpoint["model_state_dict"])

    num_t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_t_params)

    model.to(device)
    model.eval()

    y = []
    y_pred = []

    with torch.no_grad():
        print(f"Evaluating model {model_path}")
        for i_batch, data in enumerate(test_loader):
            data = data.to(device)
            data, latent_x = model(data, denorm_output=True, return_latent=True)
            y += [data.y.squeeze().cpu().numpy()]
            y_pred += [data.x.squeeze().cpu().numpy()]
    if reshape is not None:
        flat = np.concatenate(y_pred)
        # reshape ilk: (len(wt), len(wd), len(ws))
        aILK = flat.reshape(
            reshape[1],
            reshape[2],
            reshape[0],
        ).transpose(2, 0, 1)
        if temp_ret_all:
            return (
                y,
                y_pred,
                aILK,
                data,
            )
        return aILK
    return y_pred


def main():
    from utils.to_graph import test_cases_graph

    graphs = test_cases_graph()
    model_path = "/home/dgodi/ActiveProjects/eth/windfarm-gnn/gnn_framework/runs/GEN_4_layers_0.0_dropout_0.0001_lr_100_epochs_256_latent_dim_07_30_21_11/"
    results = predict(
        model_path=model_path,
        test_graphs=graphs,  # GraphSet
        model_version="best",
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
