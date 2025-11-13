import os
import math
import warnings
from itertools import product
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import (
    Cartesian,
    Delaunay,
    FaceToEdge,
    KNNGraph,
    LocalCartesian,
    Polar,
)
from torch_geometric.utils import dense_to_sparse
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


def rotate_to_west_centered(points, wd_deg):
    """
    Rotate a farm layout so the incoming wind aligns with 270°.

    Parameters
    ----------
    points : array_like, shape (n_wt, 2)
        Coordinates of the wind turbines.
    wd_deg : float
        Wind direction in degrees.

    Returns
    -------
    ndarray, shape (n_wt, 2)
        Median-centered, rotated coordinates.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n_wt, 2)")

    wd = float(wd_deg)
    center = np.median(pts, axis=0)  # (2,)
    q = pts - center  # zero-centered
    theta = np.deg2rad(270.0 - (wd % 360.0))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return np.dot(q, R)  # .T


def unrotate_from_west_centered(rot_pts, wd_deg, center):
    """
    Restore coordinates before the west-centering rotation.

    Parameters
    ----------
    rot_pts : array_like, shape (n_wt, 2)
        Rotated positions.
    wd_deg : float
        Wind direction used for rotation.
    center : array_like, shape (2,)
        Median center recorded before rotation.

    Returns
    -------
    ndarray, shape (n_wt, 2)
        Un-rotated coordinates.
    """
    rp = np.asarray(rot_pts, float)
    t = np.deg2rad(270.0 - (float(wd_deg) % 360.0))
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]], float)
    return np.dot(rp, R.T) + np.asarray(center, float)


def gen_graph_edges(
    points: np.array,
    connectivity: str,
    add_edge="cartesian",
    wd=270,
):
    """
    Build a PyG graph from turbine coordinates.

    Parameters
    ----------
    points : ndarray, shape (n_wt, 2)
        Turbine layout coordinates.
    connectivity : str
        Connectivity type ('delaunay', 'knn', or 'fully_connected').
    add_edge : str, default 'cartesian'
        Edge feature type to attach ('polar', 'cartesian', or 'local cartesian').
    wd : float, default 270
        Wind direction used for layout rotation. (not a model input)

    Returns
    -------
    Data
        Initial graph data from wind farm layout.
    """
    assert connectivity in [
        "delaunay",
        "knn",
        "fully_connected",
    ]
    points = rotate_to_west_centered(points, wd)
    assert points.shape[1] == 2

    t = torch.Tensor(points)
    x = Data(pos=t)
    if connectivity.casefold() == "delaunay":
        d = Delaunay()
        e = FaceToEdge()
        g = e(d(x))
    elif connectivity.casefold() == "knn":
        kv = math.ceil(np.sqrt(len(points)))
        knn = KNNGraph(k=kv)
        g = knn(x)
    elif connectivity.casefold() == "fully_connected":
        adj = torch.ones(t.shape[0], t.shape[0])
        g = Data(pos=t, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])
    # elif connectivity.casefold() == "wake_aware":  # TODO: implement wake-aware connectivity
    # TODO: wake aware connectivity: after the rotation, wake expansion follows the wffm-expansion to figure out which turbines are in the wake and generate edge_index accordingly
    else:
        raise ValueError("available connectivity: 'delaunay', 'knn', 'fully_connected'")

    add_edge = add_edge.strip().casefold()
    if add_edge == "polar".casefold():
        p = Polar(norm=False)
        g = p(g)
    elif add_edge == "cartesian".casefold():
        c = Cartesian(norm=False)
        g = c(g)
    elif add_edge == "local cartesian".casefold():
        lc = LocalCartesian(norm=False)
        g = lc(g)
    else:
        raise ValueError("available coord.: 'polar', 'cartesian' or 'local cartesian'")

    return g


def process_one_layout(
    layout,
    inflow,
    layout_id,
    per_turbine_features=True,
    per_turbine=None,
    connectivity="delaunay",
    yaws_model="COBYQA-QMCB",  # meta info
):
    """
    Convert a single layout and inflow pair into named graphs.

    Parameters
    ----------
    layout : dict
        Layout payload containing 'coords' and metadata.
    inflow : dict
        Dictionary with 'WS', 'TI', and 'WD' values.
    layout_id : str
        Identifier used in graph metadata.
    per_turbine_features : bool, default True
        Whether to include per-turbine features on the graph nodes.
    per_turbine : dict or None
        Optional baseline per-turbine metrics.
    connectivity : str, default 'delaunay'
        Graph connectivity scheme to use.
    yaws_model : str
        Model name recorded in graph metadata.

    Returns
    -------
    list of tuple
        (name, Data) pairs generated from this layout.
    """
    assert connectivity in ["delaunay", "knn", "fully_connected", "all"]

    wt_coords = np.asarray(layout["coords"], float)
    info = (
        str(layout_id),
        str(len(wt_coords)),
        str(layout["form"]),
    )

    WS = float(np.asarray(inflow["WS"]).item())
    TI = float(np.asarray(inflow["TI"]).item())
    WD = float(np.asarray(inflow["WD"]).item())
    WEST_WD = 270.0
    farm_center = np.median(wt_coords, axis=0)

    graphs = []

    g = gen_graph_edges(
        wt_coords,
        connectivity=connectivity,
        add_edge="cartesian",
        wd=WD,  # for rotation
    )

    ambient = np.array([WS, TI], float)

    if per_turbine_features:
        if per_turbine is None:
            from utils.get_flowmodel import get_flowmodel
            from utils.iea22s import IEA22s

            wt = IEA22s()
            wffm = get_flowmodel(wt=wt)
            x__rot = g.pos[:, 0]
            y__rot = g.pos[:, 1]
            yaw_ = np.zeros(len(x__rot))
            ws_eff, ti_eff, power, ct_eff, _, _ = wffm(
                x__rot,
                y__rot,
                yaw=yaw_,
                wd=WEST_WD,  # after graph_edges rotation
                ws=WS,
                TI=TI,
                tilt=0,
                return_simulationResult=False,
            )  # WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk
        else:
            power = np.asarray(per_turbine["power"])
            ct_eff = np.asarray(per_turbine["ct"])
            ws_eff = np.asarray(per_turbine["ws_eff"])
            ti_eff = np.asarray(per_turbine["ti_eff"])
            yaw_ = np.asarray(per_turbine["yaw"])
        ws_eff, ti_eff, power, ct_eff, yaw_ = map(
            np.ravel, (ws_eff, ti_eff, power, ct_eff, yaw_)
        )
        assert (
            power.shape
            == ct_eff.shape
            == ws_eff.shape
            == ti_eff.shape
            == yaw_.shape
            == (wt_coords.shape[0],)
        )
        # predict yaw_ only given py_wake baseline per_turbine inputs
        node_features = np.array([power, ws_eff, ct_eff, ti_eff])
        g.x = torch.tensor(node_features, dtype=torch.float32).T  # (n_wt, 4)
        t = torch.tensor(yaw_, dtype=torch.float32)  # (n_wt,)
        g.y = t.T  # (n_wt,)
    else:
        # predict all per_turbine values given ambient only
        warnings.warn(
            "currently there is no ambient-only input model but repeating freestream power, ct, ws, ti seems to work as input to predict setpoints. (still no prediction of these values themselves)"
        )
        out = np.column_stack((power, ct_eff, ws_eff, ti_eff, yaw_))  # (n_wt, 5)
        g.y = torch.tensor(out, dtype=torch.float32).T  # (5, n_wt)
    g.globals = torch.tensor(ambient, dtype=torch.float32)
    g.meta = {
        "layout_id": layout_id,
        "connectivity": connectivity,
        "per_turbine_features": bool(per_turbine_features),
        "yaws_method": yaws_model,
        "info": info,
        "wd_deg": WD,
        "ws": g.globals[0].item(),
        "ti": g.globals[1].item(),
        "farm_center_x": float(farm_center[0]),
        "farm_center_y": float(farm_center[1]),
    }
    cstr = lambda v: f"{float(v):.2f}" if hasattr(v, "__float__") else str(v)
    meta_name = lambda d: "_".join(map(cstr, d.values()))
    g.name = meta_name(g.meta)
    if torch.isnan(g.y).any() or torch.isnan(g.globals).any():
        raise ValueError("NaNs in graph?")
    graphs.append((g.name, g))

    return graphs


def generate_graphs(
    layouts,  # list of layout dicts
    inflows,  # list of inflow dicts
    num_threads=-1,
    per_turbines=None,  # list of per_turbine dicts
    per_turbine_features=True,
    connectivity="delaunay",
    yaws_model="COBYQA-QMCB",  # meta info
    save_pt_path=None,
    return_entries=False,
):
    """
    Convert layout/inflow pairs into a GraphFarmsDataset.

    Parameters
    ----------
    layouts : list of dict
        Layout dictionaries containing 'coords' entries.
    inflows : list of dict
        Inflow dictionaries with 'WS', 'TI', and 'WD'.
    num_threads : int, default -1
        Number of threads for parallel conversion.
    per_turbines : list or None
        Optional per-turbine baselines matching layouts.
    per_turbine_features : bool, default True
        Whether to attach per-turbine node features.
    connectivity : str, default 'delaunay'
        Graph connectivity scheme for all layouts.
    yaws_model : str
        Name recorded in graph metadata.
    save_pt_path : str or None
        Optional path to save the GraphFarmsDataset.

    Returns
    -------
    GraphFarmsDataset
        GraphFarmsDataset containing graphs for every case.
    """
    assert len(layouts) == len(inflows)  # == len(per_turbines)
    n_cases = len(layouts)

    if per_turbines is None:
        per_turbines = [None] * len(layouts)

    with tqdm_joblib(tqdm(desc="Converting to graphs", total=n_cases)):
        results = Parallel(n_jobs=num_threads)(
            delayed(process_one_layout)(
                layout=layouts[i],
                inflow=inflows[i],
                per_turbine=per_turbines[i],
                layout_id=str(i).zfill(7),
                per_turbine_features=per_turbine_features,
                connectivity=connectivity,
                yaws_model=yaws_model,
            )
            for i in range(n_cases)
        )

    # Flatten: list[list[(name, Data)]] -> list[(name, Data)]
    entries = [pair for case_graphs in results for pair in case_graphs]
    print("generated", len(entries), "graphs from", n_cases, "cases")

    if save_pt_path:
        torch.save(entries, save_pt_path)
        print("Saved graphs to:", save_pt_path)

    if return_entries:
        return entries

    return GraphFarmsDataset(entries)


class GraphFarmsDataset(Dataset):
    """
    Dataset wrapper that normalizes GraphSet-style inputs.
    Accepts:
      - list of (name, Data)
      - list of Data (we will synthesize names)
      - str/.pt path holding a list of (name, Data)
    """

    def __init__(self, source):
        super().__init__()
        entries = None

        # Path to .pt?
        if isinstance(source, (str, os.PathLike)) and str(source).endswith(".pt"):
            loaded = torch.load(source, map_location="cpu")
            if isinstance(loaded, list):
                entries = loaded
            else:
                raise TypeError(
                    "Unsupported .pt contents; expected list of (name, Data)"
                )

        # List / tuple?
        elif isinstance(source, (list, tuple)):
            if len(source) == 0:
                entries = []
            elif isinstance(source[0], tuple) and len(source[0]) == 2:
                # list of (name, Data)
                entries = list(source)
            elif isinstance(source[0], Data):
                # list of Data -> synthesize names
                entries = [(f"g_{i:05d}", g) for i, g in enumerate(source)]
            else:
                raise TypeError(
                    "List source must be list of Data or list of (name, Data)"
                )

        else:
            raise TypeError("pass list or .pt path")

        self._entries = entries
        self.__num_graphs = len(self._entries)

        self.x_stats = self.y_stats = self.edge_stats = self.glob_stats = None

    def names(self):
        return [n for n, _ in self._entries]

    def items(self):
        # yields (name, Data) with clones so you can mutate safely outside
        for n, d in self._entries:
            yield n, d

    def save_pt(self, path: str):
        torch.save(self._entries, path)

    @classmethod
    def load_pt(cls, path: str):
        entries = torch.load(path, map_location="cpu")
        if not isinstance(entries, list):
            raise TypeError("Unsupported .pt contents; expected list of (name, Data)")
        return cls(entries)

    @property
    def num_glob_features(self):
        data = self[0]
        return data.globals.shape[1]

    @property
    def num_glob_output_features(self):
        data = self[0]
        return getattr(data, "globals_y", torch.empty(1, 0)).shape[1]

    @property
    def num_node_output_features(self):
        data = self[0]
        return 1 if data.y.dim() == 1 else data.y.size(1)

    def __len__(self):
        return self.__num_graphs

    def __getitem__(self, idx):
        name, data = self._entries[idx]
        data = data.clone()

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
        # if data.globals.float().dim == 1:
        data.globals = data.globals.float().unsqueeze(0)  # (1, G)
        try:
            layout_idx = int(str(name).split("_", 1)[0])
        except Exception:
            layout_idx = -1

        data.provenance = {"name": name, "layout_idx": layout_idx}
        return data


def graph_maker_sequential(
    xs,
    ys,
    wds,
    wss,
    TIs,
    per_turbines=None,
    connectivity="delaunay",
):
    """
    Generate graphs for sequential layout/inflow tuples.

    Parameters
    ----------
    xs : list of array_like
        X coordinates for each layout.
    ys : list of array_like
        Y coordinates for each layout.
    wds : list of float
        Wind directions per layout.
    wss : list of float
        Wind speeds per layout.
    TIs : list of float
        Turbulence intensities per layout.
    per_turbines : list or None
        Optional per-turbine baselines for each layout.
    connectivity : str
        connectivity type for all cases.

    Returns
    -------
    GraphSet
        Graphs for each inflow case.
    """
    assert len(wds) == len(wss)

    coords = [np.column_stack((x, y)) for x, y in zip(xs, ys)]
    layouts = [{"coords": c, "form": "test"} for c in coords]
    inflows = [
        {"WS": float(ws), "TI": float(ti), "WD": float(wd)}
        for wd, ws, ti in zip(wds, wss, TIs)
    ]
    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=-1,
        per_turbines=None,
        per_turbine_features=True,  # y shape (5, n_wt)
        connectivity="delaunay",
        yaws_model="COBYQA-QMCB",
        save_pt_path=None,
    )
    return graphs


def graph_maker_lut(
    x,
    y,
    wds,
    wss,
    TI,  # single TI for now
    per_turbines=None,
    connectivity="delaunay",
):
    """
    Build a lookup table of graphs over wind direction/speed combinations for specific WF coordinates.

    Parameters
    ----------
    x : array_like
        WF layout X coordinates.
    y : array_like
        WF layout Y coordinates.
    wds : Sequence[float]
        Wind directions to iterate.
    wss : Sequence[float]
        Wind speeds to iterate.
    TI : float
        Turbulence intensity applied to all cases.
    per_turbines : list or None
        Optional per-turbine baselines.
    connectivity : str
        PyG connectivity type used to build each graph.

    Returns
    -------
    GraphSet
        Graphs across the provided WD/WS combinations.
    """
    n_cases = len(wds) * len(wss)
    coords = [np.column_stack((x, y))] * n_cases  # repeat coordinates for gnn input
    layouts = [{"coords": c, "form": "test"} for c in coords]
    # inflows should cover all combinations of wds and wss

    inflows = [
        {"WS": float(ws), "TI": float(TI), "WD": float(wd)}
        for wd, ws in product(wds, wss)
    ]

    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=-1,
        per_turbines=None,  # None for pywake baseline
        per_turbine_features=True,  # y shape (5, n_wt)
        connectivity="delaunay",
        yaws_model="COBYQA-QMCB",
        save_pt_path=None,
    )
    return graphs


def test_cases_graph(wd_=270):
    """
    Create a small set of random layouts for smoke testing.

    Parameters
    ----------
    wd_ : float, default 270
        Wind direction applied to every generated case.

    Returns
    -------
    tuple
        (GraphSet, layouts, inflows) for the test cases.
    """

    def make_case(n_wt, ws, ti, wd, rd=248.0, spacing_D=4.0):
        spacing = rd * spacing_D
        n1 = (n_wt + 1) // 2  # row 1 count
        n2 = n_wt - n1  # row 2 count

        noise_1 = np.random.uniform(-rd / 2, rd / 2, size=n1)
        noise_2 = np.random.uniform(-rd / 2, rd / 2, size=n2)
        X1 = noise_1 + np.arange(n1, dtype=float) * spacing
        X2 = noise_2 + np.arange(n2, dtype=float) * spacing

        Y1 = noise_1
        Y2 = noise_2 + 3 * rd

        coords = np.column_stack([np.concatenate([X1, X2]), np.concatenate([Y1, Y2])])

        layout = {"coords": coords, "form": "test"}
        inflow = {"WS": float(ws), "TI": float(ti), "WD": float(wd)}
        return layout, inflow

    ws_, ti_ = np.array(6.0), np.array(0.05)

    # build some cases
    n_wts = [5, 10, 19, 13]
    cases = [make_case(n, ws_, ti_, wd_) for n in n_wts]
    layouts, inflows = map(list, zip(*cases))

    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=-1,
        per_turbines=None,
        per_turbine_features=True,  # y shape (5, n_wt)
        connectivity="delaunay",
        yaws_model="COBYQA-QMCB",
        save_pt_path=None,
    )

    # assertions
    assert len(graphs) == 4
    sizes = [5, 10]
    for (name, g), n_wt in zip(graphs.items(), sizes):
        assert isinstance(name, str)
        assert isinstance(g, Data)
        assert g.pos.shape[0] == n_wt
        assert g.globals.shape == (2,)
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.shape[1] > 0
        assert not torch.isnan(g.y).any()
        assert not torch.isnan(g.globals).any()

    return graphs, layouts, inflows


def main():
    """
    Smoke-test the graph generation utilities and rotations.
    """
    # check test_cases_graph.generate_graphs
    graphs, layouts, inflows = test_cases_graph()
    for name, g in graphs.items():
        print(f"Graph {name}:")
        print(g)
        print("meta:", g.meta)
    # check rotated graphs
    wd_ = 188.0
    graphs, layouts, inflows = test_cases_graph(wd_=wd_)
    for (name, g), layout, inflow in zip(graphs.items(), layouts, inflows):
        assert np.allclose(
            unrotate_from_west_centered(
                g.pos,
                wd_deg=wd_,
                center=(g.meta["farm_center_x"], g.meta["farm_center_y"]),
            ),
            layout["coords"],
            atol=1e-3,
        ), "g.pos dont match graph to layout"
        assert np.allclose(g.globals[0], inflow["WS"]), "g.globals no match to inflow"

    # check graph_maker_sequential
    # match test_cases_graph first and compare
    graphs, layouts, inflows = test_cases_graph(wd_=wd_)
    wd_ = inflows[0]["WD"]
    xs = [l["coords"][:, 0] for l in layouts]
    ys = [l["coords"][:, 1] for l in layouts]
    n_layouts = len(xs)
    graphs = graph_maker_sequential(
        xs=xs,
        ys=ys,
        wds=[wd_] * n_layouts,
        wss=[inflows[0]["WS"]] * n_layouts,
        TIs=[inflows[0]["TI"]] * n_layouts,
    )
    for (name, g), layout, inflow in zip(graphs.items(), layouts, inflows):
        assert np.allclose(
            unrotate_from_west_centered(
                g.pos,
                wd_deg=wd_,
                center=(g.meta["farm_center_x"], g.meta["farm_center_y"]),
            ),
            layout["coords"],
            atol=1e-3,
        ), "g.pos dont match graph to layout"
        assert np.allclose(g.globals[0], inflow["WS"]), "g.globals no match to inflow"
        print(name)

    # check graph_maker_lut
    wds = np.arange(0, 360, 3)
    wss = np.arange(3, 25, 1)
    ti = inflows[0]["TI"]
    graphs = graph_maker_lut(
        x=xs[0],
        y=ys[0],
        wds=wds,
        wss=wss,
        TI=ti,
    )
    inflows = [
        {"WS": float(ws), "TI": float(ti), "WD": float(wd)}
        for wd, ws in product(wds, wss)
    ]
    layout_ = np.column_stack((xs[0], ys[0]))

    for (name, g), inflow in zip(graphs.items(), inflows):
        assert np.allclose(
            unrotate_from_west_centered(
                g.pos,
                wd_deg=inflow["WD"],
                center=(g.meta["farm_center_x"], g.meta["farm_center_y"]),
            ),
            layout_,
            atol=1e-3,
        ), "g.pos dont match graph to layout"
        assert np.allclose(g.globals[0], inflow["WS"]), "g.globals no match to inflow"


if __name__ == "__main__":
    main()
