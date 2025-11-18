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


def geometric_median(x, y):
    """
    Memory intensive way to calculate geometric median (central turbine) based
    inter-distances between all the turbines in the wind farm

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Coordinates of points.

    Returns
    -------
    center : ndarray, shape (2,)
        Coordinates of the central point (existing point minimizing
        the sum of Euclidean distances to all points).
    """

    P = np.column_stack([x, y])  # (n,2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)  # (n,n)
    idx = np.argmin(D.sum(axis=1))  # index of central turbine (medoid)
    center = P[idx]
    return center


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
    # center = np.median(pts, axis=0)  # (2,)
    center = geometric_median(pts[:, 0], pts[:, 1])
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
    per_turbine_dict=True,
    target_dict=None,
    layout_id="default",
    connectivity="delaunay",
    yaws_model="COBYQA-QMCB",  # meta info
):
    """
    Convert a single layout and inflow pair into named graphs.

    Parameters
    ----------
    layout : dict
        Turbine 'coords' and metadata. To be encoded in edge
    inflow : dict
        Dictionary with 'WS', 'TI', and 'WD' values. To be encoded as globals
    layout_id : str
        Identifier used in graph metadata.
    target_dict : None or dict of np.array
        None: predict yaw
        dict: target variables to predict
    per_turbine_dict : bool or dict of np.array
        False: repeat ambient features in each node (TODO: allow empty g.x TODO @EPD)
        True: calculate baseline node features w PyWake
        dict of np.array: use as node features
    connectivity : str, default 'delaunay'
        Graph connectivity scheme to use.
    yaws_model : str
        Model name recorded in graph metadata.

    Returns
    -------
    list of graphs
        in-memory graphs.
    """
    assert connectivity in ["delaunay", "knn", "fully_connected", "all"]

    wt_coords = np.asarray(layout["coords"], float)
    info = (
        str(layout_id),
        str(len(wt_coords)),
        str(layout["form"]),
    )

    n_wt = wt_coords.shape[0]
    WS = float(np.asarray(inflow["WS"]).item())
    TI = float(np.asarray(inflow["TI"]).item())
    WD = float(np.asarray(inflow["WD"]).item())
    WEST_WD = 270.0
    farm_center = np.median(wt_coords, axis=0)

    if target_dict is None:
        target_dict = {"yaw": np.repeat(np.nan, n_wt)}
    elif isinstance(target_dict, dict):
        if np.any([np.isnan(v) for k, v in target_dict.items()]):
            warnings.warn("nan in target_dict. Filling all targets nan for prediction")
            target_dict = {k: np.repeat(np.nan, n_wt) for k, v in target_dict.items()}
    else:
        raise ValueError("target_dict not configured ")

    g = gen_graph_edges(
        wt_coords,
        connectivity=connectivity,
        add_edge="cartesian",
        wd=WD,  # for rotation
    )

    if per_turbine_dict is False:
        from utils.iea22s import IEA22s

        warnings.warn("ambient-only wip. Repeating ambient for node features")
        wt = IEA22s()
        per_turbine_dict = {}
        per_turbine_dict["WS"] = np.repeat(WS, n_wt)
        per_turbine_dict["TI"] = np.repeat(TI, n_wt)
        per_turbine_dict["Power"] = np.repeat(wt.Power(ws=WS))
        per_turbine_dict["CT"] = np.repeat(wt.ct(ws=WS))
    elif per_turbine_dict is True:
        from utils.get_flowmodel import get_flowmodel
        from utils.iea22s import IEA22s

        wt = IEA22s()
        wffm = get_flowmodel(wt=wt)
        x__rot = g.pos[:, 0]
        y__rot = g.pos[:, 1]
        yaw_baseline = np.zeros(len(x__rot))
        WS_eff, TI_eff, Power, CT, _, _ = wffm(
            x__rot,
            y__rot,
            yaw=yaw_baseline,
            wd=WEST_WD,  # after rotation
            ws=WS,
            TI=TI,
            tilt=0,
            return_simulationResult=False,
        )  # WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk
        per_turbine_dict = {}
        per_turbine_dict["WS_eff"] = WS_eff
        per_turbine_dict["TI_eff"] = TI_eff
        per_turbine_dict["Power"] = Power
        per_turbine_dict["CT"] = CT
    elif isinstance(per_turbine_dict, dict):
        required_keys = {"WS_eff", "TI_eff", "Power", "CT"}
        missing = required_keys - per_turbine_dict.keys()
        if missing:
            raise ValueError(f"per_turbine_dict is missing required keys: {missing}")
    else:
        raise ValueError("node features not configured")
    per_turbine_dict = {k: np.ravel(v) for k, v in per_turbine_dict.items()}
    assert (
        per_turbine_dict["Power"].shape
        == per_turbine_dict["CT"].shape
        == per_turbine_dict["WS_eff"].shape
        == per_turbine_dict["TI_eff"].shape
        == (wt_coords.shape[0],)
    )

    node_features = np.column_stack([v for v in per_turbine_dict.values()])
    target_features = np.column_stack([v for v in target_dict.values()])
    # node features  # (n_wt, n_node_features)
    # g.x = torch.tensor(node_features, dtype=torch.float32).T
    g.x = torch.tensor(node_features, dtype=torch.float32)
    # target (node) features  # (n_wt, n_target_features)
    g.y = torch.tensor(target_features, dtype=torch.float32)
    # global features  # (2, ) for now
    ambient = np.array([WS, TI], float)
    g.globals = torch.tensor(ambient, dtype=torch.float32)
    # meta data
    g.node_feature_keys = list(per_turbine_dict.keys())
    g.target_feature_keys = list(target_dict.keys())
    g.meta = {
        "layout_id": layout_id,
        "connectivity": connectivity,
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
    # remove this since we will get nan's predicting g.y without an actual
    if torch.isnan(g.globals).any():  # torch.isnan(g.y).any() or
        raise ValueError("NaNs in graph?")
    return g


def generate_graphs(
    layouts,  # list of layout dicts
    inflows,  # list of inflow dicts
    num_threads=-1,
    target_dicts=None,
    per_turbines=True,  # list of per_turbine dicts
    connectivity="delaunay",
    yaws_model="COBYQA-QMCB",  # meta info
    save_pt_path=None,
    return_list_of_graphs=False,
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
    per_turbines : list of dict or None
        Optional per-turbine baselines matching layouts.
    connectivity : str, default 'delaunay'
        Graph connectivity scheme for all layouts.
    yaws_model : str
        Name recorded in graph metadata.
    save_pt_path : str or None
        Optional path to save the GraphFarmsDataset.
    return_list_of_graphs : Bool
        Return list of Data instead of GraphFarmsDataset

    Returns
    -------
    GraphFarmsDataset
        GraphFarmsDataset containing graphs for every case.
    """
    assert len(layouts) == len(inflows)
    n_cases = len(layouts)
    if isinstance(per_turbines, (bool, np.bool_)):
        per_turbines = [bool(per_turbines)] * n_cases
    if target_dicts is None:
        target_dicts = [None] * n_cases

    if num_threads is -1 or num_threads > 1:
        with tqdm_joblib(tqdm(desc="Converting to graphs", total=n_cases)):
            graphs = Parallel(n_jobs=num_threads)(
                delayed(process_one_layout)(
                    layout=layouts[i],
                    inflow=inflows[i],
                    per_turbine_dict=per_turbines[i],
                    target_dict=target_dicts[i],  # None defaults to yaw
                    layout_id=str(i).zfill(7),
                    connectivity=connectivity,
                    yaws_model=yaws_model,
                )
                for i in range(n_cases)
            )
    else:
        graphs = []
        for layout, inflow, per_turbine, target_dict in zip(
            layouts, inflows, per_turbines, target_dicts
        ):
            g = process_one_layout(
                layout=layout,
                inflow=inflow,
                per_turbine_dict=per_turbine,
                target_dict=target_dict,
                layout_id=0,
                connectivity=connectivity,
                yaws_model=yaws_model,
            )
            graphs.append(g)
    print("generated", len(graphs), "graphs from", n_cases, "cases")
    if save_pt_path:
        torch.save(graphs, save_pt_path)
        print("Saved graphs to:", save_pt_path)
    if return_list_of_graphs:
        return graphs
    return GraphFarmsDataset(graphs)


class GraphFarmsDataset(Dataset):
    """
    Dataset wrapper that normalizes GraphSet-style inputs.
    Accepts:
      - list of torch_geometric Data
      - str/.pt path with list of torch_geometric Data
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
                raise TypeError("Unsupported .pt contents; expected list of Data")
        # List of Data
        elif isinstance(source, list):
            if len(source) == 0:
                entries = []
            elif isinstance(source[0], Data):
                entries = source
            else:
                raise TypeError("List source must be list of Data")
        else:
            raise TypeError("pass list or .pt path")

        self._entries = entries
        self.__num_graphs = len(self._entries)
        # init stats
        self.x_stats = self.y_stats = self.edge_stats = self.glob_stats = None

    def names(self):
        return [n for n, _ in self._entries]

    def items(self):
        # yields Data with clones so you can mutate safely outside
        for d in self._entries:
            yield d

    def save_pt(self, path: str):
        torch.save(self._entries, path)

    @classmethod
    def load_pt(cls, path: str):
        entries = torch.load(path, map_location="cpu")
        if not isinstance(entries, list):
            raise TypeError("Unsupported .pt contents; expected list of Data")
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
        data = self._entries[idx]
        data = data.clone()

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
        # if data.globals.float().dim == 1:
        data.globals = data.globals.float().unsqueeze(0)  # (1, G)
        try:
            layout_idx = data.layout_id
        except Exception:
            layout_idx = -1

        data.provenance = {"name": data.name, "layout_idx": layout_idx}
        return data


def graph_maker_sequential(
    xs,
    ys,
    wds,
    wss,
    TIs,
    per_turbines=True,  # precalculated node features
    target_dicts=None,
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
    per_turbines : list of dict of np.array or None
        Optional pre-calculated per-turbine baselines (0yaw) for each layout.
    connectivity : str
        connectivity type for all cases.

    Returns
    -------
    GraphSet
        Graphs for each inflow case.
    """
    assert len(wds) == len(wss)
    # TODO: remove padded NaNs for all inputs!
    # coords = [np.column_stack((x, y)) for x, y in zip(xs, ys)]
    coords = [  # drop nan's (padded for xarray dataset)
        np.column_stack((x[~np.isnan(x)], y[~np.isnan(x)])) for x, y in zip(xs, ys)
    ]
    layouts = [{"coords": c, "form": "PLayGen"} for c in coords]
    inflows = [
        {"WS": float(ws), "TI": float(ti), "WD": float(wd)}
        for wd, ws, ti in zip(wds, wss, TIs)
    ]
    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=-1,
        target_dicts=target_dicts,
        per_turbines=per_turbines,
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
    target_dicts=None,
    per_turbines=True,
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
        target_dicts=target_dicts,
        per_turbines=per_turbines,  # None for pywake baseline# y shape (5, n_wt)
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
        target_dict = {"yaw": np.repeat(np.nan, n_wt)}
        return layout, inflow, target_dict

    ws_, ti_ = np.array(6.0), np.array(0.05)

    # build some cases
    n_wts = [5, 10, 19, 13]
    n_cases = len(n_wts)
    cases = [make_case(n, ws_, ti_, wd_) for n in n_wts]
    layouts, inflows, target_dicts = map(list, zip(*cases))

    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=-1,
        target_dicts=target_dicts,
        per_turbines=True,  # y shape (5, n_wt)
        connectivity="delaunay",
        yaws_model="COBYQA-QMCB",
        save_pt_path=None,
    )

    # assertions
    assert len(graphs) == n_cases
    for g, n_wt in zip(graphs, n_wts):
        assert isinstance(g.name, str)
        assert isinstance(g, Data)
        assert g.pos.shape[0] == n_wt
        assert g.globals.shape == (1, 2)
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.shape[1] > 0
        # assert not torch.isnan(g.y).any()  # assigned nan internally
        assert not torch.isnan(g.globals).any()

    return graphs, layouts, inflows


def main():
    """
    Smoke-test the graph generation utilities and rotations.
    """
    # check test_cases_graph.generate_graphs
    graphs, layouts, inflows = test_cases_graph()
    for g in graphs:
        print(f"Graph {g.name}:")
        print(g)
        print("meta:", g.meta)
    # check rotated graphs
    wd_ = 188.0
    graphs, layouts, inflows = test_cases_graph(wd_=wd_)
    for g, layout, inflow in zip(graphs, layouts, inflows):
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
    for g, layout, inflow in zip(graphs, layouts, inflows):
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

    for g, inflow in zip(graphs, inflows):
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
