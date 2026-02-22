import logging
import os
import warnings
from itertools import product

import torch
from design_friendly.utils.misc import log_execution_time
from joblib import Parallel, delayed
from py_wake import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import (
    Cartesian,
    Delaunay,
    FaceToEdge,
    LocalCartesian,
    Polar,
)
from torch_geometric.utils import dense_to_sparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # DEBUG

# based on https://github.com/gduthe/windfarm-gnn


def geometric_median(x, y):
    """
    (Memory intensive way) to calculate geometric median (central turbine) based
    inter-distances between all the turbines in the wind farm

    Parameters
    ----------
    x, y : array_like, shape (n,)
        Coordinates of points.

    Returns
    -------
    center : ndarray, shape (2,)
        Coordinates of the central point (argmin distance to all points).
    """
    P = np.column_stack([x, y])  # (n,2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)  # (n,n)
    idx = np.argmin(D.sum(axis=1))  # index of central turbine (medoid)
    center = P[idx]
    return center


def rotate_to_west_centered(points, wd_deg):
    """
    Rotate a farm layout so the incoming wind aligns with 270.

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
        raise ValueError("points must have shape (n_wt, 2), got", pts.shape)

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
    connectivity="wake_aware",
    add_edge="cartesian",
    wd=270,
    rotor_diameter=284,
    k_wake=0.33,  # cover 99th percentile of the TI distribution in the dataset (0.11) with 3 standard deviations (99.7 coverage) of wake expansion of the gaussian shape
    conn_distxmaxD=None,
    conn_topk=None,
):
    """
    Build a PyG graph edges from turbine coordinates.
    # based on https://github.com/gduthe/windfarm-gnn

    Parameters
    ----------
    points : ndarray, shape (n_wt, 2)
        Turbine layout coordinates.
    connectivity : str
        Connectivity type ('delaunay', 'wake_aware', or 'fully_connected').
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
        "fully_connected",
        "wake_aware",
    ]
    points = rotate_to_west_centered(points, wd)
    assert points.shape[1] == 2

    t = torch.Tensor(points)
    x = Data(pos=t)
    if connectivity.casefold() == "delaunay":
        d = Delaunay()
        e = FaceToEdge()
        g = e(d(x))
    elif connectivity.casefold() == "fully_connected":
        adj = torch.ones(t.shape[0], t.shape[0])
        g = Data(pos=t, edge_index=dense_to_sparse(adj.fill_diagonal_(0))[0])
    elif connectivity.casefold() == "wake_aware":
        # connect iff downstream rotor center lies inside an expanding wake cone based.
        n = t.shape[0]
        x = t[:, 0]  # (n,)
        y = t[:, 1]  # (n,)
        # Pairwise differences: dx[i,j] = x_j - x_i, dy[i,j] = y_j - y_i
        # Should be able to use Cartesian(norm=False) but not sure how it implements
        dx = x[None, :] - x[:, None]  # (n, n)
        dy = y[None, :] - y[:, None]  # (n, n)
        R = 0.5 * rotor_diameter
        r_wake = R + float(k_wake) * dx  # (n, n)
        # Interaction mask: j is downstream of i and within wake radius
        mask = (dx > 0.0) & (dy.abs() <= r_wake)
        if conn_distxmaxD is not None:
            dx_max = float(conn_distxmaxD) * rotor_diameter
            mask = mask & (dx <= dx_max)
        mask.fill_diagonal_(False)
        # directionless, connect i-j if i wakes j OR j wakes i
        if conn_topk is None:  # induces discontinuity
            src, dst = mask.nonzero(as_tuple=True)  # directed i->j
        else:
            # Keep only top-K upstream "closest" per downstream turbine
            K = int(conn_topk)
            k_eff = min(K, n - 1)
            dist = dx * dx + dy * dy  # dy.abs()
            cost = dist.masked_fill(~mask, float("inf"))  # (n, n)
            # For each downstream column j, pick k_eff upstream i with smallest cost[i,j]
            # (k_eff, n)
            vals, src_idx = torch.topk(cost, k=k_eff, dim=0, largest=False)
            dst_idx = torch.arange(n, device=t.device).view(1, n).expand(k_eff, n)
            valid = torch.isfinite(vals)  # (k_eff, n)
            src = src_idx[valid]
            dst = dst_idx[valid]
        if src.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=t.device)
        else:
            edge_index = torch.stack(
                [torch.cat([src, dst]), torch.cat([dst, src])],
                dim=0,
            )
        g = Data(pos=t, edge_index=edge_index)
    else:
        raise ValueError("'delaunay', 'knn', 'fully_connected', 'wake_aware'")

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
    per_turbine_dict=False,
    target_dict=None,
    layout_id="default",
    connectivity="wake_aware",
    source_datainfo="robust_slsqp",  # meta info
):
    """
    Convert a single layout and inflow pair graphs.

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
        False: repeat ambient features in each node
        True: calculate baseline node features w PyWake
        dict of np.array: use as node features
    connectivity : str, default 'delaunay'
        Graph connectivity scheme to use.
    source_datainfo : str
        Model name recorded in graph metadata.

    Returns
    -------
    list of graphs
        in-memory graphs.
    """
    assert connectivity in ["delaunay", "knn", "fully_connected", "all", "wake_aware"]

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
        if any(np.isnan(v).any() for v in target_dict.values()):
            bad_keys = [k for k, v in target_dict.items() if np.isnan(v).any()]
            if bad_keys:
                raise ValueError(f"NaNs found in target_dict for keys: {bad_keys}")
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

    if per_turbine_dict is True:
        warnings.warn("Not tested. slow but might help with load predictions.")
        from design_friendly.utils.get_flowmodel import get_flowmodel
        from design_friendly.utils.iea22s import IEA22s

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
    if isinstance(per_turbine_dict, dict):
        per_turbine_dict = {k: np.ravel(v) for k, v in per_turbine_dict.items()}
        assert (
            per_turbine_dict["Power"].shape
            == per_turbine_dict["CT"].shape
            == per_turbine_dict["WS_eff"].shape
            == per_turbine_dict["TI_eff"].shape
            == (wt_coords.shape[0],)
        )
        node_features = np.column_stack([v for v in per_turbine_dict.values()])
        # node features  # (n_wt, n_node_features)
        g.x = torch.tensor(node_features, dtype=torch.float32)
    else:
        g.x = None

    target_features = np.column_stack([v for v in target_dict.values()])

    # target (node) features  # (n_wt, n_target_features)
    g.y = torch.tensor(target_features, dtype=torch.float32)
    # global features  # (2, ) for now
    ambient = np.array([WS, TI], float)
    g.globals = torch.tensor(ambient, dtype=torch.float32)
    # meta data
    if per_turbine_dict is False:
        g.node_feature_keys = []
    else:
        g.node_feature_keys = list(per_turbine_dict.keys())
    g.target_feature_keys = list(target_dict.keys())
    g.meta = {
        "layout_id": layout_id,
        "connectivity": connectivity,
        "yaws_method": source_datainfo,
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
    # remove this since we will get nan's predicting g.y without inputs
    if torch.isnan(g.globals).any():  # torch.isnan(g.y).any() or
        raise ValueError("NaNs in graph?")
    return g


@log_execution_time
def generate_graphs(
    layouts,  # list of layout dicts
    inflows,  # list of inflow dicts
    num_threads=0,
    target_dicts=None,
    per_turbines=False,  # list of per_turbine dicts
    connectivity="wake_aware",
    source_datainfo="robust_slsqp",  # meta info
    save_pt_path=None,
    return_list_of_graphs=False,
):
    """
    Convert layout/inflow pairs into a GraphFarmsDataset.
    # based on https://github.com/gduthe/windfarm-gnn

    Parameters
    ----------
    layouts : list of dict
        Layout dictionaries containing 'coords' entries.
    inflows : list of dict
        Inflow dictionaries with 'WS', 'TI', and 'WD'.
    num_threads : int, default 0
        Number of threads for parallel conversion.
    per_turbines : list of dict or bool
        Optional per-turbine baselines matching layouts.
        False: repeat ambient features in each node
        True: calculate baseline node features w PyWake
        dict of np.array: use as node features
    connectivity : str, default 'delaunay'
        Graph connectivity scheme for all layouts.
    source_datainfo : str
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
    assert len(layouts) == len(inflows), (
        f"layouts {len(layouts)} and inflows {len(inflows)}"
    )
    n_cases = len(layouts)
    if isinstance(per_turbines, (bool, np.bool_)):
        per_turbines = [bool(per_turbines)] * n_cases
    if target_dicts is None:
        target_dicts = [None] * n_cases

    iter_cases = list(zip(layouts, inflows, per_turbines, target_dicts))
    if (num_threads == -1) or (num_threads > 1) or (num_threads is None):
        graphs = Parallel(n_jobs=num_threads)(
            delayed(process_one_layout)(
                layout=layout,
                inflow=inflow,
                per_turbine_dict=per_turbine,
                target_dict=target_dict,  # None defaults to yaw
                layout_id=str(i).zfill(7),
                connectivity=connectivity,
                source_datainfo=source_datainfo,
            )
            for i, (layout, inflow, per_turbine, target_dict) in enumerate(iter_cases)
        )
    else:
        graphs = [
            process_one_layout(
                layout=layout,
                inflow=inflow,
                per_turbine_dict=per_turbine,
                target_dict=target_dict,
                layout_id=str(i).zfill(7),
                connectivity=connectivity,
                source_datainfo=source_datainfo,
            )
            for i, (layout, inflow, per_turbine, target_dict) in enumerate(iter_cases)
        ]

    logger.info(f"generated {len(graphs)} graphs from {n_cases} cases")
    if save_pt_path:
        torch.save(graphs, save_pt_path)
        logger.info("Saved graphs to:", save_pt_path)
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
        if isinstance(source, (str, os.PathLike)) and str(source).endswith(".pt"):
            # Path to .pt? not tested
            loaded = torch.load(source, map_location="cpu")
            if isinstance(loaded, list):
                entries = loaded
            else:
                raise TypeError("Unsupported .pt contents; expected list of Data")
        # list of Data
        elif isinstance(source, list):
            if len(source) == 0:
                entries = []
            elif isinstance(source[0], Data):
                entries = source
            else:
                raise TypeError("List source must be list of Data")
        else:
            raise TypeError("pass list of Data or .pt path")

        self._entries = entries
        self.__num_graphs = len(self._entries)
        # init stats
        self.x_stats = self.y_stats = self.edge_stats = self.glob_stats = None

    def names(self):
        return [n for n, _ in self._entries]

    def items(self):
        # yields Data with clones to mutate safely outside
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
        data.globals = data.globals.float().unsqueeze(0)  # (1, G)
        return data


@log_execution_time
def graph_maker_sequential(
    xs,
    ys,
    wds,
    wss,
    TIs,
    per_turbines=False,  # precalculated node features
    target_dicts=None,
    connectivity="wake_aware",
    num_threads=0,
):
    """
    Generate graphs for sequential layout/inflow tuples. format inputs and call generate_graphs

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
        False: repeat ambient features in each node
        True: calculate baseline node features w PyWake
        dict of np.array: use as node features
    connectivity : str
        connectivity type for all cases.

    Returns
    -------
    GraphSet
        Graphs for each inflow case.
    """
    assert len(wds) == len(wss)
    # remove padded NaNs from all inputs (for varying layout sizes)
    coords = [  # drop nan's (padded for dataset)
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
        num_threads=num_threads,
        target_dicts=target_dicts,
        per_turbines=per_turbines,
        connectivity=connectivity,
        source_datainfo="robust_slsqp",
        save_pt_path=None,
    )
    return graphs


@log_execution_time
def graph_maker_time(
    x,
    y,
    wd_t,
    ws_t,
    TI_t,
    target_dicts=None,
    per_turbines=False,
    connectivity="wake_aware",
    num_threads=0,
):
    """
    Build a lookup table of graphs over wind direction/speed combinations for specific WF
    coordinates. format inputs and call generate_graphs. graph_maker should use the similar
    inputs as PyWake whenever possible

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
    per_turbines : list or bool
        Per-turbine baselines.
        False: repeat ambient features in each node
        True: calculate baseline node features w PyWake
        dict of np.array: use as node features
    connectivity : str
        PyG connectivity type used to build each graph.

    Returns
    -------
    GraphSet
        Graphs for the provided WD/WS combinations.
    """
    if np.size(TI_t) == 1:
        TI_t = np.ones_like(wd_t) * TI_t
    assert len(wd_t) == len(ws_t) == len(TI_t)
    n_ts = len(wd_t)
    coords = [np.column_stack((x, y))] * n_ts  # repeat coordinates for gnn input
    layouts = [{"coords": c, "form": "test"} for c in coords]
    inflows = [
        {"WS": float(ws), "TI": float(TI), "WD": float(wd)}
        for wd, ws, TI in zip(wd_t, ws_t, TI_t)
    ]
    # generate PyWake-vectorized baseline
    if per_turbines is True:
        logging.info("Generating baseline with 0-yaw PyWake")
        from design_friendly.utils.get_flowmodel import get_flowmodel
        from design_friendly.utils.iea22s import IEA22s

        wt = IEA22s()
        wffm = get_flowmodel(wt=wt)
        yaw_baseline = np.zeros(len(x))
        WS_eff, TI_eff, Power, CT, _, _ = wffm(
            x,
            y,
            yaw=yaw_baseline,
            wd=wd_t,  # after rotation
            ws=ws_t,
            TI=TI_t,
            tilt=0,
            time=True,
            return_simulationResult=False,
            n_cpu=num_threads,
        )  # WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk
        n_wt = len(x)
        assert WS_eff.squeeze().shape == (n_wt, n_ts)
        per_turbines = [
            {
                "WS_eff": WS_eff[:, i_t],
                "TI_eff": TI_eff[:, i_t],
                "Power": Power[:, i_t],
                "CT": CT[:, i_t],
            }
            for i_t in range(n_ts)
        ]
        assert len(per_turbines) == len(inflows) == n_ts

    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=num_threads,
        target_dicts=target_dicts,
        per_turbines=per_turbines,  # None for pywake baseline# y shape (5, n_wt)
        connectivity=connectivity,
        source_datainfo="robust_slsqp",
        save_pt_path=None,
    )
    return graphs


@log_execution_time
def graph_maker_lut(
    x,
    y,
    wds,
    wss,
    TI,  # single TI for now
    target_dicts=None,
    per_turbines=False,
    connectivity="wake_aware",
    num_threads=0,
):
    """
    Build a lookup table of graphs over wind direction/speed combinations for specific WF
    coordinates. format inputs and call generate_graphs. graph_maker should use the similar
    inputs as PyWake whenever possible

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
    per_turbines : list or bool
        Per-turbine baselines.
        False: repeat ambient features in each node
        True: calculate baseline node features w PyWake
        dict of np.array: use as node features
    connectivity : str
        PyG connectivity type used to build each graph.

    Returns
    -------
    GraphSet
        Graphs across the provided WD/WS combinations.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    wds = np.atleast_1d(wds)
    wss = np.atleast_1d(wss)
    TI = np.atleast_1d(TI)
    n_cases = np.size(wds) * np.size(wss)
    coords = [np.column_stack((x, y))] * n_cases  # repeat coordinates for gnn input
    layouts = [{"coords": c, "form": "test"} for c in coords]
    # inflows should cover all combinations of wds and wss

    inflows = [
        {"WS": float(ws), "TI": float(TI), "WD": float(wd)}
        for wd, ws in product(wds, wss)
    ]

    # generate PyWake-vectorized baseline
    if per_turbines:
        logging.info("Generating baseline with 0-yaw PyWake")
        from design_friendly.utils.get_flowmodel import get_flowmodel
        from design_friendly.utils.iea22s import IEA22s

        wt = IEA22s()
        wffm = get_flowmodel(wt=wt)
        yaw_baseline = np.zeros(len(x))
        WS_eff, TI_eff, Power, CT, _, _ = wffm(
            x,
            y,
            yaw=yaw_baseline,
            wd=wds,  # after rotation
            ws=wss,
            TI=TI,
            tilt=0,
            return_simulationResult=False,
            n_cpu=num_threads,
        )  # WS_eff_ilk, TI_eff_ilk, power_ilk, ct_ilk, localWind, kwargs_ilk
        n_wt = len(x)
        n_wd = len(wds)
        n_ws = len(wss)
        assert WS_eff.shape == (n_wt, n_wd, n_ws)
        per_turbines = [
            {
                "WS_eff": WS_eff[:, i_wd, j_ws],
                "TI_eff": TI_eff[:, i_wd, j_ws],
                "Power": Power[:, i_wd, j_ws],
                "CT": CT[:, i_wd, j_ws],
            }
            for i_wd in range(n_wd)  # wd-major
            for j_ws in range(n_ws)
        ]
        assert len(per_turbines) == len(inflows) == n_wd * n_ws

    graphs = generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=num_threads,
        target_dicts=target_dicts,
        per_turbines=per_turbines,  # None for pywake baseline# y shape (5, n_wt)
        connectivity=connectivity,
        source_datainfo="robust_slsqp",
        save_pt_path=None,
    )
    return graphs


@log_execution_time
def seq_graph_inputs(ds, _slice):
    """
    convert ~PyWake xarray ds to graph structure
    """
    ds_sel = ds.sel(case=_slice)
    # edge inputs (n_case, n_wt_max)
    xs = ds_sel.rob_x.values
    ys = ds_sel.rob_y.values
    # misc
    nwts = ds_sel.n_wt.values.astype(int)  # (n_case,)
    # globals (n_case,)
    wds = ds_sel.rob_WD.values
    wss = ds_sel.rob_WS.values
    TIs = ds_sel.rob_TI.values
    # targets (per turbine; (n_case, n_wt_max))
    WS_effs = ds_sel.rob_WS_eff.values
    TI_effs = ds_sel.rob_TI_eff.values
    yaws = np.round(ds_sel.rob_yaw.values, 4)
    # broadcast globals to per-turbine (n_case, 1)
    wss_b = wss[:, None]
    TIs_b = TIs[:, None]
    # deficit targets (n_case, n_wt_max)
    WS_def = wss_b - WS_effs
    TI_def = TIs_b - TI_effs
    # framework format (variable-length per case)
    train_target_dicts = [
        {
            "WS_eff_def": WS_row_def[:n_wt],
            "TI_eff_def": TI_row_def[:n_wt],
            "yaw": yaw_row[:n_wt],  # yaw stays absolute
        }
        for WS_row_def, TI_row_def, yaw_row, n_wt in zip(WS_def, TI_def, yaws, nwts)
    ]
    # assert removing only zeros (padding)
    assert all(np.isnan(row[n_wt:]).all() for row, n_wt in zip(yaws, nwts)), (
        "Found non-zero padding in yaw"
    )

    sequential_inputs = {
        "xs": xs,
        "ys": ys,
        "wds": wds,
        "wss": wss,
        "TIs": TIs,
        "target_dicts": train_target_dicts,
    }
    return sequential_inputs
