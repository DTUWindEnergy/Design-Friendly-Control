import logging
import os
import warnings
from itertools import product

import torch
from design_friendly.utils.misc import log_execution_time
from concurrent.futures import ThreadPoolExecutor
from py_wake import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import (
    Cartesian,
    Delaunay,
    FaceToEdge,
    LocalCartesian,
    Polar,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # DEBUG

# based on https://github.com/gduthe/windfarm-gnn

_WEST_WD = 270.0  # canonical upstream wind direction after farm rotation

_EDGE_TRANSFORMS = {
    "polar": Polar(norm=False),
    "cartesian": Cartesian(norm=False),
    "local cartesian": LocalCartesian(norm=False),
}


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
        raise ValueError(f"points must have shape (n_wt, 2), got {pts.shape}")

    wd = float(wd_deg)
    center = geometric_median(pts[:, 0], pts[:, 1])
    q = pts - center  # zero-centered
    theta = np.deg2rad(_WEST_WD - (wd % 360.0))
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
    t = np.deg2rad(_WEST_WD - (float(wd_deg) % 360.0))
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]], float)
    return np.dot(rp, R.T) + np.asarray(center, float)


def gen_graph_edges(
    points: np.array,
    connectivity="wake_aware",
    add_edge="cartesian",
    wd=270,
    rotor_diameter=284,
    k_wake=0.4,  # covers 99th percentile of the TI distribution in the dataset 3std.dev. of the gaussian wake profile envelope
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
    conn = connectivity.strip().casefold()
    if conn not in ("delaunay", "fully_connected", "wake_aware"):
        raise ValueError(
            f"connectivity must be 'delaunay', 'fully_connected', or 'wake_aware', got {connectivity!r}"
        )
    points = rotate_to_west_centered(points, wd)
    t = torch.as_tensor(points, dtype=torch.float32)

    if conn == "delaunay":
        g = FaceToEdge()(Delaunay()(Data(pos=t)))

    elif conn == "fully_connected":
        n = t.shape[0]
        row = torch.arange(n, device=t.device).repeat_interleave(n)
        col = torch.arange(n, device=t.device).repeat(n)
        mask = row != col
        g = Data(pos=t, edge_index=torch.stack([row[mask], col[mask]]))

    elif conn == "wake_aware":
        # TODO: upstream only directionality for Power/WS_eff preds
        n = t.shape[0]
        x = t[:, 0]  # (n,)
        y = t[:, 1]  # (n,)
        # dx[i,j] = x_j - x_i: j is downstream of i when dx > 0
        dx = x[None, :] - x[:, None]  # (n, n)
        dy = y[None, :] - y[:, None]  # (n, n)
        R = 0.5 * rotor_diameter
        r_wake = R + float(k_wake) * dx
        mask = (dx > 0.0) & (dy.abs() <= r_wake)
        if conn_distxmaxD is not None:
            mask = mask & (dx <= float(conn_distxmaxD) * rotor_diameter)
        mask.fill_diagonal_(False)
        if conn_topk is None:
            src, dst = mask.nonzero(as_tuple=True)
        else:
            k_eff = min(int(conn_topk), n - 1)
            cost = (dx * dx + dy * dy).masked_fill(~mask, float("inf"))
            vals, src_idx = torch.topk(cost, k=k_eff, dim=0, largest=False)
            dst_idx = torch.arange(n, device=t.device).view(1, n).expand_as(src_idx)
            valid = torch.isfinite(vals)
            src = src_idx[valid]
            dst = dst_idx[valid]
        if src.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=t.device)
        else:
            edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        g = Data(pos=t, edge_index=edge_index)

    add_edge = add_edge.strip().casefold()
    if add_edge not in _EDGE_TRANSFORMS:
        raise ValueError("available coord.: 'polar', 'cartesian' or 'local cartesian'")
    g = _EDGE_TRANSFORMS[add_edge](g)
    return g


def process_one_layout(
    layout,
    inflow,
    per_turbine_dict=False,
    target_dict=None,
    layout_id="default",
    connectivity="wake_aware",
    source_datainfo="robust_slsqp",  # meta info
    _edge_cache=None,  # internal: pre-built cache from generate_graphs
    rotor_diameter=284.0,
    k_wake=0.40,
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
    Data
        Single graph for this layout/inflow pair.
    """
    assert connectivity in ["delaunay", "fully_connected", "wake_aware"]

    wt_coords = np.asarray(layout["coords"], float)
    info = (
        str(layout_id),
        str(len(wt_coords)),
        str(layout["form"]),
    )

    n_wt = wt_coords.shape[0]
    WS = float(inflow["WS"])
    TI = float(inflow["TI"])
    WD = float(inflow["WD"])
    farm_center = np.median(wt_coords, axis=0)

    if target_dict is None:
        target_dict = {"yaw": np.repeat(np.nan, n_wt)}
    elif isinstance(target_dict, dict):
        bad_keys = [k for k, v in target_dict.items() if np.isnan(v).any()]
        if bad_keys:
            raise ValueError(f"NaNs found in target_dict for keys: {bad_keys}")
    else:
        raise ValueError("target_dict not configured ")

    if _edge_cache is not None:
        _key = (wt_coords.tobytes(), round(WD, 4))
        _cached = _edge_cache.get(_key)
    else:
        _cached = None
    g = (
        _cached.clone()
        if _cached is not None
        else gen_graph_edges(
            wt_coords,
            connectivity=connectivity,
            add_edge="cartesian",
            wd=WD,
            rotor_diameter=rotor_diameter,
            k_wake=k_wake,
        )
    )

    if per_turbine_dict is True:
        warnings.warn("Not tested. slow but might help with load predictions.")
        from design_friendly.utils.get_flowmodel import get_flowmodel
        from design_friendly.utils.iea22 import IEA22

        wt = IEA22()
        wffm = get_flowmodel(wt=wt)
        x__rot = g.pos[:, 0]
        y__rot = g.pos[:, 1]
        yaw_baseline = np.zeros(len(x__rot))
        WS_eff, TI_eff, Power, CT, _, _ = wffm(
            x__rot,
            y__rot,
            yaw=yaw_baseline,
            wd=_WEST_WD,  # after rotation
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
        "ws": WS,
        "ti": TI,
        "farm_center_x": float(farm_center[0]),
        "farm_center_y": float(farm_center[1]),
    }
    g.name = "_".join(
        f"{float(v):.2f}" if hasattr(v, "__float__") else str(v)
        for v in g.meta.values()
    )
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
    rotor_diameter=284.0,
    k_wake=0.40,
):
    """
    Convert layout/inflow pairs into a Graphs.

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
        Optional path to save the Graphs.
    return_list_of_graphs : Bool
        Return list of Data instead of Graphs

    Returns
    -------
    Graphs
        Graphs containing graphs for every case.
    """
    assert len(layouts) == len(inflows), (
        f"layouts {len(layouts)} and inflows {len(inflows)}"
    )
    n_cases = len(layouts)
    if isinstance(per_turbines, (bool, np.bool_)):
        per_turbines = [bool(per_turbines)] * n_cases
    if target_dicts is None:
        target_dicts = [None] * n_cases

    # Single-threaded pre-pass: build edge-structure cache (write-once, then read-only in threads)
    _edge_cache: dict = {}
    for layout, inflow in zip(layouts, inflows):
        WD = round(float(inflow["WD"]), 4)
        coords_arr = np.asarray(layout["coords"], float)
        key = (coords_arr.tobytes(), WD)
        if key not in _edge_cache:
            _edge_cache[key] = gen_graph_edges(
                coords_arr,
                connectivity=connectivity,
                add_edge="cartesian",
                wd=WD,
                rotor_diameter=rotor_diameter,
                k_wake=k_wake,
            )

    iter_cases = list(zip(layouts, inflows, per_turbines, target_dicts))

    def _make_graph(args):
        i, layout, inflow, per_turbine, target_dict = args
        return process_one_layout(
            layout=layout,
            inflow=inflow,
            per_turbine_dict=per_turbine,
            target_dict=target_dict,
            layout_id=str(i).zfill(7),
            connectivity=connectivity,
            source_datainfo=source_datainfo,
            _edge_cache=_edge_cache,
            rotor_diameter=rotor_diameter,
            k_wake=k_wake,
        )

    indexed_cases = [(i, *case) for i, case in enumerate(iter_cases)]
    use_parallel = (num_threads is None) or (num_threads == -1) or (num_threads > 1)
    if use_parallel:
        max_workers = None if num_threads in (-1, None) else num_threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            graphs = list(executor.map(_make_graph, indexed_cases))
    else:
        graphs = [_make_graph(args) for args in indexed_cases]

    logger.info(f"generated {len(graphs)} graphs from {n_cases} cases")
    if save_pt_path:
        torch.save(graphs, save_pt_path)
        logger.info(f"Saved graphs to: {save_pt_path}")
    if return_list_of_graphs:
        return graphs
    return Graphs(graphs)


class Graphs(Dataset):
    """
    Dataset wrapper that normalizes GraphSet-style inputs.
    Accepts:
      - list of torch_geometric Data
      - str/.pt path with list of torch_geometric Data
    """

    def __init__(self, source):
        super().__init__()
        if isinstance(source, (str, os.PathLike)) and str(source).endswith(".pt"):
            loaded = torch.load(source, map_location="cpu")
            if not isinstance(loaded, list):
                raise TypeError("Unsupported .pt contents; expected list of Data")
            self._entries = loaded
        elif isinstance(source, list):
            if source and not isinstance(source[0], Data):
                raise TypeError("List source must be list of Data")
            self._entries = source
        else:
            raise TypeError("pass list of Data or .pt path")
        # init stats
        self.x_stats = self.y_stats = self.edge_stats = self.glob_stats = None

    def names(self):
        return [d.name for d in self._entries]

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
        return len(self._entries)

    def __getitem__(self, idx):
        data = self._entries[idx]
        data = data.clone()

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
        data.globals = data.globals.float().unsqueeze(0)  # (1, G)
        return data


def _pywake_baseline(x, y, wd, ws, TI, *, time, num_threads):
    """Run 0-yaw PyWake simulation; returns (WS_eff, TI_eff, Power, CT)."""
    from design_friendly.utils.get_flowmodel import get_flowmodel
    from design_friendly.utils.iea22 import IEA22

    wffm = get_flowmodel(wt=IEA22())
    kwargs = dict(
        yaw=np.zeros(len(x)),
        wd=wd,
        ws=ws,
        TI=TI,
        tilt=0,
        return_simulationResult=False,
        n_cpu=num_threads,
    )
    if time:
        kwargs["time"] = True
    WS_eff, TI_eff, Power, CT, _, _ = wffm(x, y, **kwargs)
    return WS_eff, TI_eff, Power, CT


@log_execution_time
def graph_maker(
    x,
    y,
    wd,
    ws,
    TI,
    lut=False,
    per_turbines=False,
    target_dicts=None,
    connectivity="wake_aware",
    num_threads=0,
    rotor_diameter=284.0,
    k_wake=0.40,
):
    """Unified graph maker.

    Parameters
    ----------
    x, y : array-like
        ''(n_wt,)'' fixed layout repeated for all cases.
        ''(n_cases, n_wt_max)'' NaN-padded per-case layouts (sequential mode).
    wd : array-like
        ''(n_cases,)'' wind directions paired 1:1 with cases when ''lut=False''.
        ''(n_wd,)'' wind-direction grid when ''lut=True''.
    ws : array-like
        ''(n_cases,)'' wind speeds paired 1:1 with cases when ''lut=False''.
        ''(n_ws,)'' wind-speed grid when ''lut=True''.
    TI : float or array-like
        Turbulence intensity.  Scalar broadcast to all cases.  Array only
        allowed when ''lut=False'' (one value per case).
    lut : bool, default False
        ''False'' 1:1 pairing of wd/ws (time-series or sequential mode).
        ''True'' cartesian product wd x ws (look-up-table mode).
    per_turbines : bool or list of dict
        ''False'' no per-turbine baselines.
        ''True'' compute 0-yaw PyWake baseline internally.
        list pre-computed dicts with keys WS_eff/TI_eff/Power/CT.
    target_dicts : list of dict or None
        Passed directly to ''generate_graphs''.
    connectivity : str
        Edge-structure method; forwarded to ''generate_graphs''.
    num_threads : int
        Thread count forwarded to ''generate_graphs'' and PyWake calls.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if x.dtype == object or y.dtype == object:
        raise ValueError(
            "Variable-length layouts must be NaN-padded to the same length "
            "and passed as a 2-D array."
        )

    wd = np.atleast_1d(np.asarray(wd, float))
    ws = np.atleast_1d(np.asarray(ws, float))

    if lut:  # LUT to generate wdxws
        if np.size(TI) != 1:
            raise ValueError("TI must be a scalar in lut mode.")
        TI_val = float(np.atleast_1d(TI).flat[0])
        n_wd, n_ws = len(wd), len(ws)
        n_cases = n_wd * n_ws

        _coords = np.column_stack((x, y))
        layouts = [{"coords": _coords, "form": "fixed"}] * n_cases
        inflows = [
            {"WS": float(ws_i), "TI": TI_val, "WD": float(wd_i)}
            for wd_i, ws_i in product(wd, ws)
        ]

        if per_turbines is True:
            WS_eff, TI_eff, Power, CT = _pywake_baseline(
                x, y, wd, ws, TI_val, time=False, num_threads=num_threads
            )
            # WS_eff shape: (n_wt, n_wd, n_ws)
            per_turbines = [
                {
                    "WS_eff": WS_eff[:, i_wd, j_ws],
                    "TI_eff": TI_eff[:, i_wd, j_ws],
                    "Power": Power[:, i_wd, j_ws],
                    "CT": CT[:, i_wd, j_ws],
                }
                for i_wd in range(n_wd)
                for j_ws in range(n_ws)
            ]

    else:  # ts (1D x) or sequential (2D x)
        TI = np.atleast_1d(np.asarray(TI, float))
        if TI.size == 1:
            TI = np.ones(len(wd), dtype=float) * float(TI)
        if not (len(wd) == len(ws) == len(TI)):
            raise ValueError(
                f"wd, ws, TI must have the same length; got "
                f"{len(wd)}, {len(ws)}, {len(TI)}."
            )
        n_cases = len(wd)

        if x.ndim == 2:
            # sequential: each row is one layout, NaN-padded
            if x.shape[0] != n_cases:
                raise ValueError(
                    f"x has {x.shape[0]} rows but wd has {n_cases} entries."
                )
            coords_list = [
                np.column_stack((xi[~np.isnan(xi)], yi[~np.isnan(xi)]))
                for xi, yi in zip(x, y)
            ]
            layouts = [{"coords": c, "form": "PLayGen"} for c in coords_list]
        else:
            # time = True: same layout for every case
            _coords = np.column_stack((x, y))
            layouts = [{"coords": _coords, "form": "fixed"}] * n_cases

        inflows = [
            {"WS": float(ws_i), "TI": float(ti_i), "WD": float(wd_i)}
            for wd_i, ws_i, ti_i in zip(wd, ws, TI)
        ]

        if per_turbines is True:
            if x.ndim == 1:
                WS_eff, TI_eff, Power, CT = _pywake_baseline(
                    x, y, wd, ws, TI, time=True, num_threads=num_threads
                )
                # WS_eff shape: (n_wt, n_cases)
                per_turbines = [
                    {
                        "WS_eff": WS_eff[:, i],
                        "TI_eff": TI_eff[:, i],
                        "Power": Power[:, i],
                        "CT": CT[:, i],
                    }
                    for i in range(n_cases)
                ]
            else:
                warnings.warn("Pre-compute baselines per_turbines and pass.")

    return generate_graphs(
        layouts=layouts,
        inflows=inflows,
        num_threads=num_threads,
        target_dicts=target_dicts,
        per_turbines=per_turbines,
        connectivity=connectivity,
        source_datainfo="robust_slsqp",
        save_pt_path=None,
        rotor_diameter=rotor_diameter,
        k_wake=k_wake,
    )
