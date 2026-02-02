import torch
from design_friendly.utils.misc import log_execution_time
from py_wake import numpy as np
from py_wake.utils.gradients import autograd
from design_friendly.models import models_filepath

default_ts_path = models_filepath + "best.ptnox.torchscript.pt"
WEST_WD = 270.0


@log_execution_time
def make_dP_dz_inflowgrid(wf_model):  # PyWake partials
    """
    Return a packed gradient function dP/dz for total power over an (I,L,K) inflow grid
    (memory intensive).

    z packs three (I,L,K) fields: yaw, X, Y into one vector:
      n = I*L*K
      yaw = z[0:n].reshape(I,L,K)
      X   = z[n:2*n].reshape(I,L,K)
      Y   = z[2*n:3*n].reshape(I,L,K)

    Returns
    -------
    dP_dz : callable
        dP_dz(z, wds, wss, TI_lk, I, L, K) -> grad_z, shape (3*I*L*K,)
    """

    def power_inflowgrid(z, wds, wss, TI_lk, I, L, K):  # noqa: E741
        """
        Scalar objective used by dP_dz: sum of wf_model power over all (i,l,k).

        Parameters
        ----------
        z : (3*I*L*K,)  !wrt argument.
            Packed [yaw, X, Y] as described in make_dP_dz_inflowgrid.
        wds : (L,)
        wss : (K,)
        TI_lk : (L,K)
        I, L, K : int

        Returns
        -------
        P : float
            power_ilk.sum()
        """
        assert len(wds) == L
        assert len(wss) == K
        assert np.shape(TI_lk) == (L, K)
        n = I * L * K
        assert z.shape[0] == 3 * n
        yaw = z[0:n].reshape(I, L, K)  # (I,L,K)
        X = z[n : 2 * n].reshape(I, L, K)
        Y = z[2 * n : 3 * n].reshape(I, L, K)
        _, _, power_ilk, _, _, _ = wf_model(
            x=X,
            y=Y,
            # wd=wds,
            # west-aligned coords for rotated layout
            wd=np.full((L,), WEST_WD, dtype=np.float32),
            ws=wss,
            TI=TI_lk,
            yaw=yaw,
            tilt=0,
            return_simulationResult=False,
            n_cpu=1,
        )
        return power_ilk.sum()

    return autograd(
        power_inflowgrid, argnum=0, vector_interdependence=True
    )  # argnum=0 (z) includes yaw, X, Y


# def prepare_inflowgrid_from_graphs(graphfarms, batch_size=2048, edge_uv_cols=(0, 1)):
#     """Initial graph structure (from training) is a bottleneck right now and should be refactored."""
#     """
#     Prepare batched mega-graphs + inflow-grid bridge for fast VJP eval.

#     Assumes each batch forms a full (wd,ws) grid for a single layout size I:
#     - mega-graph tensors for TorchScript (one forward/backward per batch)
#     - prefilled z_buf X/Y blocks and (wd,ws)->base_lk mapping (one dP_dz per batch)
#     """
#     dev = torch.device("cpu")
#     col_u, col_v = int(edge_uv_cols[0]), int(edge_uv_cols[1])

#     cached = []
#     for g in graphfarms:
#         ei = torch.as_tensor(g.edge_index, dtype=torch.int64, device=dev)
#         ea = torch.as_tensor(g.edge_attr, dtype=torch.float32, device=dev)
#         gl = torch.as_tensor(g.globals, dtype=torch.float32, device=dev)
#         if gl.dim() == 1:
#             gl = gl[None, :]

#         if getattr(g, "num_nodes", None) is not None:
#             N = int(g.num_nodes)
#         elif getattr(g, "y", None) is not None:
#             N = int(torch.as_tensor(g.y).shape[0])
#         else:
#             N = int(ei.max().item()) + 1

#         wd = float(g["meta"]["wd_deg"])
#         ws = float(g["meta"]["ws"])
#         ti = float(g["meta"]["ti"])
#         pos = np.asarray(g["pos"], dtype=np.float32)  # (N,2)

#         theta = float(np.deg2rad(270.0 - wd))
#         cached.append(
#             dict(
#                 edge_index=ei,
#                 edge_attr0=ea,
#                 globals=gl,
#                 N=N,
#                 wd=wd,
#                 ws=ws,
#                 ti=ti,
#                 pos=pos,
#                 c=float(np.cos(theta)),
#                 s=float(np.sin(theta)),
#             )
#         )

#     prepared = []
#     for i0 in range(0, len(cached), batch_size):
#         chunk = cached[i0 : i0 + batch_size]
#         M = len(chunk)
#         if M == 0:
#             continue

#         I = int(chunk[0]["N"])  # noqa: E741
#         for c in chunk:
#             if int(c["N"]) != I:
#                 raise ValueError("Batch must have constant turbine count I")

#         # grid axes
#         wds = np.asarray(sorted({c["wd"] for c in chunk}), dtype=np.float32)  # (L,)
#         wss = np.asarray(sorted({c["ws"] for c in chunk}), dtype=np.float32)  # (K,)
#         L, K = int(wds.size), int(wss.size)
#         if M != L * K:
#             raise ValueError("Batch must be a full (wd,ws) grid: M == L*K")

#         wd_to_l = {float(wd): i for i, wd in enumerate(wds)}
#         ws_to_k = {float(ws): j for j, ws in enumerate(wss)}
#         LK = L * K
#         n = I * LK

#         # mega-graph build
#         ei_list, ea_list, src_list, dst_list, batch_list, gl_list = (
#             [],
#             [],
#             [],
#             [],
#             [],
#             [],
#         )
#         filled = np.zeros((L, K), dtype=bool)
#         base_lk = np.empty((M,), dtype=np.int64)  # (M,) base index l*K+k per graph
#         c_arr = np.empty((M,), dtype=np.float32)  # (M,)
#         s_arr = np.empty((M,), dtype=np.float32)  # (M,)

#         # packed z buffer: (3*n,) with X/Y filled here, yaw filled at runtime
#         z_buf = np.empty((3 * n,), dtype=np.float32)
#         X_flat = z_buf[n : 2 * n]
#         Y_flat = z_buf[2 * n : 3 * n]
#         TI_lk = np.empty((L, K), dtype=np.float32)  # (L,K)

#         for m, c in enumerate(chunk):
#             off = m * I

#             ei = c["edge_index"]
#             if ei.numel() == 0:
#                 raise ValueError("Empty edge_index in batch")
#             ei_off = ei + off
#             ei_list.append(ei_off)
#             src_list.append(ei_off[0])
#             dst_list.append(ei_off[1])
#             ea_list.append(c["edge_attr0"])
#             batch_list.append(torch.full((I,), m, dtype=torch.int64, device=dev))
#             gl_list.append(c["globals"])

#             l = wd_to_l[float(c["wd"])]  # noqa: E741
#             k = ws_to_k[float(c["ws"])]
#             if filled[l, k]:
#                 raise ValueError("Duplicate (wd,ws) in batch")
#             filled[l, k] = True

#             base = int(l * K + k)
#             base_lk[m] = base
#             c_arr[m] = np.float32(c["c"])
#             s_arr[m] = np.float32(c["s"])

#             pos = c["pos"]  # (I,2) expected
#             X_flat[base::LK] = pos[:, 0]
#             Y_flat[base::LK] = pos[:, 1]
#             TI_lk[l, k] = np.float32(c["ti"])

#         edge_index = torch.cat(ei_list, dim=1)
#         edge_attr0 = torch.cat(ea_list, dim=0)
#         src = torch.cat(src_list, dim=0)
#         dst = torch.cat(dst_list, dim=0)
#         batch = torch.cat(batch_list, dim=0)
#         globals_ = torch.cat(gl_list, dim=0)

#         prepared.append(
#             dict(
#                 edge_index=edge_index,
#                 edge_attr0=edge_attr0,
#                 globals=globals_,
#                 batch=batch,
#                 src=src,
#                 dst=dst,
#                 col_u=col_u,
#                 col_v=col_v,
#                 edge_attr_buf=torch.empty_like(edge_attr0),
#                 grad_uv_buf=torch.empty((M * I, 2), dtype=torch.float32, device=dev),
#                 I=I,
#                 c=c_arr,
#                 s=s_arr,
#                 base_lk=base_lk,
#                 wds=wds,
#                 wss=wss,
#                 TI_lk=TI_lk,
#                 z_buf=z_buf,
#                 v_flat=np.empty((M * I,), dtype=np.float32),
#                 direct_flat=np.empty((M * I, 2), dtype=np.float32),
#             )
#         )

#     return prepared


def gradP_vjp_xy_inflowgrid_prepared(
    ts_path,
    prepared,
    dP_dz,
    gamma_col=-1,
    uv_scale=1.0,
    return_gamma=False,
):
    """
    Evaluate dP/d(x,y) per graph using prepared mega-graph + inflow-grid partials.

    Per prepared batch:
      - 1 TorchScript forward + backward for VJP through uv edge deltas
      - 1 packed dP_dz call for v=dP/dyaw and direct=[dP/dX,dP/dY]
    Returns list of (I,2) arrays in the per-batch graph order.
    """
    dev = torch.device("cpu")
    model = torch.jit.load(ts_path, map_location=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    uv_scale = float(uv_scale)
    dP_dxy_list = []
    gamma_list = [] if return_gamma else None

    for b in prepared:
        edge_index = b["edge_index"]
        edge_attr0 = b["edge_attr0"]
        globals_ = b["globals"]
        batch = b["batch"]
        src = b["src"]
        dst = b["dst"]
        col_u = b["col_u"]
        col_v = b["col_v"]

        uv_sel = slice(col_u, col_v + 1) if (col_v == col_u + 1) else [col_u, col_v]
        duv = edge_attr0[:, uv_sel].detach().requires_grad_(True)  # (E,2)

        edge_attr_buf = b["edge_attr_buf"]
        edge_attr_buf.copy_(edge_attr0)
        edge_attr_buf[:, uv_sel] = duv

        yhat = model(edge_index, edge_attr_buf, globals_, batch)
        if yhat.dim() == 1:
            yhat = yhat[:, None]
        gamma_b = yhat[:, gamma_col]  # (M*I,)

        I = int(b["I"])  # noqa: E741
        wds = b["wds"]
        wss = b["wss"]
        LK = int(wds.size * wss.size)
        n = int(I * LK)

        z_buf = b["z_buf"]
        yaw_flat = z_buf[:n]
        base_lk = b["base_lk"]
        M = int(base_lk.size)

        gamma_np = gamma_b.detach().cpu().numpy().astype(np.float32, copy=False)
        for m in range(M):
            off = m * I
            base = int(base_lk[m])
            yaw_flat[base::LK] = gamma_np[off : off + I]
            if return_gamma:
                gamma_list.append(gamma_np[off : off + I].copy())

        g0 = np.asarray(
            dP_dz(z_buf, wds, wss, b["TI_lk"], I, int(wds.size), int(wss.size)),
            dtype=np.float32,
        )

        dP_dyaw_flat = g0[:n]
        dP_dX_flat = g0[n : 2 * n]
        dP_dY_flat = g0[2 * n : 3 * n]

        v_flat = b["v_flat"]
        direct_flat = b["direct_flat"]
        for m in range(M):
            off = m * I
            base = int(base_lk[m])
            v_flat[off : off + I] = dP_dyaw_flat[base::LK]
            direct_flat[off : off + I, 0] = dP_dX_flat[base::LK]
            direct_flat[off : off + I, 1] = dP_dY_flat[base::LK]

        v = torch.from_numpy(v_flat)  # (M*I,)
        direct = torch.from_numpy(direct_flat)  # (M*I,2)
        ell = (v * gamma_b).sum()

        g_duv = torch.autograd.grad(ell, duv, retain_graph=False, create_graph=False)[
            0
        ]  # (E,2)

        grad_uv_buf = b["grad_uv_buf"]
        grad_uv_buf.zero_()
        g_scaled = g_duv / uv_scale
        grad_uv_buf.index_add_(0, dst, g_scaled)
        grad_uv_buf.index_add_(0, src, -g_scaled)
        # this one is torch_script convention !!!

        c_arr = b["c"]
        s_arr = b["s"]
        for m in range(M):
            off = m * I
            cth = float(c_arr[m])
            sth = float(s_arr[m])

            grad_uv = grad_uv_buf[off : off + I]  # (I,2)
            grad_xy = torch.empty_like(grad_uv)
            # grad_xy[:, 0] = grad_uv[:, 0] * cth - grad_uv[:, 1] * sth
            # grad_xy[:, 1] = grad_uv[:, 0] * sth + grad_uv[:, 1] * cth
            # grad_uv: (I,2)  ->  grad_xy = R^T grad_uv
            grad_xy[:, 0] = grad_uv[:, 0] * cth + grad_uv[:, 1] * sth
            grad_xy[:, 1] = -grad_uv[:, 0] * sth + grad_uv[:, 1] * cth

            out = (
                (direct[off : off + I] + grad_xy)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            dP_dxy_list.append(out)

    return (dP_dxy_list, gamma_list) if return_gamma else dP_dxy_list


# mega-graph without to_graph calls


def _medoid_center(points_xy):
    """Return medoid center (2,) computed from full pairwise distances.

    Parameters
    ----------
    points_xy : np.ndarray
        (I, 2) turbine coordinates.

    Returns
    -------
    center : np.ndarray
        (2,) medoid center.
    """
    P = np.asarray(points_xy, dtype=np.float32)  # (I,2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)  # (I,I)
    return P[np.argmin(D.sum(axis=1))]  # (2,)


def _rotate_to_west(points_xy, wd_deg, center_xy):
    """Rotate coordinates so inflow aligns with 270 deg (west), around a fixed center.

    Parameters
    ----------
    points_xy : np.ndarray
        (I,2)
    wd_deg : float
    center_xy : np.ndarray
        (2,)

    Returns
    -------
    pos_rot : np.ndarray
        (I,2) rotated coordinates.
    c : float
        cos(theta), theta = deg2rad(270 - wd_deg)
    s : float
        sin(theta)
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    q = pts - np.asarray(center_xy, dtype=np.float32)  # (I,2)
    theta = np.deg2rad(270.0 - (float(wd_deg) % 360.0))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (q @ R), c, s  # (I,2), scalars


def _edge_index_from_rotpos(
    pos_rot,
    connectivity="wake_aware",
    rotor_diameter=284.0,
    k_wake=0.40,
    conn_distxmaxD=None,
    conn_topk=None,
):
    """Return edge_index for rotated positions.

    Parameters
    ----------
    pos_rot : np.ndarray
        (I,2) rotated coords.
    connectivity : str
        'wake_aware', 'fully_connected', or 'delaunay'
    rotor_diameter : float
    k_wake : float
    conn_distxmaxD : float | None
    conn_topk : int | None

    Returns
    -------
    edge_index : torch.Tensor
        (2,E) int64
    """
    t = torch.as_tensor(pos_rot, dtype=torch.float32)  # (I,2)
    n = int(t.shape[0])

    conn = connectivity.casefold()
    if conn == "fully_connected":
        src = torch.arange(n, dtype=torch.int64).repeat_interleave(n)
        dst = torch.arange(n, dtype=torch.int64).repeat(n)
        mask = src != dst
        return torch.stack([src[mask], dst[mask]], dim=0)

    if conn == "delaunay":
        from torch_geometric.data import Data
        from torch_geometric.transforms import Delaunay, FaceToEdge

        g = FaceToEdge()(Delaunay()(Data(pos=t)))
        return g.edge_index.to(torch.int64)

    if conn == "wake_aware":
        x = t[:, 0]  # (n,)
        y = t[:, 1]  # (n,)
        dx = x[None, :] - x[:, None]  # dx[i,j] = x_j - x_i
        dy = y[None, :] - y[:, None]
        R = 0.5 * float(rotor_diameter)
        r_wake = R + float(k_wake) * dx
        mask = (dx > 0.0) & (dy.abs() <= r_wake)
        if conn_distxmaxD is not None:
            dx_max = float(conn_distxmaxD) * float(rotor_diameter)
            mask = mask & (dx <= dx_max)
        mask.fill_diagonal_(False)

        if conn_topk is None:
            src, dst = mask.nonzero(as_tuple=True)  # directed i->j
        else:
            K = int(conn_topk)
            k_eff = min(K, n - 1)
            dist2 = dx * dx + dy * dy
            cost = dist2.masked_fill(~mask, float("inf"))
            vals, src_idx = torch.topk(cost, k=k_eff, dim=0, largest=False)  # (k_eff,n)
            dst_idx = torch.arange(n, dtype=torch.int64).view(1, n).expand_as(src_idx)
            valid = torch.isfinite(vals)
            src = src_idx[valid].to(torch.int64)
            dst = dst_idx[valid].to(torch.int64)

        if src.numel() == 0:
            return torch.empty((2, 0), dtype=torch.int64)
        return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    raise ValueError(f"Unknown connectivity: {connectivity!r}")


def prepare_inflowgrid_from_layout(
    x,
    y,
    wds,
    wss,
    TI,
    batch_size=None,
    connectivity="wake_aware",
    edge_uv_cols=(0, 1),
    rotor_diameter=284.0,
    k_wake=0.40,
    conn_distxmaxD=None,
    conn_topk=None,
):
    """
    Build 'prepared' batches directly from a single layout and an (wd,ws) inflow grid.

    Produces the same structure consumed by 'gradP_vjp_xy_inflowgrid_prepared'
    without instantiating per-case PyG Data objects.

    Notes
    -----
    Coordinates are rotated per wd so inflow is always west-aligned; therefore 'wds' stored
    for PyWake partials is a length-L vector of 270 deg (one per wd-index).
    """
    dev = torch.device("cpu")
    col_u, col_v = int(edge_uv_cols[0]), int(edge_uv_cols[1])

    pts = np.column_stack(
        [np.asarray(x, np.float32), np.asarray(y, np.float32)]
    )  # (I,2)
    I = int(pts.shape[0])  # noqa: E741

    wds_in = np.asarray(wds, dtype=np.float32)  # (L,)
    wss_in = np.asarray(wss, dtype=np.float32)  # (K,)
    L = int(wds_in.size)
    K = int(wss_in.size)
    M = int(L * K)

    # for PyWake partials in west-aligned coords
    wds_west = np.full((L,), WEST_WD, dtype=np.float32)
    TI_lk = np.full((L, K), float(TI), dtype=np.float32)

    center = _medoid_center(pts)

    # Precompute per-wd geometry once
    pos_wd = []
    eidx_wd = []
    eattr_wd = []
    c_wd = np.empty((L,), dtype=np.float32)
    s_wd = np.empty((L,), dtype=np.float32)
    for i, wd in enumerate(wds_in):
        pr, c, s = _rotate_to_west(pts, float(wd), center)  # (I,2)
        ei = _edge_index_from_rotpos(
            pr,
            connectivity=connectivity,
            rotor_diameter=rotor_diameter,
            k_wake=k_wake,
            conn_distxmaxD=conn_distxmaxD,
            conn_topk=conn_topk,
        )  # (2,E)
        src = ei[0].to(torch.int64)
        dst = ei[1].to(torch.int64)
        t = torch.as_tensor(pr, dtype=torch.float32)
        # ea = (t[dst] - t[src]).to(torch.float32)  # (E,2)
        # this is the torch_geometric convention !!!
        ea = (t[src] - t[dst]).to(torch.float32)  # (E,2)

        pos_wd.append(pr.astype(np.float32, copy=False))
        eidx_wd.append(ei)
        eattr_wd.append(ea)
        c_wd[i] = np.float32(c)
        s_wd[i] = np.float32(s)

    # One batch by default (full grid)
    if batch_size is None:
        batch_size = M
    if int(batch_size) != M:
        raise ValueError(
            "For inflow-grid partials, batch_size must equal L*K for this layout."
        )

    # Prepack z_buf with X/Y (yaw filled at runtime)
    n = I * M
    z_buf = np.empty((3 * n,), dtype=np.float32)
    X_flat = z_buf[n : 2 * n]
    Y_flat = z_buf[2 * n : 3 * n]

    # Build mega-graph tensors
    ei_list, ea_list, src_list, dst_list, batch_list, gl_list = [], [], [], [], [], []
    c_arr = np.empty((M,), dtype=np.float32)
    s_arr = np.empty((M,), dtype=np.float32)
    base_lk = np.arange(M, dtype=np.int64)  # base == case index (wd-major ordering)

    for i in range(L):
        pr = pos_wd[i]  # (I,2)
        ei0 = eidx_wd[i]  # (2,Ei)
        ea0 = eattr_wd[i]  # (Ei,2)
        for j in range(K):
            m = i * K + j
            off = m * I

            ei = ei0 + off
            ei_list.append(ei)
            src_list.append(ei[0])
            dst_list.append(ei[1])
            ea_list.append(ea0)
            batch_list.append(torch.full((I,), m, dtype=torch.int64, device=dev))
            gl_list.append(
                torch.tensor(
                    [[float(wss_in[j]), float(TI)]], dtype=torch.float32, device=dev
                )
            )

            X_flat[m::M] = pr[:, 0]
            Y_flat[m::M] = pr[:, 1]
            c_arr[m] = c_wd[i]
            s_arr[m] = s_wd[i]

    edge_index = torch.cat(ei_list, dim=1).to(torch.int64)
    edge_attr0 = torch.cat(ea_list, dim=0).to(torch.float32)
    src = torch.cat(src_list, dim=0).to(torch.int64)
    dst = torch.cat(dst_list, dim=0).to(torch.int64)
    batch = torch.cat(batch_list, dim=0).to(torch.int64)
    globals_ = torch.cat(gl_list, dim=0).to(torch.float32)  # (M,2)

    prepared = [
        dict(
            edge_index=edge_index,
            edge_attr0=edge_attr0,
            globals=globals_,
            batch=batch,
            src=src,
            dst=dst,
            col_u=col_u,
            col_v=col_v,
            edge_attr_buf=torch.empty_like(edge_attr0),
            grad_uv_buf=torch.empty((M * I, 2), dtype=torch.float32, device=dev),
            I=I,
            c=c_arr,
            s=s_arr,
            base_lk=base_lk,
            wds=wds_west,  # (L,) always 270 in west-aligned coords
            wss=wss_in,  # (K,)
            TI_lk=TI_lk,  # (L,K)
            z_buf=z_buf,  # (3*I*L*K,) with X/Y filled
            v_flat=np.empty((M * I,), dtype=np.float32),
            direct_flat=np.empty((M * I, 2), dtype=np.float32),
        )
    ]
    return prepared


# optional: full jacobian


def jac_gamma(
    ts_path,
    prepared,
    gamma_col=-1,
    uv_scale=1.0,
    return_gamma=False,
    vectorize=True,
):
    """
    Compute per-graph Jacobians J = dgamma/d(x,y) using prepared mega-graphs.

    Parameters
    ----------
    ts_path : str
        TorchScript model path, forward(edge_index, edge_attr, globals, batch) -> yhat.
    prepared : list[dict]
        Output of prepare_inflowgrid(...). Each dict is a full (wd,ws) grid batch.
    gamma_col : int
        Column index of gamma in yhat.
    uv_scale : float
        Scale used for (du,dv) columns in edge_attr. Edge updates use (dq_dst-dq_src)/uv_scale.
    return_gamma : bool
        If True, also return per-graph gamma vectors.
    vectorize : bool
        Passed to torch.autograd.functional.jacobian.

    Returns
    -------
    J_list : list[np.ndarray]
        List of per-graph Jacobians, each shape (I, I, 2), in physical (x,y).
        Axis meanings: [out_node, in_node, (dx,dy)].
    gamma_list : list[np.ndarray], optional
        If return_gamma=True: per-graph gamma vectors, each shape (I,).
    """
    dev = torch.device("cpu")
    model = torch.jit.load(ts_path, map_location=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    uv_scale = float(uv_scale)

    J_list = []
    gamma_list = [] if return_gamma else None

    for b in prepared:
        edge_index = b["edge_index"]
        edge_attr0 = b["edge_attr0"]
        globals_ = b["globals"]
        batch = b["batch"]
        src = b["src"]
        dst = b["dst"]
        col_u = int(b["col_u"])
        col_v = int(b["col_v"])
        I = int(b["I"])  # noqa: E741

        base_lk = b["base_lk"]
        M = int(base_lk.size)  # number of graphs/cases in this batch
        Ntot = int(M * I)

        c_arr = b["c"]  # (M,)
        s_arr = b["s"]  # (M,)

        def _gamma_from_dq(dq):
            # PyG Cartesian!: duv = pos[src] - pos[dst]  =>  d(duv) = dq[src] - dq[dst]
            dUV = (dq[src] - dq[dst]) / uv_scale  # (E,2)

            edge_attr = edge_attr0.clone()
            edge_attr[:, col_u] = edge_attr0[:, col_u] + dUV[:, 0]
            edge_attr[:, col_v] = edge_attr0[:, col_v] + dUV[:, 1]

            yhat = model(edge_index, edge_attr, globals_, batch)
            if yhat.dim() == 1:
                yhat = yhat[:, None]
            return yhat[:, gamma_col]

        dq0 = torch.zeros(
            (Ntot, 2), dtype=torch.float32, device=dev, requires_grad=True
        )

        gamma_b = _gamma_from_dq(dq0)  # (Ntot,)
        if return_gamma:
            gamma_np = gamma_b.detach().cpu().numpy().astype(np.float32, copy=False)
            for m in range(M):
                off = m * I
                gamma_list.append(gamma_np[off : off + I].copy())

        # J_uv: (Ntot, Ntot, 2) where last dim is (du,dv)
        J_uv = torch.autograd.functional.jacobian(
            _gamma_from_dq, dq0, vectorize=vectorize
        )
        J_uv = J_uv.detach().cpu().numpy().astype(np.float32, copy=False)

        # Split into per-graph blocks and rotate input-grad dim (du,dv)->(dx,dy) via R^T
        for m in range(M):
            off = m * I
            Jm_uv = J_uv[off : off + I, off : off + I, :]  # (I,I,2)

            cth = float(c_arr[m])
            sth = float(s_arr[m])

            Jm_xy = np.empty_like(Jm_uv)
            # lazy rotation
            Jm_xy[:, :, 0] = Jm_uv[:, :, 0] * cth - Jm_uv[:, :, 1] * sth
            Jm_xy[:, :, 1] = Jm_uv[:, :, 0] * sth + Jm_uv[:, :, 1] * cth

            J_list.append(Jm_xy)

    return (J_list, gamma_list) if return_gamma else J_list
