import torch
from design_friendly.utils.misc import log_execution_time
from design_friendly.utils.graph import _WEST_WD
from py_wake import numpy as np
from py_wake.utils.gradients import autograd
from design_friendly.models import models_filepath

default_ts_path = models_filepath + "torchscript26a.pt"


# TODO: highlight/refactor var names for rotated coordinates
@log_execution_time
def make_dP_dz_inflowgrid(wf_model, time=False):  # PyWake partials
    """
    Return a packed gradient function dP/dz for total power over an (I,L,K) inflow grid
    (memory intensive).

    z packs three (I,L,K) fields: yaw, X, Y into one vector:
      n = I*L*K
      yaw = z[0:n].reshape(I,L,K)
      Xrot_ilk = z[n:2*n].reshape(I,L,K) (_ik for time=True)
      Yrot_ilk = z[2*n:3*n].reshape(I,L,K) (_ik for time=True)
      Coordinates are rotated per-wd to align western inflow and stacked (memory-intensive)

    Returns
    -------
    dP_dz : callable
        dP_dz(z, wds, wss, TI_lk, I, L, K) -> grad_z, shape (3*I*L*K,)
    """

    def power_inflowgrid(z, wds, wss, TI_lk, I, L, K):
        """
        Scalar objective used by dP_dz: sum of wf_model power over all (i,l,k).

        Parameters
        ----------
        z : (3*I*L*K,)  !wrt argument.
            Packed [yaw, Xrot, Yrot] as described in make_dP_dz_inflowgrid.
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
        Xrot_ilk = z[n : 2 * n].reshape(I, L, K)
        Yrot_ilk = z[2 * n : 3 * n].reshape(I, L, K)
        # Sequence mode: the only supported time=True encoding here is L==1, K==T
        if L == 1 and K > 1:
            assert time is True
            yaw_it = yaw[:, 0, :]  # (I,K)
            X_it = Xrot_ilk[:, 0, :]  # (I,K)
            Y_it = Yrot_ilk[:, 0, :]  # (I,K)

            wd_t = np.full((K,), _WEST_WD, dtype=np.float32)  # (K,)
            ws_t = wss  # (K,)
            TI_t = TI_lk[0, :]  # (K,)
            _, _, power_ik, _, _, _ = wf_model(
                x=X_it,
                y=Y_it,
                wd=wd_t,
                ws=ws_t,
                TI=TI_t,  # tood: if this causes issues, use TI=float(TI_t[0])
                yaw=yaw_it,
                tilt=0,
                return_simulationResult=False,
                n_cpu=1,
                time=True,
            )
            return power_ik.sum()

        assert time is False
        _, _, power_ilk, _, _, _ = wf_model(
            # coordinates are rotated per-wd to align western inflow and stacked (X_ilk)
            x=Xrot_ilk,
            y=Yrot_ilk,
            # wd=wds,
            # west-aligned coords for rotated layout
            # ASSUMES ROTATED COORDINATES! Since this is called vjp
            wd=np.full((L,), _WEST_WD, dtype=np.float32),
            ws=wss,
            TI=TI_lk,
            yaw=yaw,
            tilt=0,
            return_simulationResult=False,
            n_cpu=1,
            time=False,
        )
        return power_ilk.sum()

    return autograd(
        power_inflowgrid, argnum=0, vector_interdependence=True
    )  # argnum=0 (z) includes yaw, X, Y


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
      - TorchScript forward + backward for VJP through uv edge deltas
      - packed dP_dz call for v=dP/dyaw and direct=[dP/dX,dP/dY]
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

        I = int(b["I"])
        wds = b["wds"]
        wss = b["wss"]
        LK = int(wds.size * wss.size)
        n = int(I * LK)

        z_buf = b["z_buf"]
        yaw_flat = z_buf[:n]
        base_lk = b["base_lk"]
        M = int(base_lk.size)

        # Fill yaw into z_buf (base_lk == arange(M) always, so this is a transpose)
        gamma_np = gamma_b.detach().cpu().numpy().astype(np.float32, copy=False)
        yaw_flat[:n] = gamma_np.reshape(M, I).T.ravel()  # (I, M) layout in z_buf
        if return_gamma:
            for row in gamma_np.reshape(M, I):
                gamma_list.append(row.copy())

        g0 = np.asarray(
            dP_dz(z_buf, wds, wss, b["TI_lk"], I, int(wds.size), int(wss.size)),
            dtype=np.float32,
        )

        # g0 packs dP/d[yaw, X, Y], each shaped (I, L, K) = (I, M) since L*K == M
        dP_dyaw_flat = g0[:n]
        dP_dX_flat = g0[n : 2 * n]
        dP_dY_flat = g0[2 * n : 3 * n]

        # Scatter from (I, M) PyWake layout to (M, I) VJP layout (transpose)
        v_flat = b["v_flat"]
        direct_flat = b["direct_flat"]
        v_flat[:] = dP_dyaw_flat.reshape(I, M).T.ravel()
        direct_flat[:, 0] = dP_dX_flat.reshape(I, M).T.ravel()
        direct_flat[:, 1] = dP_dY_flat.reshape(I, M).T.ravel()

        v = torch.from_numpy(v_flat)  # (M*I,)
        direct = torch.from_numpy(direct_flat)  # (M*I, 2)
        ell = (v * gamma_b).sum()

        g_duv = torch.autograd.grad(ell, duv, retain_graph=False, create_graph=False)[
            0
        ]  # (E, 2)

        grad_uv_buf = b["grad_uv_buf"]
        grad_uv_buf.zero_()
        g_scaled = g_duv / uv_scale
        grad_uv_buf.index_add_(0, src, g_scaled)
        grad_uv_buf.index_add_(0, dst, -g_scaled)

        # Vectorised unrotation: sum GNN + direct in rotated space, then apply R^T
        c_arr = b["c"]
        s_arr = b["s"]
        c_t = torch.from_numpy(c_arr).view(M, 1)  # (M, 1)
        s_t = torch.from_numpy(s_arr).view(M, 1)  # (M, 1)
        total_uv = (direct + grad_uv_buf).view(M, I, 2)  # (M, I, 2)
        out_u = total_uv[:, :, 0] * c_t - total_uv[:, :, 1] * s_t  # (M, I)
        out_v = total_uv[:, :, 0] * s_t + total_uv[:, :, 1] * c_t  # (M, I)
        out_np = (
            torch.stack([out_u, out_v], dim=-1)  # (M, I, 2)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        dP_dxy_list.extend(out_np)  # appends M arrays of shape (I, 2)

    return (dP_dxy_list, gamma_list) if return_gamma else dP_dxy_list


def _rotation_components(wd_deg):
    """Return cos/sin for the west-aligning rotation used by graph.py."""
    theta = np.deg2rad(_WEST_WD - (float(wd_deg) % 360.0))
    return np.float32(np.cos(theta)), np.float32(np.sin(theta))


def jac_gamma(
    ts_path,
    prepared,
    gamma_col=-1,
    uv_scale=1.0,
    return_gamma=False,
    vectorize=True,
):
    """
    Compute per-graph full Jacobians J = dgamma/d(x,y) using prepared mega-graphs.

    Parameters
    ----------
    ts_path : str
        TorchScript model path, forward(edge_index, edge_attr, globals, batch) -> yhat.
    prepared : list[dict]
        Output of prepare_from_graphs(). Each dict is a full (wd,ws) grid batch.
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
        List of per-graph full Jacobians, each shape (I, I, 2), in physical (x,y).
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
        I = int(b["I"])

        base_lk = b["base_lk"]
        M = int(base_lk.size)  # number of graphs/cases in this batch
        Ntot = int(M * I)

        c_arr = b["c"]  # (M,)
        s_arr = b["s"]  # (M,)

        def _gamma_from_dq(dq):
            # PyG Cartesian!: duv = pos[src] - pos[dst]  ->  d(duv) = dq[src] - dq[dst]
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


def prepare_from_graphs(
    graphs,
    lut=True,
    n_wd=None,
    edge_uv_cols=(0, 1),
):
    """Build the VJP 'prepared' mega-graph structure from pre-built PyG Data graphs.

    Drop-in replacement for prepare_inflowgrid_from_layout (lut=True) and
    prepare_inflowseq_from_layout (lut=False) when graphs come from graph_maker().

    Parameters
    ----------
    graphs : list[Data]
        Output of graph_maker(). Each graph must have .pos (n_wt, 2) rotated coords,
        .edge_index, .edge_attr, .globals ([WS, TI]), and .meta with 'wd_deg', 'ws', 'ti'.
    lut : bool, default True
        True -> wdxws grid, graphs in wd-major order (same as graph_maker(..., lut=True)).
        False -> time-series, one graph per timestep (graph_maker(..., lut=False)).
    n_wd : int or None
        Number of unique WD values. Required when lut=True.
    edge_uv_cols : tuple[int, int]
        Columns of edge_attr holding (u, v) Cartesian deltas.

    Returns
    -------
    list[dict]
        Single-element list with the prepared mega-graph dict, compatible with
        gradP_vjp_xy_inflowgrid_prepared() and jac_gamma().
    """
    dev = torch.device("cpu")
    col_u, col_v = int(edge_uv_cols[0]), int(edge_uv_cols[1])

    M = len(graphs)
    I = int(graphs[0].pos.shape[0])
    assert all(int(g.pos.shape[0]) == I for g in graphs), (
        "All graphs must have the same number of turbines"
    )

    if lut:
        assert n_wd is not None, "n_wd is required when lut=True"
        L = int(n_wd)
        K = M // L
        assert L * K == M, f"len(graphs) {M} must equal n_wd*n_ws"
        wds_west = np.full((L,), _WEST_WD, dtype=np.float32)
        wss_in = np.array([graphs[j].meta["ws"] for j in range(K)], dtype=np.float32)
        TI_lk = np.array(
            [[graphs[i * K + j].meta["ti"] for j in range(K)] for i in range(L)],
            dtype=np.float32,
        )  # (L, K)
    else:
        # time-series: L=1, K=T
        L, K = 1, M
        wds_west = np.full((1,), _WEST_WD, dtype=np.float32)
        wss_in = np.array([g.meta["ws"] for g in graphs], dtype=np.float32)  # (M,)
        TI_lk = np.array([[g.meta["ti"] for g in graphs]], dtype=np.float32)  # (1, M)

    # vectorized: rotation components, z_buf positions, and mega-graph assembly
    c_arr = np.empty((M,), dtype=np.float32)
    s_arr = np.empty((M,), dtype=np.float32)
    # z_buf: [yaw (filled at runtime) | X | Y], each segment (I, M) in PyWake order
    n = I * M
    z_buf = np.empty((3 * n,), dtype=np.float32)
    X_mat = z_buf[n : 2 * n].reshape(I, M)
    Y_mat = z_buf[2 * n : 3 * n].reshape(I, M)
    ei_list, ea_list, src_list, dst_list, batch_list, gl_list = [], [], [], [], [], []
    # Assemble mega-graph tensors (M graphs concatenated, node indices offset by m*I)
    for m, g in enumerate(graphs):
        c_arr[m], s_arr[m] = _rotation_components(g.meta["wd_deg"])
        pos = g.pos.detach().cpu().numpy().astype(np.float32, copy=False)
        X_mat[:, m] = pos[:, 0]
        Y_mat[:, m] = pos[:, 1]
        ei = g.edge_index.to(torch.int64) + m * I
        ea = g.edge_attr.to(torch.float32)
        ei_list.append(ei)
        src_list.append(ei[0])
        dst_list.append(ei[1])
        ea_list.append(ea)
        batch_list.append(torch.full((I,), m, dtype=torch.int64, device=dev))
        gl_list.append(g.globals.to(torch.float32).reshape(1, -1))  # (1, G)

    edge_index = torch.cat(ei_list, dim=1)
    edge_attr0 = torch.cat(ea_list, dim=0)
    src = torch.cat(src_list, dim=0)
    dst = torch.cat(dst_list, dim=0)
    batch = torch.cat(batch_list, dim=0)
    globals_ = torch.cat(gl_list, dim=0)  # (M, 2)

    return [
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
            base_lk=np.arange(M, dtype=np.int64),
            wds=wds_west,
            wss=wss_in,
            TI_lk=TI_lk,
            z_buf=z_buf,
            v_flat=np.empty((M * I,), dtype=np.float32),
            direct_flat=np.empty((M * I, 2), dtype=np.float32),
        )
    ]
