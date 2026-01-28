import torch
from design_friendly.utils.misc import log_execution_time
from py_wake import numpy as np
from py_wake.utils.gradients import autograd


@log_execution_time
def gradP_torchscript_vjp_xy(
    ts_path,
    graphfarms,
    partials_fn,
    batch_size=2048,
    edge_uv_cols=(0, 1),
    gamma_col=-1,
    uv_scale=1.0,  # or D for normalized graphs
    return_gamma=False,
):
    """Compute dP/d(x,y) using engineering partials + one VJP through the surrogate (no Jacobian).

    Parameters
    ----------
    ts_path : str
        TorchScript model path with signature forward(edge_index, edge_attr, globals, batch) -> yhat.
    graphfarms : sequence
        Graphs with:
          - edge_index: (2, E)
          - edge_attr:  (E, Fe), with (Δu,Δv) columns at edge_uv_cols (stream/cross deltas)
          - globals:    (Fg,) or (1,Fg)
          - g['meta']['wd_deg']: wind direction (meteorological / PyWake convention)
    partials_fn : callable
        Called as: out = partials_fn(g, gamma_np)
        Must return dict with:
          - 'dP_dgamma': (N,)
          - either 'dP_dxy': (N,2) or 'dP_dx': (N,) and 'dP_dy': (N,)
    batch_size : int
        Graphs per iteration chunk.
    edge_uv_cols : tuple[int,int]
        Column indices in edge_attr corresponding to (Δu,Δv).
    gamma_col : int
        Index of gamma in the model outputs (default last column).
    uv_scale : float
        If edge_attr stores normalized deltas, e.g. Δu/uv_scale, set uv_scale accordingly.
    return_gamma : bool
        If True, also return list of gamma arrays (N,).

    Notes
    -----
    Math implemented per case (fixed u, I):
      dP/dx = (∂P/∂x)|_{γ*} + (∂P/∂γ)|_{x*} (∂γ/∂x)|_{γ*}
    We avoid forming (∂γ/∂x) by computing a VJP:
      v := (∂P/∂γ)|_{x*}  (from engineering model)
      ℓ(x) := v^T γ(x)    (scalar)
      ∇_x ℓ(x*) = v^T (∂γ/∂x)|_{γ*}
    - Coordinates affect the surrogate only through edge Cartesian deltas (Δu,Δv) columns in edge_attr.
    - We inject node perturbations dq=(du,dv) and update edge deltas:
        Δu_e <- Δu_e + (du_dst - du_src)/uv_scale
        Δv_e <- Δv_e + (dv_dst - dv_src)/uv_scale
    - Backprop ∂ℓ/∂dq gives ∂ℓ/∂(u,v) per node; rotate to (x,y) using PyWake convention.

    Returns
    -------
    dP_dxy_list : list[np.ndarray]
        Each entry has shape (N,2): [:,0]=dP/dx, [:,1]=dP/dy.
    gamma_list : list[np.ndarray], optional
        Returned if return_gamma=True; each is (N,).
    """
    dev = torch.device("cpu")  # ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(ts_path, map_location=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    col_u, col_v = int(edge_uv_cols[0]), int(edge_uv_cols[1])
    uv_scale = float(uv_scale)

    def theta_from_wd_deg(wd_deg):
        return np.deg2rad(270.0 - float(wd_deg))

    def rotate_grad_uv_to_xy(grad_uv, theta):
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        grad_xy = torch.empty_like(grad_uv)
        grad_xy[:, 0] = grad_uv[:, 0] * c - grad_uv[:, 1] * s
        grad_xy[:, 1] = grad_uv[:, 0] * s + grad_uv[:, 1] * c
        return grad_xy

    dP_dxy_list = []
    gamma_list = [] if return_gamma else None
    n_cases = len(graphfarms)

    Fe = None
    e_u = None
    e_v = None

    for i0 in range(0, n_cases, batch_size):
        i1 = min(i0 + batch_size, n_cases)
        for i in range(i0, i1):
            g = graphfarms[i]
            theta = theta_from_wd_deg(g["meta"]["wd_deg"])

            edge_index = torch.as_tensor(g.edge_index, dtype=torch.int64, device=dev)
            edge_attr0 = torch.as_tensor(g.edge_attr, dtype=torch.float32, device=dev)
            globals_ = torch.as_tensor(g.globals, dtype=torch.float32, device=dev)
            if globals_.dim() == 1:
                globals_ = globals_[None, :]
            # node count
            if getattr(g, "num_nodes", None) is not None:
                N = int(g.num_nodes)
            elif getattr(g, "y", None) is not None:
                N = int(torch.as_tensor(g.y).shape[0])
            elif edge_index.numel() > 0:  # silly
                N = int(edge_index.max().item()) + 1
            else:
                raise RuntimeError("Cannot infer N")
            batch = torch.zeros((N,), dtype=torch.int64, device=dev)
            dq = torch.zeros(
                (N, 2), dtype=torch.float32, device=dev, requires_grad=True
            )
            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]
                dUV_e = (dq[dst] - dq[src]) / uv_scale  # (E,2)
                if Fe is None:
                    Fe = int(edge_attr0.shape[1])
                    e_u = torch.zeros((Fe,), dtype=torch.float32, device=dev)
                    e_v = torch.zeros((Fe,), dtype=torch.float32, device=dev)
                    e_u[col_u] = 1.0
                    e_v[col_v] = 1.0
                edge_attr = (
                    edge_attr0
                    + dUV_e[:, 0:1] * e_u[None, :]
                    + dUV_e[:, 1:2] * e_v[None, :]
                )
            else:
                edge_attr = edge_attr0
            yhat = model(edge_index, edge_attr, globals_, batch)
            if yhat.dim() == 1:
                yhat = yhat[:, None]
            gamma = yhat[:, gamma_col]
            gamma_np = gamma.detach().cpu().numpy().astype(np.float32, copy=False)
            out = partials_fn(g, gamma_np)
            v = torch.as_tensor(out["dP_dgamma"], dtype=torch.float32, device=dev)
            if "dP_dxy" in out:
                dP_dxy_direct = np.asarray(out["dP_dxy"], dtype=np.float32)
            else:
                dP_dx = np.asarray(out["dP_dx"], dtype=np.float32)
                dP_dy = np.asarray(out["dP_dy"], dtype=np.float32)
                dP_dxy_direct = np.stack([dP_dx, dP_dy], axis=1)
            ell = (v * gamma).sum()
            if ell.requires_grad:
                grad_uv = torch.autograd.grad(
                    ell, dq, retain_graph=False, create_graph=False
                )[0]
            else:
                grad_uv = torch.zeros_like(dq)
            grad_xy = (
                rotate_grad_uv_to_xy(grad_uv, theta)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            dP_dxy_list.append(dP_dxy_direct + grad_xy)
            if return_gamma:
                gamma_list.append(gamma_np)
    if return_gamma:
        return dP_dxy_list, gamma_list
    return dP_dxy_list


def dummy_partials_fn(g, gamma_np):
    # reuse forward predictions from torchscript
    N = len(gamma_np)
    dP_dgamma = np.ones(N, dtype=np.float32)
    dP_dxy = np.zeros((N, 2), dtype=np.float32)
    return {"dP_dgamma": dP_dgamma, "dP_dxy": dP_dxy}


@log_execution_time
def prepare_gradP_vjp_xy(
    graphfarms,
    batch_size=2048,
    edge_uv_cols=(0, 1),
):
    """Prepack graphs into batches for repeated grad evaluations.

    Parameters
    ----------
    graphfarms : sequence
        Graphs with:
          - edge_index: (2, E)
          - edge_attr:  (E, Fe)
          - globals:    (Fg,) or (1,Fg)
          - meta['wd_deg']
    batch_size : int
        Graphs per prepared batch.
    edge_uv_cols : tuple[int,int]
        (col_u, col_v) indices in edge_attr.

    Returns
    -------
    prepared : list[dict]
        Each dict contains a prepacked mega-graph batch plus per-graph offsets.
    """
    dev = torch.device("cpu")
    col_u, col_v = int(edge_uv_cols[0]), int(edge_uv_cols[1])

    def infer_num_nodes(g, ei):
        if getattr(g, "num_nodes", None) is not None:
            return int(g.num_nodes)
        if getattr(g, "y", None) is not None:
            return int(torch.as_tensor(g.y).shape[0])
        if ei.numel() > 0:
            return int(ei.max().item()) + 1
        raise RuntimeError("Cannot infer N")

    # Per-graph tensor cache
    cached = []
    for g in graphfarms:
        ei = torch.as_tensor(g.edge_index, dtype=torch.int64, device=dev)
        ea = torch.as_tensor(g.edge_attr, dtype=torch.float32, device=dev)
        gl = torch.as_tensor(g.globals, dtype=torch.float32, device=dev)
        if gl.dim() == 1:
            gl = gl[None, :]
        N = infer_num_nodes(g, ei)

        theta = float(np.deg2rad(270.0 - float(g["meta"]["wd_deg"])))
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        cached.append(
            dict(g=g, edge_index=ei, edge_attr0=ea, globals=gl, N=N, c=c, s=s)
        )

    prepared = []
    for i0 in range(0, len(cached), batch_size):
        chunk = cached[i0 : min(i0 + batch_size, len(cached))]
        Ns = [c["N"] for c in chunk]
        offsets = np.cumsum([0] + Ns[:-1]).astype(np.int64)
        total_N = int(sum(Ns))
        ei_list = []
        ea0_list = []
        src_list = []
        dst_list = []
        batch_list = []
        globals_list = []

        for k, c in enumerate(chunk):
            off = int(offsets[k])
            ei = c["edge_index"]
            if ei.numel() > 0:
                ei_off = ei + off
                ei_list.append(ei_off)
                src_list.append(ei_off[0])
                dst_list.append(ei_off[1])
            ea0_list.append(c["edge_attr0"])
            batch_list.append(torch.full((c["N"],), k, dtype=torch.int64, device=dev))
            globals_list.append(c["globals"])

        edge_index_b = (
            torch.cat(ei_list, dim=1)
            if len(ei_list)
            else torch.empty((2, 0), dtype=torch.int64)
        )
        edge_attr0_b = (
            torch.cat(ea0_list, dim=0)
            if len(ea0_list)
            else torch.empty((0, chunk[0]["edge_attr0"].shape[1]), dtype=torch.float32)
        )
        batch_b = torch.cat(batch_list, dim=0)
        globals_b = torch.cat(globals_list, dim=0)
        src_b = (
            torch.cat(src_list, dim=0)
            if len(src_list)
            else torch.empty((0,), dtype=torch.int64)
        )
        dst_b = (
            torch.cat(dst_list, dim=0)
            if len(dst_list)
            else torch.empty((0,), dtype=torch.int64)
        )

        # Reusable buffers (avoid per-iteration allocations)
        edge_attr_buf = torch.empty_like(edge_attr0_b)  # (E,Fe)
        grad_uv_buf = torch.empty((total_N, 2), dtype=torch.float32)  # (total_N,2)
        v_buf = torch.empty((total_N,), dtype=torch.float32)  # (total_N,)
        direct_buf = torch.empty((total_N, 2), dtype=torch.float32)  # (total_N,2)

        prepared.append(
            dict(
                chunk=chunk,
                Ns=Ns,
                offsets=offsets,
                total_N=total_N,
                edge_index=edge_index_b,
                edge_attr0=edge_attr0_b,
                globals=globals_b,  # sketchy
                batch=batch_b,
                src=src_b,
                dst=dst_b,
                col_u=col_u,
                col_v=col_v,
                edge_attr_buf=edge_attr_buf,
                grad_uv_buf=grad_uv_buf,
                v_buf=v_buf,
                direct_buf=direct_buf,
            )
        )
    return prepared


@log_execution_time
def gradP_torchscript_vjp_xy_prepared(
    ts_path,
    prepared,
    partials_fn,
    gamma_col=-1,
    uv_scale=1.0,
    return_gamma=False,
):
    """Run dP/d(x,y) using prepared batches (batched forward/backward + edge-Δu/Δv VJP).

    Parameters
    ----------
    ts_path : str
        TorchScript path.
    prepared : list[dict]
        Output of prepare_gradP_vjp_xy_cpu(...).
    partials_fn : callable
        out = partials_fn(g, gamma_np) -> dict with dP_dgamma and dP_dxy or (dP_dx,dP_dy).
    gamma_col : int
        Gamma column in yhat.
    uv_scale : float
        Scaling used in edge_attr for Δu,Δv.
    return_gamma : bool
        If True, also return gamma arrays.

    Returns
    -------
    dP_dxy_list : list[np.ndarray]
        Per-graph arrays (N,2).
    gamma_list : list[np.ndarray], optional
        Per-graph arrays (N,).
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

        edge_attr_buf = b["edge_attr_buf"]
        grad_uv_buf = b["grad_uv_buf"]
        v_buf = b["v_buf"]
        direct_buf = b["direct_buf"]

        # Build differentiable Δu/Δv with minimal allocation:
        # If cols are contiguous, slice is a view; otherwise advanced indexing makes a copy.
        if col_v == col_u + 1:
            duv = edge_attr0[:, col_u : col_v + 1].detach()
        else:
            duv = edge_attr0[:, [col_u, col_v]].detach()

        duv = duv.requires_grad_(True)  # (E,2)

        # Reuse buffer: copy base edge_attr, then write duv into uv cols
        edge_attr_buf.detach_()
        edge_attr_buf.copy_(edge_attr0)
        if col_v == col_u + 1:
            edge_attr_buf[:, col_u : col_v + 1] = duv
        else:
            edge_attr_buf[:, [col_u, col_v]] = duv

        yhat = model(edge_index, edge_attr_buf, globals_, batch)
        if yhat.dim() == 1:
            yhat = yhat[:, None]
        gamma_b = yhat[:, gamma_col]  # (total_N,)

        # Fill v_buf and direct_buf per-graph without cat allocations
        for k, c in enumerate(b["chunk"]):
            off = int(b["offsets"][k])
            N = int(c["N"])
            gk = c["g"]

            gamma_k = gamma_b[off : off + N]
            gamma_np = gamma_k.detach().numpy().astype(np.float32, copy=False)
            out = partials_fn(gk, gamma_np)

            v_k = np.asarray(out["dP_dgamma"], dtype=np.float32)
            if "dP_dxy" in out:
                direct_k = np.asarray(out["dP_dxy"], dtype=np.float32)
            else:
                dP_dx = np.asarray(out["dP_dx"], dtype=np.float32)
                dP_dy = np.asarray(out["dP_dy"], dtype=np.float32)
                direct_k = np.stack([dP_dx, dP_dy], axis=1).astype(
                    np.float32, copy=False
                )

            v_buf[off : off + N] = torch.from_numpy(v_k)
            direct_buf[off : off + N] = torch.from_numpy(direct_k)

            if return_gamma:
                gamma_list.append(gamma_np)

        ell = (v_buf * gamma_b).sum()

        if duv.numel() > 0 and ell.requires_grad:
            g_duv = torch.autograd.grad(
                ell, duv, retain_graph=False, create_graph=False
            )[0]  # (E,2)

            grad_uv_buf.zero_()
            g_scaled = g_duv / uv_scale
            grad_uv_buf.index_add_(0, dst, g_scaled)
            grad_uv_buf.index_add_(0, src, -g_scaled)
        else:
            grad_uv_buf.zero_()

        # Split, rotate, add direct term
        for k, c in enumerate(b["chunk"]):
            off = int(b["offsets"][k])
            N = int(c["N"])
            cth = float(c["c"])
            sth = float(c["s"])

            grad_uv = grad_uv_buf[off : off + N]  # (N,2)

            grad_xy = torch.empty_like(grad_uv)
            grad_xy[:, 0] = grad_uv[:, 0] * cth - grad_uv[:, 1] * sth
            grad_xy[:, 1] = grad_uv[:, 0] * sth + grad_uv[:, 1] * cth

            dP_dxy = direct_buf[off : off + N] + grad_xy
            dP_dxy = dP_dxy.detach().numpy().astype(np.float32, copy=False)
            dP_dxy_list.append(dP_dxy)

    if return_gamma:
        return dP_dxy_list, gamma_list
    return dP_dxy_list


def compare_gradP_outputs_from_lists(
    dP_baseline_list,
    dP_prepared_list,
    gamma_baseline_list=None,
    gamma_prepared_list=None,
    *,
    atol=1e-6,
    rtol=1e-5,
):
    n_cases = len(dP_baseline_list)
    if len(dP_prepared_list) != n_cases:
        raise ValueError(
            f"Length mismatch: baseline={n_cases}, prepared={len(dP_prepared_list)}"
        )

    compare_gamma = (gamma_baseline_list is not None) or (
        gamma_prepared_list is not None
    )
    if compare_gamma:
        if gamma_baseline_list is None or gamma_prepared_list is None:
            raise ValueError(
                "Provide both gamma_baseline_list and gamma_prepared_list, or neither."
            )
        if len(gamma_baseline_list) != n_cases or len(gamma_prepared_list) != n_cases:
            raise ValueError("Gamma list length mismatch with dP lists.")

    eps = 1e-12

    ok_dP = True
    ok_g = True if compare_gamma else None

    per_case = []

    # aggregate accumulators for dP
    sum_abs = 0.0
    sum_sq = 0.0
    sum_ref_sq = 0.0
    max_abs = 0.0
    n_elem = 0

    # aggregate accumulators for gamma
    sum_abs_g = 0.0
    sum_sq_g = 0.0
    sum_ref_sq_g = 0.0
    max_abs_g = 0.0
    n_elem_g = 0

    for i in range(n_cases):
        a = np.asarray(dP_baseline_list[i], dtype=np.float64)  # reference
        b = np.asarray(dP_prepared_list[i], dtype=np.float64)  # test

        if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(
                f"Case {i}: dP shape mismatch or invalid: {a.shape} vs {b.shape}"
            )

        diff = b - a
        absdiff = np.abs(diff)

        case_max = float(absdiff.max()) if absdiff.size else 0.0
        case_mae = float(absdiff.mean()) if absdiff.size else 0.0
        case_rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0

        ref_l2 = float(np.sqrt(np.sum(a * a)))
        diff_l2 = float(np.sqrt(np.sum(diff * diff)))
        case_rel_l2 = diff_l2 / max(ref_l2, eps)

        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            ok_dP = False

        sum_abs += float(absdiff.sum())
        sum_sq += float(np.sum(diff * diff))
        sum_ref_sq += float(np.sum(a * a))
        max_abs = max(max_abs, case_max)
        n_elem += int(a.size)

        row = dict(
            case=i,
            N=int(a.shape[0]),
            dP_dxy_MAX_abs=case_max,
            dP_dxy_mae=case_mae,
            dP_dxy_rmse=case_rmse,
            dP_dxy_rel_l2=case_rel_l2,
        )

        if compare_gamma:
            ga = np.asarray(gamma_baseline_list[i], dtype=np.float64)
            gb = np.asarray(gamma_prepared_list[i], dtype=np.float64)

            if ga.shape != gb.shape or ga.ndim != 1:
                raise ValueError(
                    f"Case {i}: gamma shape mismatch or invalid: {ga.shape} vs {gb.shape}"
                )

            gdiff = gb - ga
            gabs = np.abs(gdiff)

            g_case_max = float(gabs.max()) if gabs.size else 0.0
            g_case_mae = float(gabs.mean()) if gabs.size else 0.0
            g_case_rmse = float(np.sqrt(np.mean(gdiff * gdiff))) if gdiff.size else 0.0

            g_ref_l2 = float(np.sqrt(np.sum(ga * ga)))
            g_diff_l2 = float(np.sqrt(np.sum(gdiff * gdiff)))
            g_case_rel_l2 = g_diff_l2 / max(g_ref_l2, eps)

            if not np.allclose(ga, gb, rtol=rtol, atol=atol, equal_nan=True):
                ok_g = False

            sum_abs_g += float(gabs.sum())
            sum_sq_g += float(np.sum(gdiff * gdiff))
            sum_ref_sq_g += float(np.sum(ga * ga))
            max_abs_g = max(max_abs_g, g_case_max)
            n_elem_g += int(ga.size)

            row.update(
                gamma_MAX_abs=g_case_max,
                gamma_mae=g_case_mae,
                gamma_rmse=g_case_rmse,
                gamma_rel_l2=g_case_rel_l2,
            )

        per_case.append(row)

    summary = dict(
        dP_dxy_MAX_abs=max_abs,
        dP_dxy_mae=(sum_abs / max(n_elem, 1)),
        dP_dxy_rmse=float(np.sqrt(sum_sq / max(n_elem, 1))),
        dP_dxy_rel_l2=float(np.sqrt(sum_sq) / max(np.sqrt(sum_ref_sq), eps)),
    )

    if compare_gamma:
        summary.update(
            gamma_MAX_abs=max_abs_g,
            gamma_mae=(sum_abs_g / max(n_elem_g, 1)),
            gamma_rmse=float(np.sqrt(sum_sq_g / max(n_elem_g, 1))),
            gamma_rel_l2=float(np.sqrt(sum_sq_g) / max(np.sqrt(sum_ref_sq_g), eps)),
        )

    return dict(
        ok_dP_dxy=ok_dP,
        ok_gamma=ok_g,
        summary=summary,
        per_case=per_case,
    )


# new vjp  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`~~>

from py_wake import numpy as np
from py_wake.utils.gradients import autograd


@log_execution_time
def make_dP_dz_inflowgrid(wf_model):
    """Create one autograd function for packed gradients on an (I,L,K) inflow grid.

    Returns
    -------
    dP_dz : callable
        dP_dz(z, wds, wss, TI_lk, I, L, K) -> grad_z, shape (3*I*L*K,)
    """

    def power_inflowgrid(z, wds, wss, TI_lk, I, L, K):
        # z: (3*I*L*K,)
        n = I * L * K
        yaw = z[0:n].reshape(I, L, K)  # (I,L,K)
        X = z[n : 2 * n].reshape(I, L, K)  # (I,L,K)
        Y = z[2 * n : 3 * n].reshape(I, L, K)  # (I,L,K)

        _, _, power_ilk, _, _, _ = wf_model(
            x=X,
            y=Y,
            wd=wds,
            ws=wss,
            TI=TI_lk,
            yaw=yaw,
            tilt=0,  # required in your setup
            return_simulationResult=False,
            n_cpu=1,
        )
        return power_ilk.sum()  # scalar: sum over (I,L,K)

    return autograd(power_inflowgrid, argnum=0, vector_interdependence=True)


@log_execution_time
def partials_fn_inflowgrid_batch(b, gamma_b, dP_dz, return_gamma=False):
    """Compute dP/dgamma and direct dP/d(x,y) for all graphs in one prepared batch.

    Parameters
    ----------
    b : dict
        One element from `prepared` (contains b["chunk"], b["offsets"], etc.).
    gamma_b : torch.Tensor
        Predicted gamma for all nodes in the mega-graph, shape (total_N,).
    dP_dz : callable
        From make_dP_dz_inflowgrid(wf_model).
    return_gamma : bool

    Returns
    -------
    v_flat : np.ndarray
        dP/dgamma flattened in the same node order as v_buf, shape (total_N,).
    direct_flat : np.ndarray
        Direct dP/d(x,y) flattened in the same node order as direct_buf, shape (total_N,2).
    gamma_list : list[np.ndarray] | None
        Per-graph gamma vectors, each shape (I,), if return_gamma=True.
    """
    chunks = b["chunk"]
    offsets = b["offsets"]
    M = len(chunks)
    I = int(chunks[0]["N"])

    # Unique inflow grid (L,K)
    wds_u = sorted({float(c["g"]["meta"]["wd_deg"]) for c in chunks})
    wss_u = sorted({float(c["g"]["meta"]["ws"]) for c in chunks})
    L, K = len(wds_u), len(wss_u)

    assert M == L * K, (
        "Prepared batch is not a full inflow grid. "
        f"M={M}, L={L}, K={K}, L*K={L * K}. "
        f"unique_wd={wds_u}, unique_ws={wss_u}"
    )
    # Require full grid
    if M != L * K:
        raise ValueError(f"Batch is not a full (wd,ws) grid: M={M}, L*K={L * K}")

    wd_to_l = {wd: i for i, wd in enumerate(wds_u)}
    ws_to_k = {ws: j for j, ws in enumerate(wss_u)}

    # Build (I,L,K) arrays
    X_ilk = np.empty((I, L, K), dtype=np.float32)
    Y_ilk = np.empty((I, L, K), dtype=np.float32)
    yaw_ilk = np.empty((I, L, K), dtype=np.float32)
    TI_lk = np.empty((L, K), dtype=np.float32)

    # Fill from graphs
    for idx, c in enumerate(chunks):
        off = int(offsets[idx])
        g = c["g"]

        wd = float(g["meta"]["wd_deg"])
        ws = float(g["meta"]["ws"])
        ti = float(g["meta"]["ti"])

        l = wd_to_l[wd]
        k = ws_to_k[ws]

        pos = np.asarray(g["pos"], dtype=np.float32)  # (I,2)
        X_ilk[:, l, k] = pos[:, 0]
        Y_ilk[:, l, k] = pos[:, 1]

        yaw_ilk[:, l, k] = (
            gamma_b[off : off + I].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        TI_lk[l, k] = np.float32(ti)

    wds = np.asarray(wds_u, dtype=np.float32)  # (L,)
    wss = np.asarray(wss_u, dtype=np.float32)  # (K,)

    # Packed gradient (one AD pass)
    z0 = np.concatenate([yaw_ilk.ravel(), X_ilk.ravel(), Y_ilk.ravel()]).astype(
        np.float32, copy=False
    )
    g0 = np.asarray(dP_dz(z0, wds, wss, TI_lk, I, L, K), dtype=np.float32)

    n = I * L * K
    dP_dyaw = g0[0:n].reshape(I, L, K)  # (I,L,K)
    dP_dX = g0[n : 2 * n].reshape(I, L, K)
    dP_dY = g0[2 * n : 3 * n].reshape(I, L, K)

    # Flatten back into mega-graph node order (per-graph contiguous blocks)
    v_flat = np.empty((b["total_N"],), dtype=np.float32)
    direct_flat = np.empty((b["total_N"], 2), dtype=np.float32)

    gamma_list = [] if return_gamma else None

    for idx, c in enumerate(chunks):
        off = int(offsets[idx])
        g = c["g"]
        wd = float(g["meta"]["wd_deg"])
        ws = float(g["meta"]["ws"])
        l = wd_to_l[wd]
        k = ws_to_k[ws]

        v_flat[off : off + I] = dP_dyaw[:, l, k]
        direct_flat[off : off + I, 0] = dP_dX[:, l, k]
        direct_flat[off : off + I, 1] = dP_dY[:, l, k]

        if return_gamma:
            gamma_list.append(yaw_ilk[:, l, k].copy())

    return v_flat, direct_flat, gamma_list


# new vjp
@log_execution_time
def gradP_torchscript_vjp_xy_inflowgrid_prepared(
    ts_path,
    prepared,
    dP_dz,  # <-- new: packed autograd grad fn from make_dP_dz_inflowgrid
    gamma_col=-1,
    uv_scale=1.0,
    return_gamma=False,
):
    """dP/d(x,y) using prepared batches + inflow-grid batched PyWake partials + edge-Δu/Δv VJP."""
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

        edge_attr_buf = b["edge_attr_buf"]
        grad_uv_buf = b["grad_uv_buf"]
        v_buf = b["v_buf"]
        direct_buf = b["direct_buf"]

        if col_v == col_u + 1:
            duv = edge_attr0[:, col_u : col_v + 1].detach()
        else:
            duv = edge_attr0[:, [col_u, col_v]].detach()
        duv = duv.requires_grad_(True)  # (E,2)

        edge_attr_buf.detach_()
        edge_attr_buf.copy_(edge_attr0)
        if col_v == col_u + 1:
            edge_attr_buf[:, col_u : col_v + 1] = duv
        else:
            edge_attr_buf[:, [col_u, col_v]] = duv

        yhat = model(edge_index, edge_attr_buf, globals_, batch)
        if yhat.dim() == 1:
            yhat = yhat[:, None]
        gamma_b = yhat[:, gamma_col]  # (total_N,)

        # ---- batched PyWake partials over (wd,ws) grid ----
        v_flat, direct_flat, gamma_list_b = partials_fn_inflowgrid_batch(
            b, gamma_b, dP_dz, return_gamma=return_gamma
        )
        v_buf[:] = torch.from_numpy(v_flat)
        direct_buf[:] = torch.from_numpy(direct_flat)
        if return_gamma:
            gamma_list.extend(gamma_list_b)
        # --------------------------------------------------

        ell = (v_buf * gamma_b).sum()

        if duv.numel() > 0 and ell.requires_grad:
            g_duv = torch.autograd.grad(
                ell, duv, retain_graph=False, create_graph=False
            )[0]  # (E,2)

            grad_uv_buf.zero_()
            g_scaled = g_duv / uv_scale
            grad_uv_buf.index_add_(0, dst, g_scaled)
            grad_uv_buf.index_add_(0, src, -g_scaled)
        else:
            grad_uv_buf.zero_()

        # Split, rotate, add direct term (unchanged)
        for k, c in enumerate(b["chunk"]):
            off = int(b["offsets"][k])
            N = int(c["N"])
            cth = float(c["c"])
            sth = float(c["s"])

            grad_uv = grad_uv_buf[off : off + N]  # (N,2)

            grad_xy = torch.empty_like(grad_uv)
            grad_xy[:, 0] = grad_uv[:, 0] * cth - grad_uv[:, 1] * sth
            grad_xy[:, 1] = grad_uv[:, 0] * sth + grad_uv[:, 1] * cth

            dP_dxy = (
                (direct_buf[off : off + N] + grad_xy)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            dP_dxy_list.append(dP_dxy)

    if return_gamma:
        return dP_dxy_list, gamma_list
    return dP_dxy_list


if __name__ == "__main__":
    import gc

    import xarray as xr
    from design_friendly.utils.to_graph import graph_maker_sequential, seq_graph_inputs

    ds_path = "/home/dgodi/ActiveProjects/gnn_framework/torque26/data/setpoints2/dsmrob_wdeficit.nc"
    ds = xr.open_dataset(ds_path).load()
    test_slice = slice(160000, 199999)
    test_slice = slice(160000, 160000 + 9999)

    test_seq_inputs = seq_graph_inputs(ds, test_slice)
    del ds
    gc.collect()
    test_graphs = graph_maker_sequential(
        **test_seq_inputs,
        num_threads=-1,
        connectivity="wake_aware",
    )

    trained_models_dir = "/home/dgodi/ActiveProjects/design_friendly_control/design-friendly-control/design_friendly/runs/GEN_4_layers_0.0_dropout_1e-4_lr_50_epochs_256_latent_dim_01_04_19_02/trained_models/"
    ts_path = trained_models_dir + "best.ptnox.torchscript.pt"
    dP_dxy_list = gradP_torchscript_vjp_xy(
        ts_path, test_graphs, dummy_partials_fn, batch_size=2048
    )
    prepared = prepare_gradP_vjp_xy(test_graphs, batch_size=2048, edge_uv_cols=(0, 1))
    dP_dxy_list_prep = gradP_torchscript_vjp_xy_prepared(
        ts_path, prepared, dummy_partials_fn, gamma_col=-1, uv_scale=1.0
    )
    report = compare_gradP_outputs_from_lists(
        dP_baseline_list=dP_dxy_list,
        dP_prepared_list=dP_dxy_list_prep,
        # gamma_baseline_list=gamma_list_baseline,  # optional
        # gamma_prepared_list=gamma_list_prepared,  # optional
        atol=1e-6,
        rtol=1e-5,
    )

    print(report["ok_dP_dxy"], report["ok_gamma"])
    print(report["summary"])