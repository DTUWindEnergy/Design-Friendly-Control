import torch
from .misc import log_execution_time
from py_wake import numpy as np


@log_execution_time
def predict_torchscript(ts_path, graphfarms, batch_size=2048, reshape="list"):
    """Predict with an exported TorchScript WindFarmGNN (NoX signature).

    Parameters
    ----------
    ts_path : str
        Path to TorchScript artifact exported with forward(edge_index, edge_attr, globals, batch).
    graphfarms : sequence
        Sequence/dataset of graph objects with attributes:
          - edge_index: array/tensor, shape (2, E)
          - edge_attr:  array/tensor, shape (E, Fe)
          - globals:    array/tensor, shape (Fg,) or (1, Fg)
          - num_nodes (optional): int
    batch_size : int
        Number of graphs per forward pass.
    reshape : {None, "list", "array"}
        None : return list[np.ndarray] per batch with shape (sum_N, Fy) (squeezed if Fy==1)
        "list" : return list[np.ndarray] length n_cases, each (N_i, Fy)
        "array" : return np.ndarray (n_cases, max_N, Fy) max_N padded with NaN
        np.array : support and lut

    Returns
    -------
    y_pred : list or np.ndarray
    """
    if reshape not in (None, "list", "array"):
        raise ValueError("reshape must be None, 'list', or 'array'")

    dev = torch.device("cpu")
    ts = torch.jit.load(ts_path, map_location=dev).eval()

    def _num_nodes(g, edge_index_t):
        n = getattr(g, "num_nodes", None)
        if n is not None:
            return int(n)
        return int(edge_index_t.max().item()) + 1

    def _batch_graphs(graphs):
        edge_indices = []
        edge_attrs = []
        globals_list = []
        batch_vecs = []
        ptr = [0]  # cumulative node offsets within the batch output
        node_offset = 0
        for i, g in enumerate(graphs):
            ei = torch.as_tensor(g.edge_index, dtype=torch.int64)
            ea = torch.as_tensor(g.edge_attr, dtype=torch.float32)
            gl = torch.as_tensor(getattr(g, "globals"), dtype=torch.float32)

            if gl.dim() == 2 and gl.size(0) == 1:
                gl = gl.squeeze(0)  # (Fg,)
            elif gl.dim() != 1:
                raise ValueError(
                    f"globals must be (Fg,) or (1,Fg); got {tuple(gl.shape)}"
                )
            N = _num_nodes(g, ei)
            edge_indices.append(ei + node_offset)  # (2, E_i)
            edge_attrs.append(ea)  # (E_i, Fe)
            globals_list.append(gl)  # (Fg,)
            batch_vecs.append(torch.full((N,), i, dtype=torch.int64))  # (N,)
            node_offset += N
            ptr.append(node_offset)
        edge_index = torch.cat(edge_indices, dim=1)  # (2, sum_E)
        edge_attr = torch.cat(edge_attrs, dim=0)  # (sum_E, Fe)
        globals_ = torch.stack(globals_list, dim=0)  # (G, Fg)
        batch = torch.cat(batch_vecs, dim=0)  # (sum_N,)
        ptr = np.asarray(ptr, dtype=np.int64)  # (G+1,)
        return (
            edge_index.to(device=dev, dtype=torch.int64),
            edge_attr.to(device=dev, dtype=torch.float32),
            globals_.to(device=dev, dtype=torch.float32),
            batch.to(device=dev, dtype=torch.int64),
            ptr,
        )

    if reshape is None:
        y_batches = []
    else:
        y_cases = []
        max_N = 0
    with torch.inference_mode():
        n_total = len(graphfarms)
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            # avoid slicing datasets that implement __getitem__ poorly for slices
            chunk = [graphfarms[i] for i in range(s, e)]
            edge_index, edge_attr, globals_, batch_vec, ptr = _batch_graphs(chunk)
            y = ts(edge_index, edge_attr, globals_, batch_vec)  # (sum_N, Fy) typically
            y_np = y.detach().cpu().numpy()
            if reshape is None:
                y_batches.append(y_np.squeeze())
                continue
            # Ensure 2D per-case arrays: (N_i, Fy)
            if y_np.ndim == 1:
                y_np = y_np[:, None]
            Fy = y_np.shape[1]
            # Split by ptr into per-graph node blocks
            for i in range(len(chunk)):
                a = y_np[ptr[i] : ptr[i + 1], :]  # (N_i, Fy)
                y_cases.append(a)
                if a.shape[0] > max_N:
                    max_N = a.shape[0]
    if reshape is None:
        return y_batches
    if reshape == "list":
        return y_cases
    # if reshape == "array" (pad to max_N) and reshape to 3D
    n_cases = len(y_cases)
    Fy = y_cases[0].shape[1] if n_cases > 0 else 0
    out = np.full((n_cases, max_N, Fy), np.nan, dtype=np.float32)
    for i, a in enumerate(y_cases):
        out[i, : a.shape[0], :] = a.astype(np.float32, copy=False)
    return out


@log_execution_time
def torchscript_to_lut(y_cases_or_array, wds, wss):
    """
    Convert TorchScript predictions to LUT layout.

    Parameters
    ----------
    y_cases_or_array : list[np.ndarray] or np.ndarray
        If list: length n_cases, each array is (N, Fy) or (N,).
        If array: shape (n_cases, max_N, Fy) padded with NaN (or (n_cases, max_N) for Fy=1).
    wds : array-like
        Wind directions (outer loop / slow index).
    wss : array-like
        Wind speeds (inner loop / fast index).

    Returns
    -------
    lut : np.ndarray
        Shape (N, n_wd, n_ws) if Fy==1 else (N, n_wd, n_ws, Fy).

    Notes
    -----
    Assumes case index = l*n_ws + k, i.e.:
      (wd=wds[0], ws=wss[0..]) then (wd=wds[1], ws=wss[0..]) ...
    """
    n_wd, n_ws = len(wds), len(wss)
    n_cases_expected = n_wd * n_ws
    if isinstance(y_cases_or_array, np.ndarray):
        Y = y_cases_or_array
        if Y.ndim == 2:  # (n_cases, max_N) -> (n_cases, max_N, 1)
            Y = Y[..., None]
        if Y.shape[0] != n_cases_expected:
            raise ValueError(
                f"n_cases={Y.shape[0]} != {n_cases_expected} (=len(wds)*len(wss))"
            )

        # infer constant N by non-NaN rows (padding assumed all-NaN across Fy)
        mask = ~np.isnan(Y).all(axis=-1)  # (n_cases, max_N)
        Ns = mask.sum(axis=1)
        if not np.all(Ns == Ns[0]):
            raise ValueError(
                "Varying N across cases after removing NaN padding; cannot form LUT (would compare partial preds)."
            )
        N = int(Ns[0])
        Y = Y[:, :N, :]  # (n_cases, N, Fy)
    else:
        ys = y_cases_or_array
        if len(ys) != n_cases_expected:
            raise ValueError(
                f"n_cases={len(ys)} != {n_cases_expected} (=len(wds)*len(wss))"
            )
        # ensure (N, Fy) and constant N
        a0 = ys[0]
        a0 = a0[:, None] if a0.ndim == 1 else a0
        N0, Fy = a0.shape
        for a in ys:
            a = a[:, None] if a.ndim == 1 else a
            if a.shape[0] != N0:
                raise ValueError("Varying N across cases; cannot form LUT.")
        Y = np.stack(
            [(a[:, None] if a.ndim == 1 else a) for a in ys], axis=0
        )  # (n_cases, N, Fy)
    # case axis -> (wd, ws, N, Fy) -> (N, wd, ws, Fy)
    Y = Y.reshape(n_wd, n_ws, Y.shape[1], Y.shape[2]).transpose(2, 0, 1, 3)
    return Y[..., 0] if Y.shape[-1] == 1 else Y


def main():
    pass


if __name__ == "__main__":
    main()