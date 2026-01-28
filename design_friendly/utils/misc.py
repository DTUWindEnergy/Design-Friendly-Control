# Author: Deniz Gokhan Dirik
# Contact: dgodi@dtu.dk
# Date: 2/10/2025
# Description:
import functools
import logging
import sys
import time as tt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
    # force=True,
)


def log_execution_time_simple(func=None, *, label=None):
    """
    Decorator to log the execution time of a function via logging.
    """

    def _decorator(f):
        logger = logging.getLogger(f.__module__)
        name = label if label is not None else f.__name__

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            t0 = tt.perf_counter()
            try:
                return f(*args, **kwargs)
            finally:
                dt = tt.perf_counter() - t0
                logger.info("%s took %.3f s", name, dt)

        return wrapper

    if func is None:
        return _decorator
    return _decorator(func)


def log_execution_time(
    func=None, *, label=None, stream=None, enable=None, every=1, reset_on_new_cell=True
):
    """
    Decorator to log execution time.

    - Terminal (TTY): in-place '\\r' updates.
    - Jupyter: updates a single output area via display_id; resets per executed cell if requested.
    - Non-interactive: falls back to logging.info.
    """

    def _in_notebook():
        try:
            from IPython import get_ipython

            ip = get_ipython()
            return (ip is not None) and ("ipykernel" in sys.modules)
        except Exception:
            return False

    def _cell_exec_count():
        try:
            from IPython import get_ipython

            ip = get_ipython()
            return getattr(ip, "execution_count", None)
        except Exception:
            return None

    stream = sys.stderr if stream is None else stream

    def _decorator(f):
        in_nb = _in_notebook()
        logger = logging.getLogger(f.__module__)
        name = label if label is not None else f.__name__

        call_count = 0
        max_len = 0
        nb_handle = None
        nb_cell_id = None
        t_first = None

        def _report(dt):
            nonlocal max_len, nb_handle, nb_cell_id
            # Throttle updates
            if every > 1 and (call_count % every) != 0:
                return
            total = tt.perf_counter() - t_first
            msg = f"{name} | n={call_count} | {dt:.3f} s | total={total:.0f} s"
            try:
                if in_nb:
                    if reset_on_new_cell:
                        cur = _cell_exec_count()
                        if cur is not None and cur != nb_cell_id:
                            nb_cell_id = cur
                            nb_handle = None

                    from IPython.display import display

                    if nb_handle is None:
                        nb_handle = display(msg, display_id=True)
                    else:
                        nb_handle.update(msg)
                    return

                do_inline = (
                    enable
                    if enable is not None
                    else getattr(stream, "isatty", lambda: False)()
                )
                if do_inline:
                    max_len = max(max_len, len(msg))
                    stream.write("\r" + msg.ljust(max_len))
                    stream.flush()
                else:
                    logger.info("%s took %.3f s", name, dt)
            except Exception:
                # Don't let leg break the decorated function
                pass

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal t_first, call_count
            if t_first is None:
                t_first = tt.perf_counter()
            t0 = tt.perf_counter()
            try:
                out = f(*args, **kwargs)
            except Exception:
                dt = tt.perf_counter() - t0
                call_count += 1
                _report(dt)
                raise
            else:
                dt = tt.perf_counter() - t0
                call_count += 1
                _report(dt)
                return out

        return wrapper

    if func is None:
        return _decorator
    return _decorator(func)


def compare_preds(*preds, ref=0):
    """Flatten preds, drop non-finite, assert equal lengths vs ref (and not both empty)."""

    def flat_finite(x):
        x = np.asarray(x, dtype=object)
        if x.dtype == object:
            x = (
                np.concatenate([np.ravel(np.asarray(a, float)) for a in x.ravel()])
                if x.size
                else np.array([], float)
            )
        else:
            x = np.ravel(np.asarray(x, float))
        return x[np.isfinite(x)]

    F = [flat_finite(p) for p in preds]
    A = F[ref]
    out = {}
    for i, B in enumerate(F):
        if i == ref:
            continue
        assert (A.size == B.size) and (A.size > 0), (
            f"Size mismatch/empty after finite-filter: ref={A.size}, pred{i}={B.size}"
        )
        d = B - A
        r = dict(
            n=A.size,
            mean=float(np.mean(d)),
            mae=float(np.mean(np.abs(d))),
            rmse=float(np.sqrt(np.mean(d**2))),
            max_abs=float(np.max(np.abs(d))),
        )
        out[i] = r
        print(
            f"ref{ref} vs pred{i} | n {r['n']} | mean {r['mean']:.3e} | mae {r['mae']:.3e} | rmse {r['rmse']:.3e} | max {r['max_abs']:.3e}"
        )
    return out


def main():
    pass


if __name__ == "__main__":
    main()
