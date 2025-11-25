# Author: Deniz Gokhan Dirik
# Contact: dgodi@dtu.dk
# Date: 2/10/2025
# Description:
import time as tt
import functools
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
    # force=True,
)


def log_execution_time(func=None, *, label=None):
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


def main():
    pass


if __name__ == "__main__":
    main()
