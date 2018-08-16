import time
from contextlib import contextmanager


@contextmanager
def timer(title):
    print(f"== {title}")
    t0 = time.time()
    yield
    print("== done. {:.0f} [s]".format(time.time() - t0))
