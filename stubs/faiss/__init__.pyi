# Minimal stubs for faiss-cpu covering the API surface used by graph_env.py.
#
# The SWIG-generated sources expose C++ signatures (add(n, x), search(n, x, k,
# distances, labels)), but class_wrappers.py monkey-patches every Index subclass
# at import time with Python-friendly replacements:
#   add(x)            — x is a float32 ndarray of shape (n, d)
#   search(x, k)      — returns (D, I), both float32/int64 ndarrays of shape (n, k)
# These stubs reflect that actual runtime API.

import numpy as np
from numpy.typing import NDArray

class Index:
    ntotal: int
    d: int
    def add(self, x: NDArray[np.float32]) -> None: ...
    def search(
        self, x: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...

class IndexFlat(Index):
    def __init__(self, d: int) -> None: ...

class IndexFlatIP(IndexFlat):
    def __init__(self, d: int) -> None: ...

class IndexFlatL2(IndexFlat):
    def __init__(self, d: int) -> None: ...
