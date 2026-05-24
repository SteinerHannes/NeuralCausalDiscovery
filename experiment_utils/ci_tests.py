import numpy as np


class CondIndepKCI:
    """
    Kernel-based conditional independence test wrapper for causal-learn's CIT API.

    Returns True when variables are conditionally independent (p-value > threshold).
    Designed for continuous, potentially non-linear SEM data.
    """
    def __init__(
        self,
        dataset,
        threshold,
        method="kci",
        standardize=True,
        count_tests=False,
        use_cache=False,
        verbose=False,
        **kwargs,
    ):
        if dataset is None:
            raise ValueError("CondIndepKCI requires a dataset.")

        data = np.asarray(dataset, dtype=float)
        if data.ndim != 2:
            raise ValueError("CondIndepKCI expects a 2D array (n_samples, n_features).")

        if standardize:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            std[std == 0] = 1.0
            data = (data - mean) / std

        self.data = data
        self.num_records, self.num_vars = data.shape
        self.threshold = float(threshold)
        self.verbose = verbose

        self.count_tests = count_tests
        self.test_counter = [0 for _ in range(self.num_vars - 1)] if count_tests else None

        self.is_cache = use_cache
        self._cache = {} if use_cache else None

        # Lazy import to avoid heavy dependency unless needed.
        try:
            from causallearn.utils.cit import CIT  # pylint: disable=import-error
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("CondIndepKCI requires causal-learn (causallearn).") from exc

        self._cit = CIT(self.data, method=method, **kwargs)
        self.method = method

    def cond_indep(self, x, y, zz):
        zz = tuple(int(z) for z in zz) if zz else tuple()
        xi, yi = int(x), int(y)
        if self.is_cache:
            cache_key = (min(xi, yi), max(xi, yi), tuple(sorted(zz)))
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            cache_key = None

        p_val = self._cit(xi, yi, condition_set=list(zz))
        if not np.isfinite(p_val):
            res = False
        else:
            res = p_val > self.threshold

        if self.verbose:
            print(f"CI({xi}, {yi} | {zz}) p={p_val:.4g} -> indep={res}")

        if self.is_cache:
            self._cache[cache_key] = res
        if self.count_tests:
            self.test_counter[len(zz)] += 1

        return res
