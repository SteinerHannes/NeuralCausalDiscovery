import numpy as np
import torch
from graphical_models import DAG
from data.InterventionDataset import InterventionDataset
from data.utils import build_split_indices

_DATA_CACHE = {}


class ElectricalCircuitData(InterventionDataset):
    """
    Synthetic electrical circuit dataset with 6 nodes in fixed order:
        0: V (voltage), 1: R (resistance), 2: G (gain),
        3: I (current), 4: S (sensor), 5: O (output).

    Data generation:
        V ~ Normal(v_mean, v_std)
        R ~ Uniform(r_min, r_max)
        G ~ Uniform(g_min, g_max)
        I = V / (R + 1e-8)
        S = I + Normal(0, sensor_noise_std)
        O = G * S + Normal(0, output_noise_std)

    Graph structure (edges):
        V -> I <- R, I -> S, S -> O <- G

    The dataset returns (x, y) where x are the first input_size columns and y are the
    next output_size columns of the generated data. Supported splits keep the total
    number of variables fixed at 6, with these common partitions:
        input_size=3, output_size=3 -> x=[V, R, G], y=[I, S, O]
        input_size=4, output_size=2 -> x=[V, R, G, I], y=[S, O]
        input_size=5, output_size=1 -> x=[V, R, G, I, S], y=[O]
    Splits are deterministic given (seed, fold, and config params) and cached in-memory
    via _DATA_CACHE.

    edge_weights provides a linear SEM proxy for interventions.py. It is computed by
    least-squares fitting each child to its parents on the generated data, which
    approximates the nonlinear/multiplicative relationships in this dataset.
    """
    def __init__(
        self,
        subset="train",
        seed=1,
        num_samples=10000,
        input_size=3,
        output_size=3,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        extract_samples=1000,
        v_mean=0.0,
        v_std=1.0,
        r_min=1.0,
        r_max=10.0,
        g_min=0.5,
        g_max=2.0,
        sensor_noise_std=0.1,
        output_noise_std=0.0,
        fold=0,
    ):
        """
        Args:
            subset: One of {"train","val","test","extract","all"}.
            seed: Random seed for sampling (int).
            num_samples: Samples allocated to the train/val/test pool (>0).
            input_size: Number of input variables; supported {3,4,5}.
            output_size: Number of output variables; supported {3,2,1}.
            train_ratio: Train split fraction in [0,1].
            val_ratio: Validation split fraction in [0,1].
            test_ratio: Test split fraction in [0,1].
            extract_samples: Extraction samples added on top of num_samples.
            v_mean: Mean of voltage Normal distribution (any real).
            v_std: Std dev of voltage Normal distribution (>=0).
            r_min: Min resistance for Uniform (>=0, < r_max).
            r_max: Max resistance for Uniform (> r_min).
            g_min: Min gain for Uniform (>=0, < g_max).
            g_max: Max gain for Uniform (> g_min).
            sensor_noise_std: Std dev of sensor noise (>=0).
            output_noise_std: Std dev of output noise (>=0).
            fold: Fold index (int) for deterministic splits.
        """
        super().__init__()
        if subset not in ["train", "val", "test", "extract", "all"]:
            raise ValueError(f"subset must be 'train', 'val', 'test', 'extract' or 'all', got {subset}")

        self.subset = subset
        if input_size + output_size != 6:
            raise ValueError("ElectricalCircuitData requires input_size + output_size == 6.")
        if input_size not in (3, 4, 5):
            raise ValueError("ElectricalCircuitData supports input_size in {3, 4, 5}.")
        if output_size not in (3, 2, 1):
            raise ValueError("ElectricalCircuitData supports output_size in {3, 2, 1}.")
        self.input_size = input_size
        self.output_size = output_size
        self.v_mean = v_mean
        self.v_std = v_std
        self.r_min = r_min
        self.r_max = r_max
        self.g_min = g_min
        self.g_max = g_max
        self.sensor_noise_std = sensor_noise_std
        self.output_noise_std = output_noise_std
        self.num_samples = int(num_samples)
        self.extract_samples = int(extract_samples)

        cache_key = (
            seed,
            num_samples,
            input_size,
            output_size,
            v_mean,
            v_std,
            r_min,
            r_max,
            g_min,
            g_max,
            sensor_noise_std,
            output_noise_std,
            float(train_ratio),
            float(val_ratio),
            float(test_ratio),
            int(extract_samples),
            fold,
        )

        if cache_key not in _DATA_CACHE:
            rng = np.random.default_rng(seed + fold)
            total_num_samples = int(num_samples) + int(extract_samples)

            v = rng.normal(v_mean, v_std, size=total_num_samples)
            r = rng.uniform(r_min, r_max, size=total_num_samples)
            g = rng.uniform(g_min, g_max, size=total_num_samples)

            i = v / (r + 1e-8)
            s = i + rng.normal(0.0, sensor_noise_std, size=total_num_samples)
            o = g * s + rng.normal(0.0, output_noise_std, size=total_num_samples)

            dag_data = np.stack([v, r, g, i, s, o], axis=1)

            nodes = set(range(6))
            dag = DAG(nodes)
            dag.add_edges({0}, 3)  # V -> I
            dag.add_edges({1}, 3)  # R -> I
            dag.add_edges({3}, 4)  # I -> S
            dag.add_edges({4}, 5)  # S -> O
            dag.add_edges({2}, 5)  # G -> O

            def _fit_linear_weights(parent_idx, child_idx):
                X = dag_data[:, parent_idx]
                if X.ndim == 1:
                    X = X[:, None]
                y = dag_data[:, child_idx]
                weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                return weights

            weights_vr = _fit_linear_weights([0, 1], 3)
            weight_i = _fit_linear_weights([3], 4)
            weights_sg = _fit_linear_weights([4, 2], 5)

            edge_weights = {
                (0, 3): float(weights_vr[0]),
                (1, 3): float(weights_vr[1]),
                (3, 4): float(weight_i[0]),
                (4, 5): float(weights_sg[0]),
                (2, 5): float(weights_sg[1]),
            }

            _DATA_CACHE[cache_key] = {
                "dag": dag,
                "dag_data": dag_data,
                "edge_weights": edge_weights,
                "splits": build_split_indices(
                    num_samples=num_samples,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    extract_samples=extract_samples,
                ),
            }

        entry = _DATA_CACHE[cache_key]
        self.dag = entry["dag"]
        self.dag_data = entry["dag_data"]
        self.edge_weights = entry.get("edge_weights")
        if subset == "all":
            self.indices = np.arange(len(self.dag_data))
        else:
            self.indices = entry["splits"][subset]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = self.dag_data[real_idx]
        x = data[:self.input_size]
        y = data[self.input_size:self.input_size + self.output_size]
        return x, y

    def simulate_intervention(self, num_samples, cfg, intervention: dict, seed: int):
        rng = np.random.default_rng(seed)
        noise_scale = cfg.intervention.noise_scale

        def _prepare(values):
            if np.isscalar(values):
                return np.full(num_samples, values, dtype=float)
            arr = np.asarray(values, dtype=float)
            if arr.shape[0] != num_samples:
                raise ValueError("Intervention value array must match num_samples")
            return arr

        intervention = intervention or {}

        if 0 in intervention:
            v = _prepare(intervention[0])
        else:
            v = rng.normal(self.v_mean, self.v_std, size=num_samples)

        if 1 in intervention:
            r = _prepare(intervention[1])
        else:
            r = rng.uniform(self.r_min, self.r_max, size=num_samples)

        if 2 in intervention:
            g = _prepare(intervention[2])
        else:
            g = rng.uniform(self.g_min, self.g_max, size=num_samples)

        if 3 in intervention:
            i = _prepare(intervention[3])
        else:
            i = v / (r + 1e-8)

        if 4 in intervention:
            s = _prepare(intervention[4])
        else:
            s_noise = rng.normal(0.0, self.sensor_noise_std * noise_scale, size=num_samples)
            s = i + s_noise

        if 5 in intervention:
            o = _prepare(intervention[5])
        else:
            o_noise = rng.normal(0.0, self.output_noise_std * noise_scale, size=num_samples)
            o = g * s + o_noise

        data = np.stack([v, r, g, i, s, o], axis=1)
        return data
