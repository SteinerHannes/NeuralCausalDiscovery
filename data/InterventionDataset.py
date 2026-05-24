import torch


class InterventionDataset(torch.utils.data.Dataset):
    """
    Dataset interface for intervention simulations.

    Implement simulate_intervention to generate interventional samples using the
    dataset's true data-generating process (including nonlinear equations if any).
    The method must return a NumPy array with shape (num_samples, num_nodes).
    """

    def simulate_intervention(self, num_samples, cfg, intervention: dict, seed: int):
        """
        Args:
            num_samples: Number of samples to generate (>0).
            cfg: Hydra config used for experiment-level settings (e.g., noise_scale).
            intervention: Dict {node_index: value} for do-interventions. Values can be
                scalars or arrays of length num_samples.
            seed: Random seed for reproducibility.
        Returns:
            NumPy array of interventional samples with shape (num_samples, num_nodes).
        """
        raise NotImplementedError("Dataset must implement simulate_intervention().")
