import unittest
import warnings

import numpy as np

from experiment_utils.correlation_sanitization import sanitize_dataset_for_corr


class CorrelationSanitizationTest(unittest.TestCase):
    def test_subset_specific_constant_columns_are_stabilized(self):
        dataset = np.array(
            [
                [0.0, 5.0, 10.0],
                [0.0, 5.0, 11.0],
                [0.0, 5.0, 12.0],
                [0.0, 5.0, 13.0],
                [1.0, 5.0, 14.0],
            ]
        )

        full_dataset = sanitize_dataset_for_corr(dataset, seed=7)
        subset = full_dataset[:4]

        self.assertEqual(int((subset.std(axis=0) == 0).sum()), 1)

        with warnings.catch_warnings(record=True) as raw_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            raw_corr = np.corrcoef(subset, rowvar=False)

        self.assertFalse(np.isfinite(raw_corr).all())
        self.assertTrue(raw_warnings)

        stabilized_subset = sanitize_dataset_for_corr(subset, seed=17)

        self.assertEqual(int((stabilized_subset.std(axis=0) == 0).sum()), 0)

        with warnings.catch_warnings(record=True) as stabilized_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            stabilized_corr = np.corrcoef(stabilized_subset, rowvar=False)

        self.assertTrue(np.isfinite(stabilized_corr).all())
        self.assertFalse(stabilized_warnings)


if __name__ == "__main__":
    unittest.main()
