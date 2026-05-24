import unittest
import numpy as np

from data.ElectricalCircuitData import ElectricalCircuitData


class ElectricalCircuitDataTest(unittest.TestCase):
    def test_edge_weights_and_split_sizes(self):
        num_samples = 100
        train = ElectricalCircuitData(
            subset="train",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=10,
        )
        val = ElectricalCircuitData(
            subset="val",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=10,
        )
        test = ElectricalCircuitData(
            subset="test",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=10,
        )
        extract = ElectricalCircuitData(
            subset="extract",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=10,
        )
        all_ds = ElectricalCircuitData(
            subset="all",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=10,
        )

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)
        self.assertEqual(len(extract), 10)
        self.assertEqual(len(all_ds), num_samples + 10)

        edge_weights = train.edge_weights
        expected_keys = {(0, 3), (1, 3), (3, 4), (4, 5), (2, 5)}
        self.assertIsInstance(edge_weights, dict)
        self.assertEqual(set(edge_weights.keys()), expected_keys)
        for key in expected_keys:
            weight = edge_weights[key]
            self.assertTrue(isinstance(weight, float))
            self.assertTrue(np.isfinite(weight))

    def test_extract_samples_adds_fixed_holdout(self):
        num_samples = 5000
        train = ElectricalCircuitData(
            subset="train",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=1000,
        )
        val = ElectricalCircuitData(
            subset="val",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=1000,
        )
        test = ElectricalCircuitData(
            subset="test",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=1000,
        )
        extract = ElectricalCircuitData(
            subset="extract",
            num_samples=num_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            extract_samples=1000,
        )

        self.assertEqual(len(train), 4000)
        self.assertEqual(len(val), 500)
        self.assertEqual(len(test), 500)
        self.assertEqual(len(extract), 1000)


if __name__ == "__main__":
    unittest.main()
