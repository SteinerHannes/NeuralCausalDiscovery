import unittest
import numpy as np
from data.LinearSEMDataset import LinearSEMDataset

class MyTestCase(unittest.TestCase):
    def test_dag_data(self):
        data1 = LinearSEMDataset(subset="train")
        data2 = LinearSEMDataset(subset="test")
        data3 = LinearSEMDataset(subset="val")
        data4 = LinearSEMDataset(subset="train")
        data5 = LinearSEMDataset(subset="train", seed=2)
        data6 = LinearSEMDataset(subset="extract", seed=2)
        data7 = LinearSEMDataset(subset="extract", seed=2)
        data8 = LinearSEMDataset(subset="extract", seed=2, fold=1)
        data9 = LinearSEMDataset(subset="train", num_samples=10)
        data10 = LinearSEMDataset(subset="all")

        self.assertTrue(np.array_equal(data1[0], data4[0]))
        self.assertFalse(np.array_equal(data1[0], data2[0]))
        self.assertFalse(np.array_equal(data1[0], data3[0]))
        self.assertFalse(np.array_equal(data1[0], data5[0]))
        self.assertFalse(np.array_equal(data5[0], data6[0]))
        self.assertTrue(np.array_equal(data6[0], data7[0]))
        self.assertFalse(np.array_equal(data7[0], data8[0]))
        self.assertTrue(np.array_equal(data1[0], data10[0]))
        self.assertTrue(np.array_equal(data1[1], data10[1]))
        self.assertFalse(np.array_equal(data5[0], data10[0]))

        self.assertTrue(len(data1) == len(data5))
        self.assertTrue(len(data9) < len(data1))

    def test_edge_weights_are_fixed_across_folds_and_sample_sizes(self):
        fold0 = LinearSEMDataset(subset="all", seed=2, fold=0)
        fold1 = LinearSEMDataset(subset="all", seed=2, fold=1)
        fewer_samples = LinearSEMDataset(subset="all", seed=2, fold=0, num_samples=10)

        self.assertEqual(fold0.edge_weights, fold1.edge_weights)
        self.assertEqual(fold0.edge_weights, fewer_samples.edge_weights)
        self.assertFalse(np.array_equal(fold0.dag_data, fold1.dag_data))

if __name__ == '__main__':
    unittest.main()
