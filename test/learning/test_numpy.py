from unittest import TestCase

import numpy as np


class TestNumpy(TestCase):
    def test_take_with_ndarray(self):
        a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [0, 1]]])
        b = a.take([1, 2], axis=0)

        self.assertEqual((3, 2, 2), a.shape)
        self.assertEqual([[[4, 5], [6, 7]], [[8, 9], [0, 1]]], b.tolist())
