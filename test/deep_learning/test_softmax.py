from unittest import TestCase

import numpy as np


def softmax(input_array):
    exp_array = np.exp(input_array)
    exp_array_sum = np.sum(exp_array, axis=0)

    return exp_array / exp_array_sum


class SoftmaxTest(TestCase):
    def test_softmax1(self):
        result = softmax([2.0, 1.0, 0.1])

        np.testing.assert_almost_equal([0.66, 0.24, 0.1], result, decimal=2)
        self.assertEqual(1.0, sum(result))

    def test_softmax2(self):
        result = softmax([0.2, 0.1, 0.01])

        np.testing.assert_almost_equal([0.37, 0.33, 0.3], result, decimal=2)
        self.assertEqual(1.0, sum(result))
