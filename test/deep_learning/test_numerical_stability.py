from unittest import TestCase


class NumericalStabilityTest(TestCase):
    def test_numerical_stability(self):
        number = 1000000000

        for i in range(1000000):
            number += 1e-6

        number -= 1000000000

        self.assertEqual(0.95367431640625, number)
