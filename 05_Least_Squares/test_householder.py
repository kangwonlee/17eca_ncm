import unittest

import numpy as np
import numpy.linalg as na

import householder_reflection as hh


class TestHouseHolder(unittest.TestCase):
    def test_householder_xy(self):
        x = np.matrix([[3, 2]]).T
        y = np.matrix([[7, -4]]).T

        k = 0

        hx, hy, u = hh.householder_xy(x, y, k)
        self.assertAlmostEqual(na.norm(x), na.norm(hx))
        self.assertAlmostEqual(na.norm(y), na.norm(hy))

        # x - hx  // u
        x_minus_hx = x - hx
        dot_product = x_minus_hx.T * u
        self.assertAlmostEqual(na.norm(x_minus_hx) * na.norm(u), dot_product[0, 0])

        # y - hy  // u
        y_minus_hy = y - hy
        dot_product = y_minus_hy.T * u
        self.assertAlmostEqual(na.norm(y_minus_hy) * na.norm(u), dot_product[0, 0])


if __name__ == '__main__':
    unittest.main()
