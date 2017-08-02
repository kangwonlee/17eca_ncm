import unittest

import numpy as np
import numpy.linalg as na

import householder_reflection as hh


class TestHouseHolder(unittest.TestCase):
    def test_householder_k(self):
        x = np.matrix([[3, 2]]).T
        k = 0

        hx, rho, u = hh.householder_k(x, k)
        # hx supposed to have same size as x
        self.assertAlmostEqual(na.norm(x), na.norm(hx))

        # x - hx  // u
        x_minus_hx = x - hx
        dot_product = x_minus_hx.T * u
        self.assertAlmostEqual(na.norm(x_minus_hx) * na.norm(u), dot_product[0, 0])

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


class TestQR(unittest.TestCase):
    def test_qrsteps(self):
        # data from C. Moler, Numerical computation with MATLAB, SIAM, 2008.
        s = (np.matrix([np.arange(1950, 2000 + 1, 10)]).T - 1950.0) / 50
        y = np.matrix([np.array([150.6970, 179.3230, 203.2120, 226.5050, 249.6330, 281.4220])]).T

        # design matrix
        mat_x = np.column_stack([np.power(s, 2), s, np.ones_like(s)])

        # QR factorization result
        # ** function under test **
        mat_qr_r, mat_qr_z, mat_residual_qr = hh.qrsteps(mat_x, y, False)

        # solve for beta using result from QR
        mat_beta_qr = na.solve(mat_qr_r, mat_qr_z)

        # calculate residue
        mat_y_x_beta = y - mat_x * mat_beta_qr

        # least square result
        mat_beta_ls, residual_ls, rank_ls, s_ls = na.lstsq(mat_x, y)

        # compare with least square result
        ## beta
        self.assertAlmostEqual(0.0, na.norm(mat_beta_qr - mat_beta_ls))
        ## residue
        ls_residue = np.sqrt(residual_ls)[0, 0]
        self.assertAlmostEqual(ls_residue, na.norm(mat_y_x_beta))
        self.assertAlmostEqual(ls_residue, na.norm(mat_residual_qr))


if __name__ == '__main__':
    unittest.main()
