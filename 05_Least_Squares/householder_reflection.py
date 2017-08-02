import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as na


def main_householder():
    x = np.matrix([[3, 2]]).T
    y = np.matrix([[7, -4]]).T

    k = 0

    Hx, Hy, u = householder_xy(x, y, k)

    # prepare data to draw figure
    x_arrow = [[0, 0], x.T.tolist()[0]]
    y_arrow = [[0, 0], y.T.tolist()[0]]
    u_arrow = [[0, 0], u.T.tolist()[0]]
    u_ortho_arrow = np.array([[-u[1, 0], u[0, 0]],
                              [u[1, 0], -u[0, 0]]]) * 1.6
    Hx_arrow = [[0, 0], Hx.T.tolist()[0]]
    Hy_arrow = [[0, 0], Hy.T.tolist()[0]]

    # prepare axis
    ax = plt.gca()

    # clear axis
    ax.cla()

    # draw arrows
    ax.arrow(x_arrow[0][0], x_arrow[0][1], x_arrow[1][0], x_arrow[1][1], head_width=0.5, head_length=0.5)
    ax.arrow(y_arrow[0][0], y_arrow[0][1], y_arrow[1][0], y_arrow[1][1], head_width=0.5, head_length=0.5)
    ax.arrow(u_arrow[0][0], u_arrow[0][1], u_arrow[1][0], u_arrow[1][1], head_width=0.5, head_length=0.5)
    ax.arrow(Hx_arrow[0][0], Hx_arrow[0][1], Hx_arrow[1][0], Hx_arrow[1][1], head_width=0.5, head_length=0.5)
    ax.arrow(Hy_arrow[0][0], Hy_arrow[0][1], Hy_arrow[1][0], Hy_arrow[1][1], head_width=0.5, head_length=0.5)

    # draw texts
    offset = 0.5
    ax.text(x_arrow[1][0] + offset, x_arrow[1][1], 'x')
    ax.text(y_arrow[1][0] + offset, y_arrow[1][1], 'y')
    ax.text(u_arrow[1][0] + offset, u_arrow[1][1], 'u')

    ax.text(Hx_arrow[1][0] - offset * 3, Hx_arrow[1][1], 'Hx')
    ax.text(Hy_arrow[1][0] - offset * 3, Hy_arrow[1][1], 'Hy')

    ax.text(u_ortho_arrow[0, 0] * 0.5, u_ortho_arrow[0, 1] * 0.5, '$u_\perp$')

    # draw line orthogonal to u
    plt.plot(u_ortho_arrow[:, 0], u_ortho_arrow[:, 1])

    # adjust aspect ratio
    plt.axis('equal')

    # set axis limits
    plt.xlim(-10, 10)
    plt.ylim(-8, 8)

    # draw grid
    plt.grid(True)

    # present plot
    plt.show()


def householder_xy(x, y, k):
    """
    Householder reflection such that Hx points to kth axis
    
    :param numpy.matrix x: 
    :param numpy.matrix y: 
    :param int k: 
    :return: 
    """
    Hx, rho, u = householder_k(x, k)

    tau_y = (rho * u.T * y)
    Hy = y + u * (-tau_y)

    return Hx, Hy, u


def householder_k(x, k):
    """
    Householder reflection such that Hx points to kth axis
    
    :param numpy.matrix x: 
    :param int k: 
    :return: 
    """
    # find u bisecting x and x axis
    sigma = na.norm(x)
    u = x + 0.0

    if u[k, 0]:
        sigma *= np.sign(u[k, 0])

    u = x + 0.0
    u[k, 0] += sigma

    # Householder reflection
    # rho_scala = (2 / (||u||^2)) = 1 / (sigma_scala * u[k])
    rho = 1.0 / (sigma * u[k, 0])

    # if x.shape == [m, n]: tau[1, n] = rho * u.T[1, m] * x[m, n]
    tau_x = (rho * u.T * x)
    # if x.shape == [m, n]: Hx[m, n] = x[m, n] -  u[m, 1] * tau_x[1, n]
    Hx = x + u * (-tau_x)

    return Hx, rho, u


def qrsteps(mat_a, mat_b=None, b_step=False):
    """
    Orthogonal-triangular decomposition.
    Demonstrates Python + Numpy version of QR function.
    R is the upper trapezoidal matrix R that
    results from the orthogonal transformation, R = Q'*A.
    If b_step = True, this function shows the steps in
    the computation of R.  Press <enter> key after each step.
    
    Ref : QRSTEPS by Prof. Cleve Moler
    
    :param numpy.matrix mat_a: 
    :param numpy.matrix mat_b: 
    :param bool b_step: 
    :return: R, bout, residual
    """

    # check type
    assert isinstance(mat_a, np.matrix)
    assert isinstance(mat_b, np.matrix) or (mat_b is None)

    m_height_a, n_width_a = mat_a.shape

    def present_step():
        print('mat_a = \n%r' % mat_a)
        if (mat_b is not None):
            print('mat_b = \n%r' % mat_b)

    if b_step:
        present_step()

    for index_k in range(0, min([m_height_a - 1, n_width_a])):
        if b_step:
            print(('make elements below diagonal in the %d-th column ' % (index_k + 1)).ljust(60, '='))

        # Householder transformation
        # for kth iteration, operate on mat_a[k:m, k:n]
        col_vec_u = mat_a[index_k:, index_k].copy()
        sigma_scala = na.norm(col_vec_u)

        # skip if column already zero
        if sigma_scala:
            if col_vec_u[0, 0]:
                sigma_scala *= np.sign(col_vec_u[0, 0])

            # u = x + sigma e_k
            # hence, .copy() above is necessary
            col_vec_u[0, 0] += sigma_scala

            # rho_scala = (2 / (||u||^2)) = 1 / (sigma_scala * u[k])
            rho_scala = 1 / (np.conj(sigma_scala) * col_vec_u[0, 0])

            # kth column
            mat_a[index_k:, index_k] = 0.0
            mat_a[index_k, index_k] = -sigma_scala

            # remaining columns
            # tau[1, n] = rho * u.T[1, m] * x[m, n]
            row_vec_tau_x = rho_scala * (col_vec_u.T * mat_a[index_k:, (index_k + 1):])
            # Hx[m, n] = x[m, n] -  u[m, 1] * tau_x[1, n]
            mat_a[index_k:, (index_k + 1):] += (col_vec_u * (-row_vec_tau_x))

            # transform b
            if mat_b is not None:
                # tau_y[1, 1] = (rho * u.T[1, m] * y[m, 1])
                row_vec_tau_y = rho_scala * (col_vec_u.T * mat_b[index_k:, :])
                # Hy[m, 1] = y[m, 1] - u[m, 1] * tau_y[1, 1]
                mat_b[index_k:, :] += (col_vec_u * (-row_vec_tau_y))
        # end if sigma_scala
        if b_step:
            present_step()

    # return economical R
    return mat_a[:n_width_a, :], mat_b[:n_width_a, :], mat_b[n_width_a:, :]


def main_qrsteps():
    # data from C. Moler, Numerical computation with MATLAB, SIAM, 2008.
    s = (np.matrix([np.arange(1950, 2000 + 1, 10)]).T - 1950.0) / 50
    y = np.matrix([np.array([150.6970, 179.3230, 203.2120, 226.5050, 249.6330, 281.4220])]).T
    print('     s                   y')
    print(np.column_stack([s, y]))

    mat_x = np.column_stack([np.power(s, 2), s, np.ones_like(s)])

    mat_qr_r, mat_qr_z, mat_residual_qr = qrsteps(mat_x, y, True)
    mat_beta_qr = na.solve(mat_qr_r, mat_qr_z)
    mat_y_x_beta = y - mat_x * mat_beta_qr

    mat_beta_ls, residual_ls, rank_ls, s_ls = na.lstsq(mat_x, y)

    print('mat_beta_qr = \n%s' % repr(mat_beta_qr))
    print('mat_beta_ls = \n%s' % repr(mat_beta_ls))

    print('mat_residual_qr = \n%s' % repr(mat_residual_qr))
    print('mat_y_x_beta = \n%s' % repr(mat_y_x_beta))
    print('na.norm(mat_residual_qr) = \n%s' % repr(na.norm(mat_residual_qr)))
    print('na.norm(mat_y_x_beta) = \n%s' % repr(na.norm(mat_y_x_beta)))
    print('residual_ls = \n%s' % repr(residual_ls))
    print('np.sqrt(residual_ls) = \n%s' % repr(np.sqrt(residual_ls)))


if __name__ == '__main__':
    import sys
    if 1 < len(sys.argv):
        if sys.argv[1].startswith('qr'):
            main_qrsteps()
        elif sys.argv[1].startswith('householder'):
            main_householder()
        else:
            raise ValueError(sys.argv[1])
    else:
        raise ValueError('No script argument')
