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

    tau_y = (rho * u.T * y)[0, 0]
    Hy = y - tau_y * u

    return Hx, Hy, u


def householder_k(x, k):
    """
    Householder reflection such that Hx points to kth axis
    
    :param numpy.matrix x: 
    :param int k: 
    :return: 
    """
    # find u bisecting x and x axis
    sigma = np.sqrt((x.T * x)[0, 0])
    e_k = np.matrix(np.zeros_like(x))
    e_k[k, 0] = 1.0
    u = x + sigma * e_k

    # Householder reflection
    rho = (2 / (u.T * u))[0, 0]
    tau_x = (rho * u.T * x)[0, 0]
    Hx = x - tau_x * u

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
    if isinstance(mat_b, np.matrix) :
        assert (1 == mat_b.shape[1])

    size_m, size_n = mat_a.shape

    def present_step():
        print('mat_a = \n%r' % mat_a)
        if (mat_b is not None):
            print('mat_b = \n%r' % mat_b)

    if b_step:
        present_step()

    for index_k in range(0, min([size_m-1, size_n])):
        if b_step:
            print(('make elements below diagonal in the %d-th column ' % (index_k + 1)).ljust(60, '='))

        # Householder transformation
        index_array_i = np.arange(index_k, size_m, dtype=int)
        mat_u = mat_a[index_array_i, index_k].copy()
        sigma = na.norm(mat_u)

        # skip if column already zero
        if sigma:
            if mat_u[0, 0]:
                sigma *= np.sign(mat_u[0, 0])

            mat_u[0, 0] += sigma

            rho = 1 / (np.conj(sigma) * mat_u[0, 0])

            # kth column
            mat_a[index_array_i, index_k] = 0.0
            mat_a[index_k, index_k] = -sigma

            # remaining columns
            index_array_j = np.arange(index_k + 1, size_n, dtype=int)

            mat_v = rho * (mat_u.T * mat_a[index_k:size_m, (index_k + 1):size_n])
            mat_a[index_k:size_m, (index_k + 1):size_n] += ((mat_u * mat_v) * -1)

            # transform b
            if mat_b is not None:
                tau = rho * (mat_u.T * mat_b[index_k:size_m, 0])[0, 0]
                mat_b[index_k:size_m, 0] += (-tau * mat_u)
        # end if sigma
        if b_step:
            present_step()

    # return economical R
    return mat_a[:size_n, :], mat_b[:size_n, :], mat_b[size_n:, :]


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
