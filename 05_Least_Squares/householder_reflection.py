import matplotlib.pyplot as plt
import numpy as np


def main():
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
    :return: R, bout
    """

    # check type
    assert isinstance(mat_a, np.matrix)
    assert isinstance(mat_b, np.matrix) or (mat_b is None)

    m, n = mat_a.shape

    def present_step():
        print('mat_a = %r' % mat_a)
        if (mat_b is not None):
            print('mat_b = %r' % mat_b)

    if b_step:
        present_step()


if __name__ == '__main__':
    main()
