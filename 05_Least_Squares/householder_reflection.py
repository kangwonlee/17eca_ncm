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
    Hx, rho, u = householder_k(x, k)

    tau_y = (rho * u.T * y)[0, 0]
    Hy = y - tau_y * u

    return Hx, Hy, u


def householder_k(x, k):
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


if __name__ == '__main__':
    main()
