import numpy as np


def magic(N):
    if (N % 2):
        result = magic_odd(N)
    elif 0 == (N % 4):
        result = magic_four(N)
    elif (N % 4):
        result = magic_even_odd(N)
    else:
        raise NotImplementedError

    # check result
    assert_magic_square(result, N)

    return result


def magic_odd(N):
    # scipython.com/book/chapter-6-numpy/examples/creating-a-magic-square/
    magic_square = np.matrix(np.zeros((N, N), dtype=int))

    n = 1
    i, j = 0, N // 2

    while n <= N ** 2:
        magic_square[i, j] = n
        n += 1
        newi, newj = (i - 1) % N, (j + 1) % N
        if magic_square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj

    return magic_square


def magic_four(N):
    # https://m.blog.naver.com/askmrkwon/220768685076 (in Korean)
    magic_ascending = np.array(range(1, N ** 2 + 1), dtype=int).reshape((N, N))
    magic_descending = np.array(range(N ** 2, 0, -1), dtype=int).reshape((N, N))
    magic_four_mat = np.matrix(np.zeros((N, N)), dtype=int)

    for i_row in range(0, N):
        for j_col in range(0, N):
            if (0 == (abs(i_row - j_col) % 4)) or (0 == (i_row + j_col + 1) % 4):
                magic_four_mat[i_row, j_col] = magic_descending[i_row, j_col]
            else:
                magic_four_mat[i_row, j_col] = magic_ascending[i_row, j_col]

    assert 0 < magic_four_mat.min()

    return magic_four_mat


def magic_even_odd(even):
    # https://m.blog.naver.com/askmrkwon/220768685076 (in Korean)

    assert 0 == (even % 2)

    odd = even // 2  # 2n + 1

    assert (odd % 2)

    upper_left = magic_odd(odd)
    lower_right = upper_left + (odd * odd)
    upper_right = lower_right + (odd * odd)
    lower_left = upper_right + (odd * odd)

    assert upper_left.min() == 1
    assert upper_left.max() == odd * odd

    assert lower_right.min() == upper_left.max() + 1
    assert lower_right.max() == odd * odd * 2

    assert upper_right.min() == lower_right.max() + 1
    assert upper_right.max() == odd * odd * 3

    assert lower_left.min() == upper_right.max() + 1
    assert lower_left.max() == even * even

    n = (odd - 1) // 2

    # exchange left
    temp = upper_left[:, 0:n].copy()
    upper_left[:, 0:n] = lower_left[:, 0:n]
    lower_left[:, 0:n] = temp

    # exchange right
    if 1 < n:
        temp = upper_right[:, -(n - 1):].copy()
        upper_right[:, -(n - 1):] = lower_right[:, -(n - 1):]
        lower_right[:, -(n - 1):] = temp

    # exchange left middle
    temp = upper_left[n, (n - 1):(n + 1)].copy()
    upper_left[n, (n - 1):(n + 1)] = lower_left[n, (n - 1):(n + 1)]
    lower_left[n, (n - 1):(n + 1)] = temp

    result = np.row_stack((np.column_stack((upper_left, upper_right)),
                           np.column_stack((lower_left, lower_right))))

    # check result
    assert_magic_square(result, even)

    return result


def assert_magic_square(mat, n):
    # check result
    magic_sum = np.sum(mat) / n
    row_sum_vector = np.sum(mat, 0)
    col_sum_vector = np.sum(mat, 1)
    assert np.abs(row_sum_vector - magic_sum).max() < 1e-7
    assert np.abs(row_sum_vector - magic_sum).min() < 1e-7
    assert np.abs(col_sum_vector - magic_sum).max() < 1e-7
    assert np.abs(col_sum_vector - magic_sum).min() < 1e-7


if __name__ == '__main__':
    import numpy.linalg as na


    def main():
        print(np.array([(n, na.matrix_rank(magic(n))) for n in range(3, 24 + 1)]))


    main()
