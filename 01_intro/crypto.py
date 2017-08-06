import numpy as np


def crypto(x_str):
    """
    CRYPTO Cryptography example.
    y = crypto(x) converts an ASCII text string into another, coded string.
    The function is its own inverse, so crypto(crypto(x)) gives x back.
    See also: ENCRYPT.

      Copyright 2014 Cleve Moler
      Copyright 2014 The MathWorks, Inc.

    Ref : C. Moler, Numerical computation with MATLAB, SIAM, 2008.
    """
    # Use a two-character Hill cipher with arithmetic modulo 97, a prime.
    p = 97

    # Choose two characters above ASCII 128 to expand set from 95 to 97.
    c1 = chr(169)
    c2 = chr(174)
    v1 = 127
    v2 = 128

    x_int_array = convert_2_int_array(x_str, c1, v1, c2, v2, p)

    # Reshape into a matrix with 2 rows and floor(length(x)/2) columns.
    n = int(2 * np.floor(len(x_int_array) / 2))
    x_int_mat = np.matrix(np.reshape(x_int_array[0:n], (2, n // 2)))

    # Encode with matrix multiplication modulo p.
    mat_a_int = np.matrix([[71, 2],
                           [2, 26]], dtype=int)

    mat_y = np.mod(mat_a_int * x_int_mat, p)

    # Reshape into a single row.
    y_int_list = np.reshape(mat_y, (1, n)).tolist()[0]

    # If length(x) is odd, encode the last character.
    if len(x_int_array) > n:
        y_int_list.append(np.mod((p - 1) * x_int_array[-1], p))

    # Convert to ASCII characters.
    result = convert_to_ascii(y_int_list, c1, v1, c2, v2)

    return result


def convert_2_int_array(x_str, c1, v1, c2, v2, p):
    x_char_array = np.array(tuple(x_str), dtype='|U1')
    x_char_array[x_char_array == c1] = v1
    x_char_array[x_char_array == c2] = v2

    # Convert to integers mod p.
    x_int_array = np.mod((x_char_array.view(np.int8) - 32), p)

    return x_int_array


def convert_to_ascii(y_int_list, c1, v1, c2, v2):
    y_int_array = np.array(y_int_list, dtype=np.uint8)
    # y_int_array += 32

    y_int_no_offset_list = [y_int_offset + 32 for y_int_offset in y_int_list]

    # y_char_array = y_int_array.astype(np.uint8).view(dtype='U1')
    # y_char_array[y_char_array == chr(127)] = c1
    # y_char_array[y_char_array == chr(128)] = c2
    y_char_recover_127_list = []

    for y_int in y_int_no_offset_list:
        append_this = chr(y_int)
        if chr(v1) == append_this:
            append_this = c1
        elif chr(v2) == append_this:
            append_this = c2
        y_char_recover_127_list.append(append_this)

    result = ''.join(y_char_recover_127_list)

    return result
