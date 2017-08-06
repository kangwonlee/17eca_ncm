import unittest

import numpy as np
import numpy.linalg as na

import crypto


class TestCrypto(unittest.TestCase):
    def test_crypto(self):
        input_str = ''.join([chr(i) for i in range(32, (95 + 32))])

        encrypted = crypto.crypto(input_str)
        expected_for_now = '''®yyyhyyyPyyy8yyy yyyiyyyQyyy9yyy!yyyjyyyRyyy:yyy"yyykyyySyyy;yyy#yyylyyyTyyy<yyy$yyymyyyUyyy=yyy%yyynyyyVyyy>yyy&yyyoyyyWyyy?yyy'yyypyyyXyyy@yyy(yyyqyyyYyyyAyyy)yyyryyyZyyyByyy*yyysyyy[yyyCyyytjjj/jjjKjjjgjjj"jjj>jjjZjjjvjjj1jjjMjjjijjj$jjj@jjj\jjjxjjj3jjjOjjjkjjj&jjjBjjj^jjjzjjj5jjjQjjjmjjj(jjjDjjj`jjj|jjj7jjjSjjjojjj*jjjFjjjbjjj~jjj9jjjUjjjqjjj,jjjHjjjdjjj®jjj;jjjWjjjsjjj.jjjJjjj'''

        self.assertEqual(len(input_str), len(expected_for_now))

        self.assertEqual(expected_for_now, encrypted)

        decrypted = crypto.crypto(encrypted)

        self.assertEqual(input_str, decrypted)

    def test_convert_2_int_array(self):
        p = 97
        c1 = chr(169)
        c2 = chr(174)
        v1 = 127
        v2 = 128

        input_str = 'TV'
        x_int_array = crypto.convert_2_int_array(input_str, c1, v1, c2, v2, p)
        self.assertEqual(len(input_str), len(x_int_array))

        expected_array = np.array([52, 54])
        norm = na.norm(expected_array - x_int_array)

        self.assertAlmostEqual(0.0, norm)

    def test_convert_to_ascii(self):
        p = 97
        c1 = chr(169)
        c2 = chr(174)
        v1 = 127
        v2 = 128

        input_str = 'TV'
        x_int_array = crypto.convert_2_int_array(input_str, c1, v1, c2, v2, p)
        result = crypto.convert_to_ascii(x_int_array, c1, v1, c2, v2)
        self.assertEqual(len(input_str), len(result))

        self.assertEqual(input_str, result)

    def test_convert_to_ascii(self):
        p = 97
        c1 = chr(169)
        c2 = chr(174)
        v1 = 127
        v2 = 128

        x_int_array = np.array((17, 53))
        result = crypto.convert_to_ascii(x_int_array, c1, v1, c2, v2)
        self.assertEqual(len(x_int_array), len(result))

        expected = '1U'

        self.assertEqual(expected, result)

    def test_convert_to_ascii_long(self):
        p = 97
        c1 = chr(169)
        c2 = chr(174)
        v1 = 127
        v2 = 128

        input_str = ''.join([chr(i) for i in range(32, (95 + 32))])
        x_int_array = crypto.convert_2_int_array(input_str, c1, v1, c2, v2, p)
        result = crypto.convert_to_ascii(x_int_array, c1, v1, c2, v2)
        self.assertEqual(len(input_str), len(result))

        for c_input, c_result in zip(input_str, result):
            self.assertEqual(c_input, c_result)


if __name__ == '__main__':
    unittest.main()
