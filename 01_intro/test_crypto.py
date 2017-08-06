import unittest

import crypto


class TestCrypto(unittest.TestCase):
    def test_crypto(self):
        input_str = ''.join([chr(i) for i in range(32, (95+32+1))])

        encrypted = crypto.crypto(input_str)
        expected_for_now = '''®yyyhyyyPyyy8yyy yyyiyyyQyyy9yyy!yyyjyyyRyyy:yyy"yyykyyySyyy;yyy#yyylyyyTyyy<yyy$yyymyyyUyyy=yyy%yyynyyyVyyy>yyy&yyyoyyyWyyy?yyy'yyypyyyXyyy@yyy(yyyqyyyYyyyAyyy)yyyryyyZyyyByyy*yyysyyy[yyyCyyytjjj/jjjKjjjgjjj"jjj>jjjZjjjvjjj1jjjMjjjijjj$jjj@jjj\jjjxjjj3jjjOjjjkjjj&jjjBjjj^jjjzjjj5jjjQjjjmjjj(jjjDjjj`jjj|jjj7jjjSjjjojjj*jjjFjjjbjjj~jjj9jjjUjjjqjjj,jjjHjjjdjjj®jjj;jjjWjjjsjjj.jjjJjjj'''

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


if __name__ == '__main__':
    unittest.main()
