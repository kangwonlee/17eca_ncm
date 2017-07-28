import os
import sys

import ipynb_remove_output as ir


def main(argv):
    if 1 < len(argv):
        # argument # check passed

        # path walk
        for dir_name, dir_list, filename_list in os.walk(argv[1]):
            # ignore list
            if not (('.git' in dir_name) or ('utils3' in dir_name)):
                # filename loop
                for filename in filename_list:
                    if '.ipynb' == os.path.splitext(filename)[-1]:
                        nb_path = os.path.join(dir_name, filename)
                        print('process %s' % nb_path)
                        ir.process_nb_file(nb_path)
    else:
        # pass?
        ValueError('path not given')


if __name__ == '__main__':
    main(sys.argv)
