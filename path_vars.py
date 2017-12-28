import os
import sys
import py_utils.dir_utils as dir_utils
user_root = os.path.expanduser('~')
dataset_root = os.path.join(user_root,'datasets', 'RRSVM_dataset')
dataset_root = dir_utils.get_dir(dataset_root)
print "DEBUG"

