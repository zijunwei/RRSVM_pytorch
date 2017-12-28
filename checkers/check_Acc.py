import os
from py_utils import dir_utils
import torch

user_root = os.path.expanduser('~')
directory = os.path.join(user_root, 'Dev/RRSVM_pytorch/snapshots')

snapshots = dir_utils.get_immediate_subdirectories(directory)
snapshots = sorted(snapshots, key=str.lower)
for s_snapshot in snapshots:
    snapshot_file = os.path.join(directory, s_snapshot, 'ckpt.t7')
    try:
        checkpoint = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
        best_acc = checkpoint['acc']
        print "{:s}\t{:0.4f}".format(s_snapshot, best_acc)
    except:
        print '{:s} Not Readable'.format(s_snapshot)
