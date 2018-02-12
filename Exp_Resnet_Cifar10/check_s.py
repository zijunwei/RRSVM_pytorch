from __future__ import print_function

import torch

import glob
import os
import numpy as np

save_dir='checkpoint'
checkpoint_files=glob.glob(os.path.join(save_dir, '*.th'))
checkpoint_files.sort()
for s_checkpointfile in checkpoint_files:

    checkpoint = torch.load(s_checkpointfile, map_location=lambda storage, loc: storage)
    s = checkpoint['state_dict']['pool.s']
    print("Check S!")



