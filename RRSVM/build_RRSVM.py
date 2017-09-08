import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['RRSVM_src/RRSVM.c']
headers = ['RRSVM_src/RRSVM.h']
defines = []
with_cuda = False

# if torch.cuda.is_available():
#     print('Including CUDA code.')
#     sources += ['RRSVM_src/RRSVM_cuda.c']
#     headers += ['RRSVM_src/RRSVM_cuda.h']
#     defines += [('WITH_CUDA', None)]
#     with_cuda = True

# extra_objects = ['RRSVM_src//roi_pooling.cu.o']
# extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.RRSVM',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()