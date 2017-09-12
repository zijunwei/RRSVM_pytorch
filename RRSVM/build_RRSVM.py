import os
import torch
from torch.utils.ffi import create_extension

# set to True to enable cuda usage
use_cuda = False

sources = ['RRSVM_src/RRSVM.c']
headers = ['RRSVM_src/RRSVM.h']
defines = []
with_cuda = False
extra_objects = []


if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['RRSVM_src/RRSVM_cuda.c']
    headers += ['RRSVM_src/RRSVM_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True


    this_file = os.path.dirname(os.path.realpath(__file__))
    print(this_file)

    # run nvcc
    os.system('nvcc -c -o RRSVM_src/RRSVM_kernel.c.o RRSVM_src/cuda/RRSVM_kernel.c -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52')

    extra_objects = ['RRSVM_src/RRSVM_kernel.c.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]


ffi = create_extension(
    '_ext.RRSVM',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=['-fopenmp'],
    extra_link_args=["-fopenmp"],
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()