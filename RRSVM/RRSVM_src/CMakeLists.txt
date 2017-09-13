cmake_minimum_required(VERSION 3.8)
project(RRSVM_src)

set(CMAKE_CXX_STANDARD 11)
include_directories(/home/zwei/anaconda2/envs/pytorch/lib/python2.7/site-packages/torch/lib/include)

set(SOURCE_FILES
        cuda/RRSVM_kernel.c
        cuda/RRSVM_kernel.h
        RRSVM.c
        RRSVM.h
        RRSVM_cuda.c
        RRSVM_cuda.h
        RRSVM_singlethread.c
        RRSVM_cuda_fast1.0.c
        RRSVM_cuda_fast2.0.c)

add_executable(RRSVM_src ${SOURCE_FILES})