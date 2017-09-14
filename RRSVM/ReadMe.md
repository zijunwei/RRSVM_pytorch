## README:

### structure:
The python interface is in `RRSVM.py`, the c backend is implemented in `RRSVM_src` in which the cuda code is in `cuda` directory, currently only float is supported

### Compile

Run build_RRSVM.py to set up

### Correctly passed all the test, but the "erros" still shows up?:

1. When the padding is not 0, there will be multiple 0s in the input, the sort algorithm will return different order, this will cause error message, but it's alright
2. For finite differeciation, the sort order of the input will change if the input changes significantly, so this is cause error, one way to check is to check if there are roughly same number of passess vs. fails, also check the s parameter's graident is correct because the change of the s doesn't change the order

### Speed:
Due to the sorting used, the GPU speed is slightly faster than the CPU version (sometimes 7times faster, some times 1.5), but it seems that this RRSVM operation is consistently faster than Max-pooling