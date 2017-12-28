## RRSVM_pytorch: An End-to-End Implementation of Zwei, MHoai@CVPR16

Currently tested and worked on bigmind. 
###  Description
The modules of **RRSVM** and **SoftMaxRRSVM** are implemented in the **RRSVM** directory.

The source code are saved in RRSVM_src and SoftMaxRRSVM_src respectively.

Some tests on the modules are saved in RRSVM/Tests

For each of the module, there are two implementations: the CPU version and the GPU version (XXX_cuda.c), the XXX_cuda.c calls the kernel file saved in cuda dir.



### How to start:
1.  Download pytorch with python 2.7 and GPU support
2.  Compile the RRSVM module and SoftMaxRRSVM module by:
	```
	cd RRSVM
	python build_RRSVM.py
	python build_SoftMaxRRSVM.py
	```
3. Run Mnist_main.py to check if everything is correct.


### PS

the dataset is saved in ~\datasets\RRSVM_datasets,  if not available, it will download and create them.
