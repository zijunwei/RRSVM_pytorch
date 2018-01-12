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


### Progress:

#### Jan 12, 2018

**Sanity Check**
I checked the gradients & backpropgation of all the variantations of RRSVM in CPU and Cuda versions. 
All seem to be correct. The only error comes from the GradInput when the value is large. This is mostly
likely to be caused by the instabilities of float format. I checked the officially implemented conv2d
when using FloatTensor as input, it happens to have errors. But when I change the input variables to be
DoubleTensor, the errors are gone. 


**A possible Problem** In constrast to conv operations, in RRSVM, suppose the weight is a 1 by 4 vector. Then the first
is always corresponding to the largest value (since the input is sorted), it will receive the largest gradient change all the
time. Will this bring instabilities? Should we apply lower learning rates to the higher indices in w? 

At least the first thing to do is to reduce the learning rate of the whole RRSVM layer to see if it gets stable


**A possbile extension** We have been attempting to investigate the accuracy/speed/memory gain of RRSVM but received all 
negative gains. The possible next direction would be investigating the explaination/intepretibility gain... 
(See the paper: https://arxiv.org/pdf/1711.05611.pdf)
