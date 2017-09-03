int c_forward_cuda(THFloatTensor *input, THFloatTensor *s,
		       THFloatTensor *output);

int c_backward_grad_input_cuda(THFloatTensor *grad_output, THFloatTensor *grad_input, THFloatTensor *input, THFloatTensor *s);

int c_backward_grad_params_cuda(THFloatTensor *grad_output, THFloatTensor *grad_input, THFloatTensor *input, THFloatTensor *s);