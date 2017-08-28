int c_forward(THFloatTensor *input, THFloatTensor *s,
		       THFloatTensor *output);

int c_backward_grad_input(THFloatTensor *grad_output, THFloatTensor *grad_input, THFloatTensor *input, THFloatTensor *s);

int c_backward_grad_params(THFloatTensor *grad_output, THFloatTensor *grad_input, THFloatTensor *input, THFloatTensor *s);

