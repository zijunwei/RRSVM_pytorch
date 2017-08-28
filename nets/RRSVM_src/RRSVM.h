int c_forward(THFloatTensor *output, THFloatTensor *input, THFloatTensor *s, THFloatTensor *indices);

int c_backward_grad_input(THFloatTensor *grad_input, THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *s, THFloatTensor *indices);

int c_backward_grad_params(THFloatTensor *grad_input, THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *s, THFloatTensor * indices);

