void RRSVM_updateOutput_cuda(THCudaTensor *input, THCudaTensor *s, THCudaTensor *output, THCudaLongTensor *indices,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

void RRSVM_updateGradInput_cuda( THCudaTensor *s, THCudaLongTensor *indices, THCudaTensor *gradOutput, THCudaTensor *gradInput,
    int inputWidth, int inputHeight,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

void RRSVM_accGradParameters_cuda(THCudaTensor *input, THCudaLongTensor * indices, THCudaTensor *gradOutput, THCudaTensor *gradS,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

