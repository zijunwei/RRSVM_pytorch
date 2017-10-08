
void RRSVM_updateOutput(THFloatTensor *input, THFloatTensor *s, THFloatTensor *softmax_s, THFloatTensor *output, THLongTensor *indices,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

void RRSVM_updateGradInput( THFloatTensor *s, THLongTensor *indices, THFloatTensor *gradOutput, THFloatTensor *gradInput,
    int inputWidth, int inputHeight,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH );

void RRSVM_accGradParameters(THFloatTensor *input, THFloatTensor *softmax_s, THLongTensor * indices, THFloatTensor *gradOutput, THFloatTensor *gradS, THFloatTensor *gradSoftmaxS,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

