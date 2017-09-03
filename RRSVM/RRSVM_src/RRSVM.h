void RRSVM_updateOutput(THFloatTensor *input, THFloatTensor *s, THFloatTensor *output, THLongTensor *indices,
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

void RRSVM_accGradParameters(THFloatTensor *input, THLongTensor * indices, THFloatTensor *gradOutput, THFloatTensor *gradS,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

