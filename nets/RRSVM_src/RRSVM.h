void RRSVM_updateOutput(THFloatTensor *input, THFloatTensor *s, THFloatTensor *output, THLongTensor *indices, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

void RRSVM_updateGradInput(THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradInput, THFloatTensor *s, THLongTensor *indices, THFloatTensor * gradColumns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH );

void RRSVM_accGradParameters(THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradS, THLongTensor * indices, THFloatTensor *columns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH);

