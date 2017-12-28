#include <THC/THC.h>
#include "cuda/RRSVM_kernel.h"
#include <stdio.h>
extern THCState *state;
#define real float

void RRSVM_updateOutput_cuda(THCudaTensor *input, THCudaTensor *s, THCudaTensor *softmax_s, THCudaTensor *output, THCudaLongTensor *indices,
                             int kW, int kH,
                             int dW, int dH,
                             int padW, int padH,
                             int dilationW, int dilationH){

    int batchSize = THCudaTensor_size(state, input, 0);
    int nInputPlane = THCudaTensor_size(state, s, 0);
    int inputHeight  = THCudaTensor_size(state, input, 2);
    int inputWidth   = THCudaTensor_size(state, input, 3);

    int outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    int outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;


    input = THCudaTensor_newContiguous(state, input);
    s = THCudaTensor_newContiguous(state, s);

    THCudaTensor_resize4d(state, output, batchSize, nInputPlane, outputHeight, outputWidth);
    //TODO: this will cause memory leak
//    output = THCudaTensor_newContiguous(state, output);
    THCudaTensor_zero(state,output);

    THCudaLongTensor_resize5d(state, indices, batchSize, nInputPlane, outputHeight, outputWidth, kH * kW);
//    indices = THCudaLongTensor_newContiguous(state, indices);
    THCudaLongTensor_zero(state, indices);

    THCudaTensor_resize2d(state, softmax_s, nInputPlane, kW*kH);
    THCudaTensor_zero(state, softmax_s);


    real *output_data, *s_data, *softmax_s_data;
    long *indices_data;

    output_data = THCudaTensor_data(state, output);
    s_data = THCudaTensor_data(state, s);
    indices_data = THCudaLongTensor_data(state, indices);
    softmax_s_data = THCudaTensor_data(state, softmax_s);

    SoftMax_updateOutput(THCState_getCurrentStream(state), s_data, softmax_s_data, kH, kW, nInputPlane);


    int elt;

//#pragma omp parallel for private(elt)

    for (elt = 0; elt < batchSize; elt ++) {
        THCudaTensor *input_d_h_w = THCudaTensor_new(state);

        THCudaTensor_select(state, input_d_h_w, input, 0, elt);

        THCudaTensor * columns = THCudaTensor_newWithSize2d(state, kW*kH, nInputPlane * outputHeight * outputWidth);
        THCudaTensor_zero(state,columns);

        im3col(THCState_getCurrentStream(state), THCudaTensor_data(state, input_d_h_w), nInputPlane,
               inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));

        THCudaLongTensor *sorted_index_2d = THCudaLongTensor_new(state);
        THCudaTensor *sorted_input_2d = THCudaTensor_new(state);

        THCudaTensor_sort(state, sorted_input_2d, sorted_index_2d, columns, 0, 1);
        real *sorted_input_2d_data = THCudaTensor_data(state, sorted_input_2d);
        long *sorted_index_2d_data = THCudaLongTensor_data(state, sorted_index_2d);

        fill_output_3d(sorted_input_2d_data, sorted_index_2d_data, softmax_s_data, elt, kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  output_data, indices_data, THCState_getCurrentStream(state));
        
        THCudaLongTensor_free(state, sorted_index_2d);
        THCudaTensor_free(state, sorted_input_2d);
        THCudaTensor_free(state, columns);
        THCudaTensor_free(state, input_d_h_w);
    }

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, s);
//    THCudaTensor_free(state, output);
//    THCudaLongTensor_free(state, indices);
}


void RRSVM_updateGradInput_cuda(THCudaTensor *s, THCudaLongTensor *indices, THCudaTensor *gradOutput, THCudaTensor *gradInput,
                                int inputWidth, int inputHeight,
                                int kW, int kH,
                                int dW, int dH,
                                int padW, int padH,
                                int dilationW, int dilationH){

    int batchSize = THCudaTensor_size(state, gradOutput, 0);
    int nInputPlane = THCudaTensor_size(state, s, 0);
    int outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    int outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;


    s = THCudaTensor_newContiguous(state,s);
    gradOutput = THCudaTensor_newContiguous(state,gradOutput);
    indices = THCudaLongTensor_newContiguous(state, indices);


    THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
//    gradInput = THCudaTensor_newContiguous(state, gradInput);
    THCudaTensor_zero(state, gradInput);

    real *gradOutput_data, *s_data;
    long *indices_data;

    gradOutput_data = THCudaTensor_data(state, gradOutput);
    s_data = THCudaTensor_data(state, s);
    indices_data = THCudaLongTensor_data(state, indices);
    int elt;

    //Do a softmax cuda layer:



    for ( elt = 0; elt < batchSize; elt ++) {
        // Matrix mulitply per output:
        THCudaTensor *gradinput_d_h_w = THCudaTensor_new(state);
        THCudaTensor_select(state, gradinput_d_h_w, gradInput, 0, elt);

        THCudaTensor * gradInputColumns = THCudaTensor_newWithSize2d(state, nInputPlane*kW*kH, outputHeight*outputWidth);
        THCudaTensor_zero(state, gradInputColumns);
        real * gradInputColumns_data = THCudaTensor_data(state, gradInputColumns);

        fill_gradInput_3d(gradOutput_data, s_data, indices_data,
                          elt,  kH, kW, nInputPlane, outputHeight, outputWidth, gradInputColumns_data, THCState_getCurrentStream(state));
        col2im(THCState_getCurrentStream(state), gradInputColumns_data, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, gradinput_d_h_w));

        THCudaTensor_free(state, gradInputColumns);
        THCudaTensor_free(state, gradinput_d_h_w);
    }

    THCudaTensor_free(state, s);
    THCudaLongTensor_free(state, indices);
    THCudaTensor_free(state, gradOutput);
//    THCudaTensor_free(state, gradInput);
}

void RRSVM_accGradParameters_cuda(THCudaTensor *input, THCudaTensor *softmax_s, THCudaLongTensor * indices, THCudaTensor *gradOutput, THCudaTensor *gradS,
                                  THCudaTensor *gradSgradSoftmax,
                                  int kW, int kH,
                                  int dW, int dH,
                                  int padW, int padH,
                                  int dilationW, int dilationH){
    int batchSize = THCudaTensor_size(state, input,0);
    int nInputPlane = THCudaTensor_size(state, input, 1);
    int inputWidth   =THCudaTensor_size(state, input, 3);
    int inputHeight  = THCudaTensor_size(state, input, 2);
    int outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    int outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    input = THCudaTensor_newContiguous(state, input);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    indices = THCudaLongTensor_newContiguous(state, indices);
    softmax_s = THCudaTensor_newContiguous(state, softmax_s);

    THCudaTensor_resize2d(state, gradS, nInputPlane, kH * kW);
//    gradS = THCudaTensor_newContiguous(state, gradS);
    THCudaTensor_zero(state, gradS);

    THCudaTensor_resize2d(state, gradSgradSoftmax, nInputPlane, kH * kW);
//    gradS = THCudaTensor_newContiguous(state, gradS);
    THCudaTensor_zero(state, gradSgradSoftmax);

    real *gradOutput_data, *gradS_data, *gradSgradSoftmax_data, *softmax_s_data;
    long *indices_data;

    gradOutput_data = THCudaTensor_data(state, gradOutput);
    indices_data = THCudaLongTensor_data(state, indices);
    gradS_data = THCudaTensor_data(state, gradS);
    gradSgradSoftmax_data = THCudaTensor_data(state, gradSgradSoftmax);
    softmax_s_data = THCudaTensor_data(state, softmax_s);

    //This is not approprate for add?
    int elt;
    for (elt = 0; elt < batchSize; elt ++) {

        THCudaTensor *input_d_h_w = THCudaTensor_new(state);
        THCudaTensor_select(state, input_d_h_w, input, 0, elt);

        THCudaTensor * columns = THCudaTensor_newWithSize2d(state, nInputPlane*kW*kH, outputHeight*outputWidth);

        im2col(THCState_getCurrentStream(state), THCudaTensor_data(state, input_d_h_w), nInputPlane,
               inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));

        real * column_data = THCudaTensor_data(state, columns);

        fill_gradS_3d(gradOutput_data, column_data, indices_data,
                      elt, kH, kW, nInputPlane, outputHeight, outputWidth, gradS_data, THCState_getCurrentStream(state));

        THCudaTensor_free(state, columns);
        THCudaTensor_free(state, input_d_h_w);
    }

    SoftMax_UpdateGradInput(THCState_getCurrentStream(state), softmax_s_data, gradS_data, gradSgradSoftmax_data, kH, kW, nInputPlane );
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, gradOutput);
    THCudaTensor_free(state, softmax_s);
    THCudaLongTensor_free(state, indices);

//    THCudaTensor_free(state, gradS);

}
