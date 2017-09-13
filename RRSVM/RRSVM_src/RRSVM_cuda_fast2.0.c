#include <THC/THC.h>
#include "cuda/RRSVM_kernel.h"
#include <stdio.h>
extern THCState *state;
#define real float

//void RRSVM_updateOutput_cuda(THCudaTensor *input, THCudaTensor *s, THCudaTensor *output, THCudaLongTensor *indices,
//                             int kW, int kH,
//                             int dW, int dH,
//                             int padW, int padH,
//                             int dilationW, int dilationH){
////    printf("RRSVM_updateOutput_cuda fast1.0 called\n");
//
//    int nInputPlane = THCudaTensor_size(state, s, 0);
//    int inputWidth   = THCudaTensor_size(state, input, 3);
//    int inputHeight  = THCudaTensor_size(state, input, 2);
//    int outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
//    int outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
//
//    int batchSize = THCudaTensor_size(state, input, 0);
//
//    input = THCudaTensor_newContiguous(state, input);
//    s = THCudaTensor_newContiguous(state, s);
//
//    THCudaTensor_resize4d(state, output, batchSize, nInputPlane, outputHeight, outputWidth);
//    THCudaTensor_zero(state,output);
//
//    THCudaLongTensor_resize5d(state, indices, batchSize, nInputPlane, outputHeight, outputWidth, kH * kW);
//    THCudaLongTensor_zero(state, indices);
//
//    real *output_data, *s_data;
//    long *indices_data;
//
////  input_data = THFloatTensor_data(input);
//    output_data = THCudaTensor_data(state, output);
//    indices_data = THCudaLongTensor_data(state, indices);
//    s_data = THCudaTensor_data(state, s);
//    int elt;
//    cudaStream_t stream = THCState_getCurrentStream(state);
//
//#pragma omp parallel for private(elt)
//
//    for (elt = 0; elt < batchSize; elt ++) {
//        THCudaTensor *input_d_h_w = THCudaTensor_new(state);
//
//        THCudaTensor_select(state, input_d_h_w, input, 0, elt);
//
//        for (int chl = 0; chl < nInputPlane; chl ++){
//            THCudaTensor *input_h_w = THCudaTensor_new(state);
//
//            THCudaTensor_select(state, input_h_w, input_d_h_w, 0, chl);
//
//            THCudaTensor * columns = THCudaTensor_newWithSize2d(state, 1*kW*kH, outputHeight*outputWidth);
//            THCudaTensor_zero(state,columns);
//
//            im2col(THCState_getCurrentStream(state), THCudaTensor_data(state, input_h_w), 1,
//                   inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));
////            real * column_data;
////            column_data = THCudaTensor_data(state, columns);
//            THCudaLongTensor *sorted_index_2d = THCudaLongTensor_new(state);
//            THCudaTensor *sorted_input_2d = THCudaTensor_new(state);
//            //TODO: check here to see if dim is 0 or 1
//            THCudaTensor_sort(state, sorted_input_2d, sorted_index_2d, columns, 0, 1);
//            real *sorted_input_2d_data = THCudaTensor_data(state, sorted_input_2d);
//            long *sorted_index_2d_data = THCudaLongTensor_data(state, sorted_index_2d);
//            fill_output_2d(sorted_input_2d_data, s_data, elt, chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  output_data, stream);
//            fill_indices_2d(sorted_index_2d_data, elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  indices_data, stream);
//
//            THCudaLongTensor_free(state, sorted_index_2d);
//            THCudaTensor_free(state, sorted_input_2d);
//            THCudaTensor_free(state, columns);
//            THCudaTensor_free(state, input_h_w);
//        }
//        THCudaTensor_free(state, input_d_h_w);
//    }
//
//    THCudaTensor_free(state, input);
//    THCudaTensor_free(state, s);
//}

void RRSVM_updateOutput_cuda(THCudaTensor *input, THCudaTensor *s, THCudaTensor *output, THCudaLongTensor *indices,
                             int kW, int kH,
                             int dW, int dH,
                             int padW, int padH,
                             int dilationW, int dilationH){
//    printf("RRSVM_updateOutput_cuda fast1.0 called\n");

    int nInputPlane = THCudaTensor_size(state, s, 0);
    int inputWidth   = THCudaTensor_size(state, input, 3);
    int inputHeight  = THCudaTensor_size(state, input, 2);
    int outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    int outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    int batchSize = THCudaTensor_size(state, input, 0);

    input = THCudaTensor_newContiguous(state, input);
    s = THCudaTensor_newContiguous(state, s);

    THCudaTensor_resize4d(state, output, batchSize, nInputPlane, outputHeight, outputWidth);
    THCudaTensor_zero(state,output);

    THCudaLongTensor_resize5d(state, indices, batchSize, nInputPlane, outputHeight, outputWidth, kH * kW);
    THCudaLongTensor_zero(state, indices);

    real *output_data, *s_data;
    long *indices_data;

//  input_data = THFloatTensor_data(input);
    output_data = THCudaTensor_data(state, output);
    indices_data = THCudaLongTensor_data(state, indices);
    s_data = THCudaTensor_data(state, s);
    int elt;
    cudaStream_t stream = THCState_getCurrentStream(state);

#pragma omp parallel for private(elt)

    for (elt = 0; elt < batchSize; elt ++) {
        THCudaTensor *input_d_h_w = THCudaTensor_new(state);

        THCudaTensor_select(state, input_d_h_w, input, 0, elt);

//        for (int chl = 0; chl < nInputPlane; chl ++){
//            THCudaTensor *input_h_w = THCudaTensor_new(state);

//            THCudaTensor_select(state, input_h_w, input_d_h_w, 0, chl);

            THCudaTensor * columns = THCudaTensor_newWithSize2d(state, kW*kH, nInputPlane * outputHeight * outputWidth);
            THCudaTensor_zero(state,columns);

            im3col(THCState_getCurrentStream(state), THCudaTensor_data(state, input_d_h_w), nInputPlane,
                   inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));
//            real * column_data;
//            column_data = THCudaTensor_data(state, columns);
            THCudaLongTensor *sorted_index_2d = THCudaLongTensor_new(state);
            THCudaTensor *sorted_input_2d = THCudaTensor_new(state);
            //TODO: check here to see if dim is 0 or 1
            THCudaTensor_sort(state, sorted_input_2d, sorted_index_2d, columns, 0, 1);
            real *sorted_input_2d_data = THCudaTensor_data(state, sorted_input_2d);
            long *sorted_index_2d_data = THCudaLongTensor_data(state, sorted_index_2d);
//            long chl = 0;

            fill_output_3d(sorted_input_2d_data, sorted_index_2d_data, s_data, elt, kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  output_data, indices_data, stream);

//            fill_output_2d(sorted_input_2d_data, s_data, elt, chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  output_data, stream);
//            fill_indices_2d(sorted_index_2d_data, elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  indices_data, stream);

            THCudaLongTensor_free(state, sorted_index_2d);
            THCudaTensor_free(state, sorted_input_2d);
            THCudaTensor_free(state, columns);
//            THCudaTensor_free(state, input_h_w);
//        }
        THCudaTensor_free(state, input_d_h_w);
    }

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, s);
}


void RRSVM_updateGradInput_cuda(THCudaTensor *s, THCudaLongTensor *indices, THCudaTensor *gradOutput, THCudaTensor *gradInput,
                                int inputWidth, int inputHeight,
                                int kW, int kH,
                                int dW, int dH,
                                int padW, int padH,
                                int dilationW, int dilationH){
//    printf("RRSVM_updateGradInput_cuda called\n");

    int nInputPlane = THCudaTensor_size(state, s, 0);

    s = THCudaTensor_newContiguous(state,s);
    gradOutput = THCudaTensor_newContiguous(state,gradOutput);
    indices = THCudaLongTensor_newContiguous(state, indices);

    long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long batchSize = gradOutput->size[0];

    THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_zero(state, gradInput);

//  THFloatTensor *column_1d = THFloatTensor_new();

    real *gradOutput_data, *s_data;
    long *indices_data;

    gradOutput_data = THCudaTensor_data(state, gradOutput);
    s_data = THCudaTensor_data(state, s);
    indices_data = THCudaLongTensor_data(state, indices);
    int elt;
    cudaStream_t stream = THCState_getCurrentStream(state);

    for ( elt = 0; elt < batchSize; elt ++) {
        // Matrix mulitply per output:
        THCudaTensor *gradinput_d_h_w = THCudaTensor_new(state);

        THCudaTensor_select(state, gradinput_d_h_w, gradInput, 0, elt);


//        for (int chl = 0; chl < nInputPlane; chl ++){
//            THCudaTensor *gradinput_h_w = THCudaTensor_new(state);

//            THCudaTensor_select(state, gradinput_h_w, gradinput_d_h_w, 0, chl);
//            THCudaTensor_resize3d(gradinput_h_w, 1, inputHeight, inputWidth);
            THCudaTensor * gradInputColumns = THCudaTensor_newWithSize2d(state, nInputPlane*kW*kH, outputHeight*outputWidth);
            THCudaTensor_zero(state, gradInputColumns);
            real * gradInputColumns_data = THCudaTensor_data(state, gradInputColumns);

            fill_gradInput_3d(gradOutput_data, s_data, indices_data, elt,  kH, kW, nInputPlane, outputHeight, outputWidth, gradInputColumns_data, stream);
            col2im(THCState_getCurrentStream(state), gradInputColumns_data, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, gradinput_d_h_w));

            THCudaTensor_free(state, gradInputColumns);
//            THCudaTensor_free(state, gradinput_h_w);

//        }
        THCudaTensor_free(state, gradinput_d_h_w);

    }

    THCudaTensor_free(state, s);
    THCudaLongTensor_free(state, indices);
    THCudaTensor_free(state, gradOutput);
}

void RRSVM_accGradParameters_cuda(THCudaTensor *input, THCudaLongTensor * indices, THCudaTensor *gradOutput, THCudaTensor *gradS,
                                  int kW, int kH,
                                  int dW, int dH,
                                  int padW, int padH,
                                  int dilationW, int dilationH){

    int nInputPlane = THCudaTensor_size(state, input, 1);

    input = THCudaTensor_newContiguous(state, input);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    indices = THCudaLongTensor_newContiguous(state, indices);



    long inputWidth   = input->size[3];
    long inputHeight  = input->size[2];
    long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long batchSize = input->size[0];

    THCudaTensor_zero(state, gradS);

    real /**input_data,*/ *gradOutput_data, *gradS_data;
    long *indices_data;

    gradOutput_data = THCudaTensor_data(state, gradOutput);
    indices_data = THCudaLongTensor_data(state, indices);
    gradS_data = THCudaTensor_data(state, gradS);
    long elt;
    cudaStream_t stream = THCState_getCurrentStream(state);

    for ( elt = 0; elt < batchSize; elt ++) {

        THCudaTensor *input_d_h_w = THCudaTensor_new(state);

        THCudaTensor_select(state, input_d_h_w, input, 0, elt);

//        for (int chl = 0; chl < nInputPlane; chl ++){
//            THCudaTensor *input_h_w = THCudaTensor_new(state);
//            THCudaTensor_select(state, input_h_w, input_d_h_w, 0, chl);

            THCudaTensor * columns = THCudaTensor_newWithSize2d(state, nInputPlane*kW*kH, outputHeight*outputWidth);

            im2col(THCState_getCurrentStream(state), THCudaTensor_data(state, input_d_h_w), nInputPlane,
                   inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));


            real * column_data = THCudaTensor_data(state, columns);

            fill_gradS_3d(gradOutput_data, column_data, indices_data,
                          elt, kH, kW, nInputPlane, outputHeight, outputWidth, gradS_data, stream);

            THCudaTensor_free(state, columns);
//            THCudaTensor_free(state, input_h_w);
//        }
        THCudaTensor_free(state, input_d_h_w);
    }

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, gradOutput);
    THCudaLongTensor_free(state, indices);

}
//
//void RRSVM_accGradParameters_cuda(THCudaTensor *input, THCudaLongTensor * indices, THCudaTensor *gradOutput, THCudaTensor *gradS,
//                                  int kW, int kH,
//                                  int dW, int dH,
//                                  int padW, int padH,
//                                  int dilationW, int dilationH){
//
//    int nInputPlane = THCudaTensor_size(state, input, 1);
//
//    input = THCudaTensor_newContiguous(state, input);
//    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
//    indices = THCudaLongTensor_newContiguous(state, indices);
//
//
//
//    long inputWidth   = input->size[3];
//    long inputHeight  = input->size[2];
//    long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
//    long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
//
//    long batchSize = input->size[0];
//
//    THCudaTensor_zero(state, gradS);
//
//    real /**input_data,*/ *gradOutput_data, *gradS_data;
//    long *indices_data;
//
//    gradOutput_data = THCudaTensor_data(state, gradOutput);
//    indices_data = THCudaLongTensor_data(state, indices);
//    gradS_data = THCudaTensor_data(state, gradS);
//    long elt;
//    cudaStream_t stream = THCState_getCurrentStream(state);
//
//    for ( elt = 0; elt < batchSize; elt ++) {
//
//        THCudaTensor *input_d_h_w = THCudaTensor_new(state);
//
//        THCudaTensor_select(state, input_d_h_w, input, 0, elt);
//
//        for (int chl = 0; chl < nInputPlane; chl ++){
//            THCudaTensor *input_h_w = THCudaTensor_new(state);
//            THCudaTensor_select(state, input_h_w, input_d_h_w, 0, chl);
//
//            THCudaTensor * columns = THCudaTensor_newWithSize2d(state, 1*kW*kH, outputHeight*outputWidth);
//
//            im2col(THCState_getCurrentStream(state), THCudaTensor_data(state, input_h_w), 1,
//                   inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THCudaTensor_data(state, columns));
//
//
//            real * column_data = THCudaTensor_data(state, columns);
//
//            fill_gradS_2d(gradOutput_data, column_data, indices_data,
//                          elt, chl, kH, kW, nInputPlane, outputHeight, outputWidth, gradS_data, stream);
//
//            THCudaTensor_free(state, columns);
//            THCudaTensor_free(state, input_h_w);
//        }
//        THCudaTensor_free(state, input_d_h_w);
//    }
//
//    THCudaTensor_free(state, input);
//    THCudaTensor_free(state, gradOutput);
//    THCudaLongTensor_free(state, indices);
//
//}