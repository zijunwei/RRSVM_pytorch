#include <THC/THC.h>
#include "cuda/RRSVM_kernel.h"
#include <stdio.h>
extern THCState *state;
#define real float



void RRSVM_updateOutput_cuda(THCudaTensor *input, THCudaTensor *s, THCudaTensor *output, THCudaLongTensor *indices,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH){

  input = THCudaTensor_newContiguous(state, input);
  s = THCudaTensor_newContiguous(state, s);

  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, outputHeight, outputWidth);
  THCudaTensor_zero(state,output);
  THCudaLongTensor_resize5d(state, indices, batchSize, nInputPlane, outputHeight, outputWidth, kH * kW);
  THCudaLongTensor_zero(state, indices);


  long nInputPlane = THCudaTensor_size(state, s, 0);
  long inputWidth   = THCudaTensor_size(state, input, 3);
  long inputHeight  = THCudaTensor_size(state, input, 2);
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long batchSize = THCudaTensor_size(state, input, 0);



  real *input_data, *output_data, *s_data;
  long *indices_data;

  input_data = THCudaTensor_data(state, input);
  output_data = THCudaTensor_data(state, output);
  indices_data = THCudaLongTensor_data(state, indices);
  s_data = THCudaTensor_data(state, s);

  //TODO: This does the major job!
  RRSVM_updateOutput_cuda_laucher(

  );

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, s);
 }

void RRSVM_updateGradInput_cuda(THCudaTensor *s, THCudaLongTensor *indices, THCudaTensor *gradOutput, THCudaTensor *gradInput,
    int inputWidth, int inputHeight,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH){

  int nInputPlane = THCudaTensor_size(state, s, 0);

  s = THCudaTensor_newContiguous(state,s);
  gradOutput = THCudaTensor_newContiguous(state,gradOutput);
  indices = THCudaLongTensor_newContiguous(state, indices);

  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = gradOutput->size[0];

  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_zero(state, gradInput);


  real *gradOutput_data, *s_data, *gradInput_data;
  long *indices_data;

  gradOutput_data = THCudaTensor_data(state, gradOutput);
  s_data = THCudaTensor_data(state, s);
  indices_data = THCudaLongTensor_data(state, indices);
  gradInput_data = THCudaTensor_data(state, gradInput)

  cudaStream_t stream = THCState_getCurrentStream(state);
    //TODO: This does the major job!
  RRSVM_updateGradInput_cuda_laucher(

  );

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
//  int nOutputPlane = input->size[1];

  input = THCudaTensor_newContiguous(state, input);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  indices = THCudaLongTensor_newContiguous(state, indices);

//  THArgCheck(THCudaTensor_isContiguous(gradS), 3, "gradS needs to be contiguous");


  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = input->size[0];

  THCudaTensor_zero(state, gradS);

//  THCudaTensor *input_d_h_w = THCudaTensor_new();
//  THCudaTensor *input_h_w = THCudaTensor_new();

  real *input_data, *output_data, *gradS_data;
  long *indices_data;




  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);
  THCudaLongTensor_free(state, indices);

    }

