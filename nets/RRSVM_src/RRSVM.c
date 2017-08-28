#include <TH/TH.h>


int c_forward(THFloatTensor *input, THFloatTensor *s,
		       THFloatTensor *output){
//Adapted from Convolution:


  // Params:
//  int nInputPlane = weight->size[1];
//  int nOutputPlane = weight->size[0];

  input = THTensor_(newContiguous)(input);
  s = THTensor_(newContiguous)(s);

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];

  long outputWidth  = output -> size[3];
  long outputHeight = output -> size[2];

  // Batch size + input planes
  long batchSize = input->size[0];
   int nPlane = input->size[1]

  // Resize output
  THTensor_(resize4d)(output, batchSize, nPlane, outputHeight, outputWidth);
  THTensor_(zero)(output);

  // Resize temporary columns
  THTensor_(resize2d)(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THTensor_(resize2d)(ones, outputHeight, outputWidth);
    THTensor_(fill)(ones, 1);
  }

  // Helpers
  THTensor *input_n = THTensor_(new)();
  THTensor *output_n = THTensor_(new)();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_n, input, 0, elt);
    THTensor_(select)(output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (bias) {
      THBlas_(gemm)(
        't', 'n',
        n_, m_, k_,
        1,
        THTensor_(data)(ones), k_,
        THTensor_(data)(bias), k_,
        0,
        THTensor_(data)(output_n), n_
      );
    } else {
      THTensor_(zero)(output_n);
    }

    // Extract columns:
    THNN_(im2col)(
      THTensor_(data)(input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      THTensor_(data)(columns)
    );

    // M,N,K are dims of matrix A and B
    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane*kH*kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THBlas_(gemm)(
      'n', 'n',
      n, m, k,
      1,
      THTensor_(data)(columns), n,
      THTensor_(data)(weight), k,
      1,
      THTensor_(data)(output_n), n
    );
  }

  // Free
  THTensor_(free)(input_n);
  THTensor_(free)(output_n);

  THTensor_(free)(input);
  THTensor_(free)(weight);
  if (bias) THTensor_(free)(bias);

}

int c_backward_grad_input(THFloatTensor *grad_output, THFloatTensor *grad_input, THFloatTensor *input, THFloatTensor *s){


}

int c_backward_grad_params(THFloatTensor *grad_output, THFloatTensor *grad_s, THFloatTensor *input, THFloatTensor *s){


}

