#include <TH/TH.h>
#define real float

void RRSVM_updateOutput(THTensor *input, THTensor *output, THTensor *s, THLongTensor *indices, THTensor *columns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH){

  int nInputPlane = s->size[0];
  int nOutputPlane = s->size[0];

//  TODO: make contiguous outside
  input = THFloatTensor_newContiguous(input);
  s = THFloatTensor_newContiguous(s);

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = input->size[0];

  THFloatTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);
  THFloatTensor_zero(output);

  THFloatTensor_resize2d(columns, 1*kW*kH, outputHeight*outputWidth);

  THLongTensor_resize5d(indices, batchSize, nOutputPlane, outputHeight, outputWidth, kH * kW);
  THLongTensor_zero(indices);

  //Turn s from [D, kH, kW] to [D, kH * kW]
  THFloatTensor_resize2d(s, nOutputPlane, kH * kW);


  THFloatTensor *input_d_h_w = THFloatTensor_new();
  THFloatTensor *output_d_h_w = THFloatTensor_new();
  THFloatTensor *input_h_w = THFloatTensor_new();
  THFloatTensor *output_h_w = THFloatTensor_new();
  THFloatTensor *s_h_w_1d = THFloatTensor_new();
  THFloatTensor *column_1d = THFloatTensor_new();

  THLongTensor *sorted_index_1d = THLongTensor_new();
  THFloatTensor *sorted_input_1d = THFloatTensor_new();

   for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THFloatTensor_select(input_d_h_w, input, 0, elt);
    THFloatTensor_select(output_d_h_w, output, 0, elt);


        for (int chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor_select(input_h_w, input_d_h_w, 0, chl);
            THFloatTensor_select(output_h_w, output_d_h_w, 0, chl);
            THFloatTensor_select(s_h_w_1d, s, 0, chl);

            THFloatTensor_resize3d(input_h_w, 1, nInputPlane*kW*kH, outputHeight*outputWidth);

            THNN_im2col(THFloatTensor_data(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(columns));
            long i, j, index, inner_product;
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                     index = i * outputWidth + j;
                     THTensor_(select)(column_1d, columns, 1, index);
                     THTensor_(sort)(sorted_input_1d, sorted_index_1d, column_1d, 0, 1)
                     for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        output[elt][chl][i][j]  += s_h_w_1d[inner_product] * sorted_input_1d[inner_product]
                        indices[elt][chl][i][j][inner_product] = sorted_input_1d[inner_product]
                     }
                }
            }

    }

  }
  // Resize back
  THTensor_(resize3d)(s, nOutputPlane, kH, kW)

  // Free
  THTensor_(free)(input_d_h_w);
  THTensor_(free)(output_d_h_w);
  THTensor_(free)(input_h_w);
  THTensor_(free)(output_h_w);
  THTensor_(free)(s_h_w_1d);
  THTensor_(free)(column_1d);

  THTensor_(free)(sorted_index_1d);
  THTensor_(free)(sorted_input_1d);

  THTensor_(free)(input);
  THTensor_(free)(s);
  THTensor_(free)(columns);
  //output and indexes are useful, so not free

}

//TODO: check the variable names
void RRSVM_updateGradInput(THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradInput, THFloatTensor *s, THLongTensor *indices, THFloatTensor * gradColumns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH ){

      int nInputPlane = s->size[0]
  int nOutputPlane = s->size[0]

  input = THTensor_(newContiguous)(input);
  s = THTensor_(newContiguous)(s);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = input->size[0];

  THTensor_(resize4d)(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  THTensor_(resize2d)(gradColumns, 1*kW*kH, outputHeight*outputWidth);
  THTensor_(zero)(gradColumns)

  //Turn s from [D, kH, kW] to [D, kH * kW]
  THTensor_(resize2d)(s, nOutputPlane, kH * kW)


  THTensor *gradinput_d_h_w = THTensor_(new)();
  THTensor *gradoutput_d_h_w = THTensor_(new)();
  THTensor *gradinput_h_w = THTensor_(new)();
  THTensor *gradoutput_h_w = THTensor_(new)();
  THTensor *s_h_w_1d = THTensor_(new)();
  THTensor *column_1d = THTensor_(new)();

  THTensor *sorted_index_1d = THTensor_(new)();
  THTensor *sorted_input_1d = THTensor_(new)();

   for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(graddinput_d_h_w, gradInput, 0, elt);
    THTensor_(select)(gradoutput_d_h_w, gradOutput, 0, elt);


        for (int chl = 0; chl < nInputPlane; chl ++){
            THTensor_(select)(gradinput_h_w, gradinput_d_h_w, 0, chl);
            THTensor_(select)(gradoutput_h_w, gradoutput_d_h_w, 0, chl);
            THTensor_(select)(s_h_w_1d, s, 0, chl);

            THTensor_(resize3d)(input_h_w, nInputPlane*kW*kH, outputHeight*outputWidth);

            THNN_(im2col)(THTensor_(data)(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THTensor_(data)(columns)
            long i, j, index, inner_product
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                     index = i * outputWidth + j
                     THTensor_(select)(column_1d, gradColumns, 1, index);
                     sorted_index_1d = index[elt][chl][i][j]

                     THTensor_(sort)(sorted_input_1d, sorted_index_1d, column_1d, 0, 1)
                     for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        column_1d[inner_product] += s_h_w_1d[sorted_index_1d[inner_product]] * gradoutput_h_w[i][j]
                     }
                }
            }
        THNN_(col2im)(THTensor_(data)(gradColumns), 1, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THTensor_(data)(input_h_w));
    }

  }
  // Resize back
  THTensor_(resize3d)(s, nOutputPlane, kH, kW)

  // Free
  THTensor_(free)(input_d_h_w);
  THTensor_(free)(output_d_h_w);
  THTensor_(free)(input_h_w);
  THTensor_(free)(output_h_w);
  THTensor_(free)(s_h_w_1d);
  THTensor_(free)(column_1d);

  THTensor_(free)(sorted_index_1d);
  THTensor_(free)(sorted_input_1d);

  THTensor_(free)(input);
  THTensor_(free)(s);
  THTensor_(free)(columns);
  //output and indexes are useful, so not free

    }

void RRSVM_accGradParameters(THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradS, THLongTensor * indices. THFloatTensor *columns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH){

          int nInputPlane = s->size[0]
  int nOutputPlane = s->size[0]

  input = THTensor_(newContiguous)(input);
  s = THTensor_(newContiguous)(s);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  THArgCheck(THTensor_(isContiguous)(gradS), 3, "gradS needs to be contiguous");


  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = input->size[0];

//  THTensor_(resize4d)(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  THTensor_(resize2d)(columns, 1*kW*kH, outputHeight*outputWidth);

  //Turn s from [D, kH, kW] to [D, kH * kW]
  THTensor_(resize2d)(s, nOutputPlane, kH * kW)


  THTensor *input_d_h_w = THTensor_(new)();
  THTensor *gradoutput_d_h_w = THTensor_(new)();
  THTensor *input_h_w = THTensor_(new)();
  THTensor *gradoutput_h_w = THTensor_(new)();
  THTensor *s_h_w_1d = THTensor_(new)();
  THTensor *column_1d = THTensor_(new)();

  THTensor *sorted_index_1d = THTensor_(new)();
  THTensor *sorted_input_1d = THTensor_(new)();

   for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THTensor_(select)(input_d_h_w, gradInput, 0, elt);
    THTensor_(select)(gradoutput_d_h_w, gradOutput, 0, elt);


        for (int chl = 0; chl < nInputPlane; chl ++){
            THTensor_(select)(input_h_w, input_d_h_w, 0, chl);
            THTensor_(select)(gradoutput_h_w, gradoutput_d_h_w, 0, chl);
            THTensor_(select)(s_h_w_1d, s, 0, chl);

            THTensor_(resize3d)(input_h_w, nInputPlane*kW*kH, outputHeight*outputWidth);

            THNN_(im2col)(THTensor_(data)(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THTensor_(data)(columns)
            long i, j, index, inner_product
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                     index = i * outputWidth + j
                     THTensor_(select)(column_1d, columns, 1, index);
                     sorted_index_1d = index[elt][chl][i][j]

                     THTensor_(sort)(sorted_input_1d, sorted_index_1d, column_1d, 0, 1)
                     for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        gradS[chl][inner_product] += sorted_input_1d[inner_product]

                     }
                }
            }
    }

  }
  // Resize back
  THTensor_(resize3d)(s, nOutputPlane, kH, kW)

  // Free
  THTensor_(free)(input_d_h_w);
  THTensor_(free)(gradoutput_d_h_w);
  THTensor_(free)(input_h_w);
  THTensor_(free)(gradoutput_h_w);
  THTensor_(free)(s_h_w_1d);
  THTensor_(free)(column_1d);

  THTensor_(free)(sorted_index_1d);
  THTensor_(free)(sorted_input_1d);

  THTensor_(free)(input);
  THTensor_(free)(gradOutput);

//  THTensor_(free)(columns);
  //output and indexes are useful, so not free

    }

