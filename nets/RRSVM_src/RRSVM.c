#include <TH/TH.h>
#define real float

static void THNN_Floatim2col(const real* data_im, const int channels,
      const int height, const int width, const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      real* data_col) {
  const int height_col = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_col = (width + 2 * pad_w -
                         (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
          data_im[(c_im * height + h_im) * width + w_im] : 0;
      }
    }
  }
}

static void THNN_Floatcol2im(const real* data_col, const int channels,
      const int height, const int width,
      const int output_height, const int output_width,
      const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      real* data_im) {
  memset(data_im, 0, sizeof(real) * height * width * channels);
  const int height_col = output_height;
  const int width_col = output_width;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
            data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}


void RRSVM_updateOutput(THFloatTensor *input, THFloatTensor *output, THFloatTensor *s, THLongTensor *indices, THFloatTensor *columns, int kW, int kH,
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

  real *input_data, *output_data, *s_1d_data;
  long *indices_data;

  input_data = THFloatTensor_data(input);
  output_data = THFloatTensor_data(output);
  indices_data = THLongTensor_data(indices);
  s_1d_data = THFloatTensor_data(s_h_w_1d);


   for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THFloatTensor_select(input_d_h_w, input, 0, elt);
    THFloatTensor_select(output_d_h_w, output, 0, elt);


        for (int chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor_select(input_h_w, input_d_h_w, 0, chl);
            THFloatTensor_select(output_h_w, output_d_h_w, 0, chl);
            THFloatTensor_select(s_h_w_1d, s, 0, chl);

            THFloatTensor_resize3d(input_h_w, 1, nInputPlane*kW*kH, outputHeight*outputWidth);

            THNN_Floatim2col(THFloatTensor_data(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(columns));
//            THNN_(im2col)(THTensor_(data)(input_h_w),
//      nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
//      kH, kW, padH, padW, dH, dW,
//      dilationH, dilationW,
//      THTensor_(data)(columns)
//    );

            long i, j, index, inner_product;
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                     index = i * outputWidth + j;
                     THFloatTensor_select(column_1d, columns, 1, index);

                     THFloatTensor_sort(sorted_input_1d, sorted_index_1d, column_1d, 0, 1);

                     real *sorted_input_1d_data = THFloatTensor_data(sorted_input_1d);
                     long *sorted_index_1d_data = THLongTensor_data(sorted_index_1d);


                     for (inner_product = 0; inner_product < kW * kH; inner_product ++){
//                        output_data[elt][chl][i][j]  += s_1d_data[inner_product] * sorted_input_1d_data[inner_product]
//                        indices_data[elt][chl][i][j][inner_product] = sorted_index_1d_data[inner_product]
                        output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j] += s_1d_data[inner_product] * sorted_input_1d_data[inner_product];
                        indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i * outputWidth*kH*kW + j*kH*kW + inner_product] = sorted_index_1d_data[inner_product];
                     }
                }
            }

    }

  }
  // Resize back
  THFloatTensor_resize3d(s, nOutputPlane, kH, kW);

  // Free
  THFloatTensor_free(input_d_h_w);
  THFloatTensor_free(output_d_h_w);
  THFloatTensor_free(input_h_w);
  THFloatTensor_free(output_h_w);
  THFloatTensor_free(s_h_w_1d);
  THFloatTensor_free(column_1d);

  THLongTensor_free(sorted_index_1d);
  THFloatTensor_free(sorted_input_1d);

  THFloatTensor_free(input);
  THFloatTensor_free(s);
  THFloatTensor_free(columns);
  //output and indexes are useful, so not free

}

void RRSVM_updateGradInput(THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradInput, THFloatTensor *s, THLongTensor *indices, THFloatTensor * gradColumns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH ){

      int nInputPlane = s->size[0];
  int nOutputPlane = s->size[0];

  input = THFloatTensor_newContiguous(input);
  s = THFloatTensor_newContiguous(s);
  gradOutput = THFloatTensor_newContiguous(gradOutput);

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = input->size[0];

  THFloatTensor_resize4d(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  THFloatTensor_resize2d(gradColumns, 1*kW*kH, outputHeight*outputWidth);
  THFloatTensor_zero(gradColumns);

  //Turn s from [D, kH, kW] to [D, kH * kW]
  THFloatTensor_resize2d(s, nOutputPlane, kH * kW);


  THFloatTensor *gradinput_d_h_w = THFloatTensor_new();
  THFloatTensor *gradoutput_d_h_w = THFloatTensor_new();
  THFloatTensor *gradinput_h_w = THFloatTensor_new();
  THFloatTensor *gradoutput_h_w = THFloatTensor_new();
  THFloatTensor *s_h_w_1d = THFloatTensor_new();
  THFloatTensor *column_1d = THFloatTensor_new();

  THLongTensor *sorted_index_1d = THLongTensor_new();
  THFloatTensor *sorted_input_1d = THFloatTensor_new();

  real *input_data, *output_data, *s_1d_data;
  long *indices_data;

  input_data = THFloatTensor_data(gradInput);
  output_data = THFloatTensor_data(gradOutput);
  indices_data = THLongTensor_data(indices);
  s_1d_data = THFloatTensor_data(s_h_w_1d);

   for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THFloatTensor_select(gradinput_d_h_w, gradInput, 0, elt);
    THFloatTensor_select(gradoutput_d_h_w, gradOutput, 0, elt);


        for (int chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor_select(gradinput_h_w, gradinput_d_h_w, 0, chl);
            THFloatTensor_select(gradoutput_h_w, gradoutput_d_h_w, 0, chl);
            THFloatTensor_select(s_h_w_1d, s, 0, chl);

            THFloatTensor_resize3d(gradinput_h_w, 1, nInputPlane*kW*kH, outputHeight*outputWidth);

            THNN_Floatim2col(THFloatTensor_data(gradinput_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(gradColumns));
            long i, j, index, inner_product;
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                     index = i * outputWidth + j;
                     THFloatTensor_select(column_1d, gradColumns, 1, index);
                     real * column_1d_data = THFloatTensor_data(column_1d);
                     long *sorted_index_1d_data = &indices_data[elt*nInputPlane*outputHeight*outputWidth + chl* outputHeight*outputWidth + i*outputWidth + j];

                     for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        column_1d_data[inner_product] += s_1d_data[sorted_index_1d_data[inner_product]] * output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i*outputWidth + j];
                     }
                }
            }
        THNN_Floatcol2im(THFloatTensor_data(gradColumns), 1, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(gradinput_h_w));
    }

  }
  // Resize back
  THFloatTensor_resize3d(s, nOutputPlane, kH, kW);

  // Free
  THFloatTensor_free(gradinput_d_h_w);
  THFloatTensor_free(gradoutput_d_h_w);
  THFloatTensor_free(gradinput_h_w);
  THFloatTensor_free(gradoutput_h_w);
  THFloatTensor_free(s_h_w_1d);
  THFloatTensor_free(column_1d);

  THLongTensor_free(sorted_index_1d);
  THFloatTensor_free(sorted_input_1d);

  THFloatTensor_free(input);
  THFloatTensor_free(s);
  THFloatTensor_free(column_1d);
  //output and indexes are useful, so not free

    }


void RRSVM_accGradParameters(THFloatTensor *input, THFloatTensor *gradOutput, THFloatTensor *gradS, THLongTensor * indices, THFloatTensor *columns, int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH){

          int nInputPlane = gradS->size[0];
  int nOutputPlane = gradS->size[0];

  input = THFloatTensor_newContiguous(input);
  gradS = THFloatTensor_newContiguous(gradS);
  gradOutput = THFloatTensor_newContiguous(gradOutput);

  THArgCheck(THFloatTensor_isContiguous(gradS), 3, "gradS needs to be contiguous");


  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  long batchSize = input->size[0];

//  THTensor_(resize4d)(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  THFloatTensor_resize2d(columns, 1*kW*kH, outputHeight*outputWidth);

  //Turn s from [D, kH, kW] to [D, kH * kW]
  THFloatTensor_resize2d(gradS, nOutputPlane, kH * kW);


  THFloatTensor *input_d_h_w = THFloatTensor_new();
  THFloatTensor *gradoutput_d_h_w = THFloatTensor_new();
  THFloatTensor *input_h_w = THFloatTensor_new();
  THFloatTensor *gradoutput_h_w = THFloatTensor_new();
  THFloatTensor *s_h_w_1d = THFloatTensor_new();
  THFloatTensor *column_1d = THFloatTensor_new();

  THLongTensor *sorted_index_1d = THLongTensor_new();
  THFloatTensor *sorted_input_1d = THFloatTensor_new();

   real *input_data, *output_data, *s_1d_data, *gradS_data;
  long *indices_data;

  input_data = THFloatTensor_data(input);
  output_data = THFloatTensor_data(gradOutput);
  indices_data = THLongTensor_data(indices);
  s_1d_data = THFloatTensor_data(s_h_w_1d);
  gradS_data = THFloatTensor_data(gradS);

   for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THFloatTensor_select(input_d_h_w, input, 0, elt);
    THFloatTensor_select(gradoutput_d_h_w, gradOutput, 0, elt);


        for (int chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor_select(input_h_w, input_d_h_w, 0, chl);
            THFloatTensor_select(gradoutput_h_w, gradoutput_d_h_w, 0, chl);
            THFloatTensor_select(s_h_w_1d, gradS, 0, chl);

            THFloatTensor_resize3d(input_h_w, 1, nInputPlane*kW*kH, outputHeight*outputWidth);

            THNN_Floatim2col(THFloatTensor_data(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(columns));
            long i, j, index, inner_product;
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                     index = i * outputWidth + j;
                     THFloatTensor_select(column_1d, columns, 1, index);
//                     sorted_index_1d = indices_data[elt][chl][i][j]
                     long *sorted_index_1d_data = &indices_data[elt*nInputPlane*outputHeight*outputWidth + chl* outputHeight*outputWidth + i*outputWidth + j];
                     real *sorted_input_1d_data = THFloatTensor_data(column_1d);
                     for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        gradS_data[chl*(kW*kH) + inner_product] += sorted_input_1d_data[sorted_index_1d_data[inner_product]];

                     }
                }
            }
    }

  }
  // Resize back
  THFloatTensor_resize3d(gradS, nOutputPlane, kH, kW);

  // Free
  THFloatTensor_free(input_d_h_w);
  THFloatTensor_free(gradoutput_d_h_w);
  THFloatTensor_free(input_h_w);
  THFloatTensor_free(gradoutput_h_w);
  THFloatTensor_free(s_h_w_1d);
  THFloatTensor_free(column_1d);

  THLongTensor_free(sorted_index_1d);
  THFloatTensor_free(sorted_input_1d);

  THFloatTensor_free(input);
  THFloatTensor_free(gradOutput);

//  THTensor_(free)(columns);
  //output and indexes are useful, so not free
    }

