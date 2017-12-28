#include <TH/TH.h>
#include <stdio.h>
#include <math.h>
#define real float

static void THNN_Floatim2col(const real* data_im, const int channels,
                             const long height, const long width, const int kernel_h, const int kernel_w,
                             const int pad_h, const int pad_w,
                             const int stride_h, const int stride_w,
                             const int dilation_h, const int dilation_w,
                             real* data_col) {
    const long height_col = (height + 2 * pad_h -
                             (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const long width_col = (width + 2 * pad_w -
                            (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channels_col = channels * kernel_h * kernel_w;
    int c_col, h_col, w_col;
    for (c_col = 0; c_col < channels_col; ++c_col) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / kernel_h / kernel_w;
        for (h_col = 0; h_col < height_col; ++h_col) {
            for (w_col = 0; w_col < width_col; ++w_col) {
                int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                data_col[(c_col * height_col + h_col) * width_col + w_col] =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                        data_im[(c_im * height + h_im) * width + w_im] : 0; // This is padding with zero
            }
        }
    }
}

static void THNN_Floatcol2im(const real* data_col, const int channels,
                             const long height, const long width,
                             const long output_height, const long output_width,
                             const int kernel_h, const int kernel_w,
                             const int pad_h, const int pad_w,
                             const int stride_h, const int stride_w,
                             const int dilation_h, const int dilation_w,
                             real* data_im) {
    memset(data_im, 0, sizeof(real) * height * width * channels);
    const long height_col = output_height;
    const long width_col = output_width;
    const int channels_col = channels * kernel_h * kernel_w;
    int c_col, h_col, w_col;
    for (c_col = 0; c_col < channels_col; ++c_col) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / kernel_h / kernel_w;
        for (h_col = 0; h_col < height_col; ++h_col) {
            for (w_col = 0; w_col < width_col; ++w_col) {
                int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
                int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
                    data_im[(c_im * height + h_im) * width + w_im] +=
                            data_col[(c_col * height_col + h_col) * width_col + w_col];
            }
        }
    }
}


void RRSVM_updateOutput(THFloatTensor *input, THFloatTensor *s, THFloatTensor *softmax_s, THFloatTensor *output, THLongTensor *indices,
                        int kW, int kH,
                        int dW, int dH,
                        int padW, int padH,
                        int dilationW, int dilationH){

    long nInputPlane = s->size[0];
    long nOutputPlane = s->size[0];

    input = THFloatTensor_newContiguous(input);
    s = THFloatTensor_newContiguous(s);

    long inputWidth   = input->size[3];
    long inputHeight  = input->size[2];
    long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long batchSize = input->size[0];

    THFloatTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);
    THFloatTensor_zero(output);

    THLongTensor_resize5d(indices, batchSize, nOutputPlane, outputHeight, outputWidth, kH * kW);
    THLongTensor_zero(indices);

    THFloatTensor_resize2d(softmax_s, nOutputPlane, kW*kH);
    THFloatTensor_zero(softmax_s);


    real *output_data, *s_data, *softmax_s_data;
    long *indices_data;

//  input_data = THFloatTensor_data(input);
    output_data = THFloatTensor_data(output);
    indices_data = THLongTensor_data(indices);
    s_data = THFloatTensor_data(s);
    softmax_s_data = THFloatTensor_data(softmax_s);


    int chl;
#pragma omp parallel for private(chl)
    for (chl = 0; chl < nInputPlane; chl ++){
        float inputMax = -INFINITY;
        float sum = 0;
        int inner = 0;
        for (inner=0; inner < kW*kH; ++inner){
            if (s_data[chl* kH *kW + inner] >= inputMax) inputMax = s_data[chl* kH *kW + inner];
        }
        for (inner=0; inner < kW*kH; ++inner)
        {
            float z = exp(s_data[chl* kH *kW + inner] - inputMax);
            softmax_s_data[chl* kH *kW + inner] = z;
            sum += z;
        }
        for (inner=0; inner < kW*kH; ++inner)
        {
            softmax_s_data[chl* kH *kW + inner] *= 1/sum;
        }
    }

    long elt;
#pragma omp parallel for private(elt)
    for (elt = 0; elt < batchSize; elt ++) {
        THFloatTensor *input_d_h_w = THFloatTensor_new();

        THFloatTensor_select(input_d_h_w, input, 0, elt);

        for ( chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor *input_h_w = THFloatTensor_new();

            THFloatTensor_select(input_h_w, input_d_h_w, 0, chl);

//            THFloatTensor_resize3d(input_h_w, 1, inputHeight, inputWidth);
            THFloatTensor * columns = THFloatTensor_newWithSize2d(1*kW*kH, outputHeight*outputWidth);
            THFloatTensor_zero(columns);
            THNN_Floatim2col(THFloatTensor_data(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(columns));

            long i, j, index, inner_product;


            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                    index = i * outputWidth + j;
                    //TODO: select or newselect?
                    THFloatTensor *column_1d = THFloatTensor_newSelect(columns, 1, index);

                    THLongTensor *sorted_index_1d = THLongTensor_new();
                    THFloatTensor *sorted_input_1d = THFloatTensor_new();

                    THFloatTensor_sort(sorted_input_1d, sorted_index_1d, column_1d, 0, 1);


                    real *sorted_input_1d_data = THFloatTensor_data(sorted_input_1d);
                    long *sorted_index_1d_data = THLongTensor_data(sorted_index_1d);

                    for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j] += softmax_s_data[chl* kH *kW + inner_product] * sorted_input_1d_data[inner_product];
                        indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i * outputWidth*kH*kW + j*kH*kW + inner_product] = sorted_index_1d_data[inner_product];
                    }
                    THLongTensor_free(sorted_index_1d);
                    THFloatTensor_free(sorted_input_1d);
                    THFloatTensor_free(column_1d);
                }
            }
            THFloatTensor_free(columns);
            THFloatTensor_free(input_h_w);
        }
        THFloatTensor_free(input_d_h_w);
    }

    THFloatTensor_free(input);
    THFloatTensor_free(s);

}


void RRSVM_updateGradInput(THFloatTensor *s, THLongTensor *indices, THFloatTensor *gradOutput, THFloatTensor *gradInput,
                           int inputWidth, int inputHeight,
                           int kW, int kH,
                           int dW, int dH,
                           int padW, int padH,
                           int dilationW, int dilationH ){

    long nInputPlane = s->size[0];

    s = THFloatTensor_newContiguous(s);
    gradOutput = THFloatTensor_newContiguous(gradOutput);
    indices = THLongTensor_newContiguous(indices);

    long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long batchSize = gradOutput->size[0];

    THFloatTensor_resize4d(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
    THFloatTensor_zero(gradInput);


//  THFloatTensor *column_1d = THFloatTensor_new();

    real *gradOutput_data, *s_data;
    long *indices_data;

    gradOutput_data = THFloatTensor_data(gradOutput);
    s_data = THFloatTensor_data(s);
    indices_data = THLongTensor_data(indices);
    long elt;
#pragma omp parallel for private(elt)
    for ( elt = 0; elt < batchSize; elt ++) {
        // Matrix mulitply per output:
        THFloatTensor *gradinput_d_h_w = THFloatTensor_new();

        THFloatTensor_select(gradinput_d_h_w, gradInput, 0, elt);

        int chl;
        for (chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor *gradinput_h_w = THFloatTensor_new();

            THFloatTensor_select(gradinput_h_w, gradinput_d_h_w, 0, chl);
//            THFloatTensor_resize3d(gradinput_h_w, 1, inputHeight, inputWidth);
            THFloatTensor * gradInputColumns = THFloatTensor_newWithSize2d(1*kW*kH, outputHeight*outputWidth);
            THFloatTensor_zero(gradInputColumns);
            real * gradInputColumns_data = THFloatTensor_data(gradInputColumns);
            long i, j, index, inner_product;
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                    index = i * outputWidth + j;
                    for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        long idx = indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i*outputWidth*kH*kW + j*kW*kH + inner_product];
                        //TODO: HERE is problematic
                        gradInputColumns_data[idx*(outputWidth*outputHeight)+index] += s_data[chl*kW*kH+inner_product] * gradOutput_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i*outputWidth + j];
                    }
                }
            }

            THNN_Floatcol2im(gradInputColumns_data, 1, inputHeight, inputWidth, outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(gradinput_h_w));
            THFloatTensor_free(gradInputColumns);
            THFloatTensor_free(gradinput_h_w);

        }
        THFloatTensor_free(gradinput_d_h_w);

    }
//  THFloatTensor_resize3d(s, nOutputPlane, kH, kW);


    THFloatTensor_free(s);
    THLongTensor_free(indices);
    THFloatTensor_free(gradOutput);
    //output and indexes are useful, so not free

}

void RRSVM_accGradParameters(THFloatTensor *input, THFloatTensor *softmax_s, THLongTensor * indices, THFloatTensor *gradOutput, THFloatTensor *gradS, THFloatTensor *gradSoftmaxS,
                             int kW, int kH,
                             int dW, int dH,
                             int padW, int padH,
                             int dilationW, int dilationH){

    long nInputPlane = input->size[1];
//  int nOutputPlane = input->size[1];

    input = THFloatTensor_newContiguous(input);
    gradOutput = THFloatTensor_newContiguous(gradOutput);
    indices = THLongTensor_newContiguous(indices);
    softmax_s = THFloatTensor_newContiguous(softmax_s);

    THArgCheck(THFloatTensor_isContiguous(gradS), 3, "gradS needs to be contiguous");
    THArgCheck(THFloatTensor_isContiguous(gradSoftmaxS), 3, "gradSoftmax needs to be contiguous");


    long inputWidth   = input->size[3];
    long inputHeight  = input->size[2];
    long outputWidth  = (inputWidth + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1;
    long outputHeight = (inputHeight + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1;

    long batchSize = input->size[0];

    THFloatTensor_zero(gradS);
    THFloatTensor_zero(gradSoftmaxS);

//  THFloatTensor *input_d_h_w = THFloatTensor_new();
//  THFloatTensor *input_h_w = THFloatTensor_new();

    real /**input_data,*/ *output_data, *gradS_data, *softmax_s_data, *gradSoftmaxS_data;
    long *indices_data;

//  input_data = THFloatTensor_data(input);
    output_data = THFloatTensor_data(gradOutput);
    indices_data = THLongTensor_data(indices);
    gradS_data = THFloatTensor_data(gradS);
    gradSoftmaxS_data = THFloatTensor_data(gradSoftmaxS);
    softmax_s_data = THFloatTensor_data(softmax_s);
    long elt;
    int chl;

#pragma omp parallel for private(elt)
    for ( elt = 0; elt < batchSize; elt ++) {
        // Matrix mulitply per output:
        THFloatTensor *input_d_h_w = THFloatTensor_new();
        THFloatTensor_select(input_d_h_w, input, 0, elt);
//    THFloatTensor_select(gradoutput_d_h_w, gradOutput, 0, elt);
        for ( chl = 0; chl < nInputPlane; chl ++){
            THFloatTensor *input_h_w = THFloatTensor_new();
            THFloatTensor_select(input_h_w, input_d_h_w, 0, chl);
//            THFloatTensor_select(gradoutput_h_w, gradoutput_d_h_w, 0, chl);
//            THFloatTensor_select(s_h_w_1d, gradS, 0, chl);
//            THFloatTensor_resize3d(input_h_w, 1, inputHeight, inputWidth);
            THFloatTensor * columns = THFloatTensor_newWithSize2d(1*kW*kH, outputHeight*outputWidth);

            THNN_Floatim2col(THFloatTensor_data(input_h_w), 1, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, THFloatTensor_data(columns));
            real * columns_data = THFloatTensor_data(columns);

            long i, j, index, inner_product;
            for (i = 0; i < outputHeight; i ++)
            {
                for(j = 0; j < outputWidth; j ++)
                {
                    index = i * outputWidth + j;
                    for (inner_product = 0; inner_product < kW * kH; inner_product ++){
                        long idx = indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i*outputWidth*kH*kW + j*kW*kH + inner_product];
                        gradS_data[chl*(kW*kH) + inner_product] += columns_data[idx*outputHeight*outputHeight+index] * output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i*outputWidth + j];
                    }
                }
            }
            THFloatTensor_free(columns);
            THFloatTensor_free(input_h_w);

        }
        THFloatTensor_free(input_d_h_w);


    }

#pragma omp parallel for private(chl)
    for (chl = 0; chl < nInputPlane; chl ++){
        float sum = 0;
        int d;
        for (d = 0; d < kH*kW; d++){
            sum += gradS_data[chl*kW*kH + d] * softmax_s_data[chl*kW*kH + d];

        }
        for (d = 0; d < kH*kW; d++){
            gradSoftmaxS_data[chl*kW*kH + d] = softmax_s_data[chl*kW*kH + d] * (gradS_data[chl*kW*kH + d] - sum);
        }
    }

    THFloatTensor_free(input);
    THFloatTensor_free(gradOutput);
    THLongTensor_free(indices);
    THFloatTensor_free(softmax_s);
}

