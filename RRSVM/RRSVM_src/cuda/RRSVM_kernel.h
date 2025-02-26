#ifdef __cplusplus
extern "C" {
#endif

void im2col(cudaStream_t stream, const float* data_im, const int channels,
            const int height, const int width,
            const int ksize_h, const int ksize_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, float* data_col);

void col2im(cudaStream_t stream, const float* data_col, const int channels,
            const int height, const int width,
            const int output_height, const int output_width,
            const int patch_h, const int patch_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, float* data_im);





void fill_output(const float * sorted_input_1d_data, const float *s_data, int i, int j, int inner_product,
                 int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data, cudaStream_t stream);
void fill_indices(const long *sorted_index_1d_data, int i, int j, int inner_product,
                  int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long* indices_data, cudaStream_t stream);

void fill_gradInput(const float * gradOutput_data, const float * s_data, const long * indices_data, int i, int j, int inner_product,
                    int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *gradInputColumns_data, cudaStream_t stream);

void fill_gradS(const float * gradOutput_data, const float * column_data, const long * indices_data, int i, int j, int inner_product,
                int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *gradS_data, cudaStream_t stream);


//faster 1.0 interfaces
void fill_output_2d(const float * sorted_input_2d_data, const float *s_data,
                 int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data, cudaStream_t stream);
void fill_indices_2d(const long *sorted_index_2d_data,
                  int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long* indices_data, cudaStream_t stream);

void fill_gradInput_2d(const float* gradOutput_data, const float * s_data, const long * indices_data,
                       int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float * gradInputColumns_data,cudaStream_t stream);

void fill_gradS_2d(const float * gradOutput_data, const float * column_data, const long * indices_data,
        int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth,  float * gradS_data, cudaStream_t stream);

//faster 2.0 interfaces:
void im3col(cudaStream_t stream, const float* data_im, const int channels,
            const int height, const int width,
            const int ksize_h, const int ksize_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, float* data_col);



//void fill_output_3d(sorted_input_2d_data, sorted_index_2d_data, s_data, elt, kH,  kW,  nInputPlane,  outputHeight,  outputWidth,  output_data, indices_data, stream);

void fill_output_3d(const float * sorted_input_2d_data, const long*sorted_index_2d_data,  const float *s_data,
                    int elt, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data, long *indices_data,
                    cudaStream_t stream);

void fill_gradInput_3d(const float* gradOutput_data, const float * s_data, const long * indices_data,
                       int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float * gradInputColumns_data,cudaStream_t stream);


void fill_gradS_3d(const float * gradOutput_data, const float * column_data, const long * indices_data,
                   int elt, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth,  float * gradS_data, cudaStream_t stream);
//void  RRSVM_updateOutput_cuda_laucher(const float *input, const float *s, float *output, long *indices,
//    const long batchSize, const long nInputPlane, const long inputHeight, const long inputWidth,
//    const long outputHeight, const outputWidth,
//    int kW, int kH,
//    int dW, int dH,
//    int padW, int padH,
//    int dilationW, int dilationH, cudaStream_t stream);
//
//void RRSVM_updateGradInput_cuda_laucher(
//
//  );
//void  RRSVM_accGradParameters_cuda_laucher(
//
//  );


#ifdef __cplusplus
}
#endif