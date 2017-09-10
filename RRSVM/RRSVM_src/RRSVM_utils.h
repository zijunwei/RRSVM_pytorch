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

//                        output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j] +=
//                         s_data[chl* kH *kW + inner_product] * sorted_input_1d_data[inner_product];
//                        indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i * outputWidth*kH*kW + j*kH*kW + inner_product] =
//                         sorted_index_1d_data[inner_product];

void fill_output(const float * sorted_input_1d_data, const float *s_data, int i, int j, int inner_product,
int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data);
void fill_indices(const long *sorted_index_1d_data, int i, int j, int inner_product,
int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long* indices_data);

#ifdef __cplusplus
}
#endif