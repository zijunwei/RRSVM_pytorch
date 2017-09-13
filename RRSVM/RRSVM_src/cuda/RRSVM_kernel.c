#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <limits.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "RRSVM_kernel.h"

#define Dtype float
#define Acctype float
#define In float
#define Out float

const int CUDA_NUM_THREADS = 1024;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int (i) = blockIdx.x * blockDim.x + threadIdx.x; (i) < (n); (i) += blockDim.x * gridDim.x)



inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

//template <typename In, typename Out>
//struct ScalarConvert {
//  static
__host__ __device__ Out to(const In v) { return (Out) v; }
//};


__global__ void im2col_kernel(const int n, const Dtype* data_im,
                              const int height, const int width,
                              const int ksize_h, const int ksize_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              const int height_col, const int width_col,
                              Dtype* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;
        index /= width_col;
        int h_out = index % height_col;
        int channel_in = index / height_col;
        int channel_out = channel_in * ksize_h * ksize_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        data_col += (channel_out * height_col + h_out) * width_col + w_out;
        data_im += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize_h; ++i) {
            for (int j = 0; j < ksize_w; ++j) {
                int h = h_in + i * dilation_h;
                int w = w_in + j * dilation_w;
                *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
                            data_im[i * dilation_h * width + j * dilation_w] : to(0);
                data_col += height_col * width_col;
            }
        }
    }
}

//template <typename Dtype>
void im2col(cudaStream_t stream, const Dtype* data_im, const int channels,
            const int height, const int width,
            const int ksize_h, const int ksize_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, Dtype* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1))
                     / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1))
                    / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    // Launch
    im2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
            num_kernels, data_im, height, width, ksize_h, ksize_w,
                    pad_h, pad_w, stride_h, stride_w,
                    dilation_h, dilation_w,
                    height_col, width_col, data_col
    );
//  THCudaCheck(cudaGetLastError());
}

//template <typename Dtype, typename Acctype>
__global__ void col2im_kernel(const int n, const Dtype* data_col,
                              const int height, const int width, const int channels,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              const int height_col, const int width_col,
                              Dtype* data_im) {
    CUDA_KERNEL_LOOP(index, n) {
//    Acctype val = Acctype(0);
        Acctype val = 0;
        const int w_im = index % width + pad_w;
        const int h_im = (index / width) % height + pad_h;
        const int c_im = index / (width * height);
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
                (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
                (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                          height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = to(val);
//    data_im[index] = ScalarConvert<Acctype, Dtype>::to(val);
//    data_im[index] = val;

    }
}

//template <typename Dtype, typename Acctype>
void col2im(cudaStream_t stream, const Dtype* data_col, const int channels,
            const int height, const int width,
            const int output_height, const int output_width,
            const int patch_h, const int patch_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, Dtype* data_im) {
    int num_kernels = channels * height * width;

    col2im_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
            num_kernels, data_col, height, width, channels,
                    patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
                    dilation_h, dilation_w,
                    output_height, output_width, data_im
    );

    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
//  col2im_kernel<Dtype, Acctype> <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
//      num_kernels, data_col, height, width, channels,
//      patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
//      dilation_h, dilation_w,
//      output_height, output_width, data_im
//  );
//  THCudaCheck(cudaGetLastError());
}

__global__ void fill_output_kernel(const int n, const float * sorted_input_1d_data, const float *s_data, int i, int j, int inner_product,
                                   int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data){

    output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j] +=
            s_data[chl* kH *kW + inner_product] * sorted_input_1d_data[inner_product];
    //DEBUG
//    printf("output_data: %d : %f\n", inner_product,
//           output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j]);
}


void fill_output(const float * sorted_input_1d_data, const float *s_data, int i, int j, int inner_product,
                 int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data, cudaStream_t stream){

    const int kThreadsPerBlock = 1;
    const int output_size = 1;
    cudaError_t err;
    fill_output_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size,
            sorted_input_1d_data, s_data,  i,  j,  inner_product,
                    elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth, output_data
    );
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

}

__global__ void fill_indices_kernel(const int n, const long *sorted_index_1d_data, int i, int j, int inner_product,
                                    int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long* indices_data ){

    indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i * outputWidth*kH*kW + j*kH*kW + inner_product] =
            sorted_index_1d_data[inner_product];

//    printf("Incides_data: %d : %ld\n", inner_product,
//           indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i * outputWidth*kH*kW + j*kH*kW + inner_product]);

}

void fill_indices(const long *sorted_index_1d_data, int i, int j, int inner_product,
                  int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long* indices_data, cudaStream_t stream){

    const int kThreadsPerBlock = 1;
    const int output_size = 1;
    cudaError_t err;
    fill_indices_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size,
            sorted_index_1d_data,  i,  j,  inner_product,
                    elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth, indices_data
    );
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

}

__global__ void fill_gradInput_kernel(const int n, const float * gradOutput_data, const float * s_data, const long * indices_data, int i, int j, int inner_product,
                                      int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *gradInputColumns_data ){


    long idx = indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i*outputWidth*kH*kW + j*kW*kH + inner_product];
    gradInputColumns_data[idx*(outputWidth*outputHeight)+i*outputWidth + j] +=
            s_data[chl*kW*kH+inner_product] * gradOutput_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i*outputWidth + j];

//    printf("GradInput: %ld, gradOutput: %f, s_data: %f\n",
//           idx,   gradOutput_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i*outputWidth + j] ,s_data[chl*kW*kH+inner_product]);
}

void fill_gradInput(const float * gradOutput_data, const float * s_data, const long * indices_data, int i, int j, int inner_product,
                    int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *gradInputColumns_data, cudaStream_t stream){

    const int kThreadsPerBlock = 1;
    const int output_size = 1;
    cudaError_t err;

    fill_gradInput_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size,
            gradOutput_data, s_data, indices_data, i,  j,  inner_product,
            elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth, gradInputColumns_data
    );
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

}

__global__ void fill_gradS_kernel(const int n, const float * gradOutput_data, const float * column_data, const long * indices_data, int i, int j, int inner_product,
                                      int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *gradS_data ){


    long idx = indices_data[elt*nInputPlane*outputHeight*outputWidth*kH*kW + chl*outputHeight*outputWidth*kH*kW + i*outputWidth*kH*kW + j*kW*kH + inner_product];
    gradS_data[chl*(kW*kH) + inner_product] += column_data[idx*outputHeight*outputHeight+ i * outputWidth + j] * gradOutput_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i*outputWidth + j];


}



void fill_gradS(const float * gradOutput_data, const float * column_data, const long * indices_data, int i, int j, int inner_product,
                int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *gradS_data, cudaStream_t stream){

    const int kThreadsPerBlock = 1;
    const int output_size = 1;
    cudaError_t err;
    fill_gradS_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size,
            gradOutput_data, column_data, indices_data, i,  j,  inner_product,
            elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth, gradS_data
    );
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}


//faster version:

__global__ void fill_output2d_kernel(const int n, const float * sorted_input_2d_data, const float *s_data,
                                   int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data){

    CUDA_KERNEL_LOOP(index, n){
        Acctype val = 0;
        for (int i = 0; i < kH*kW; ++i) {
            val += s_data[chl * kH * kW + i] * sorted_input_2d_data[index+ i*outputHeight*outputWidth];
        }
        output_data[elt * nInputPlane * outputHeight * outputWidth + chl * outputHeight * outputWidth + index] = to(val);
    }
    //DEBUG
//    printf("output_data: %d : %f\n", inner_product,
//           output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j]);
}

void fill_output_2d(const float * sorted_input_2d_data, const float *s_data,
                    int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, float *output_data, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    const int output_size = outputHeight * outputWidth;
    cudaError_t err;
    fill_output2d_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size,
            sorted_input_2d_data, s_data,
            elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth, output_data
    );
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }



}

__global__ void fill_indices2d_kernel(const int n, const long * sorted_index_2d_data,
                                     int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long *indices_data){

    CUDA_KERNEL_LOOP(index, n){

        for (int i = 0; i < kH*kW; ++i) {
            indices_data[elt * nInputPlane * outputHeight * outputWidth*kH*kW + chl * outputHeight * outputWidth*kH*kW + index*kH*kW + i] =
                    sorted_index_2d_data[i*outputHeight*outputWidth + index];

        }
    }
    //DEBUG
//    printf("output_data: %d : %f\n", inner_product,
//           output_data[elt*nInputPlane*outputHeight*outputWidth + chl*outputHeight*outputWidth + i * outputWidth + j]);
}


void fill_indices_2d(const long *sorted_index_2d_data,
                     int elt, int chl, int kH, int kW, int nInputPlane, int outputHeight, int outputWidth, long* indices_data, cudaStream_t stream){
    const int kThreadsPerBlock = 1024;
    const int output_size = outputHeight * outputWidth;
    cudaError_t err;
    fill_indices2d_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size,
            sorted_index_2d_data,
            elt,  chl,  kH,  kW,  nInputPlane,  outputHeight,  outputWidth, indices_data
    );
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }


}



//__global__ void RRSVM_updateOutput_kernel(const int nthreads,
//                                          const float* /*[kW*kH, nInputPlan*outputHeight*outputwidth]*/im_col,
//                                          const float* /*[nInputPlan, kW*kH]*/ s,
//                                          float * /*[nInputPlan, outputHeight, outputWidth]*/ output_data,
//                                          long * /*[nInputPlan, outputHeight, outputWidth, kW*kH]*/indices,
//                                          const long nInputPlane, const long outputHeight, const outputWidth,
//                                          int kW, int kH,
//                                          int dW, int dH,
//                                          int padW, int padH,
//                                          int dilationW, int dilationH)
//{
//    CUDA_KERNEL_LOOP(index, nthreads)
//    {
//        // (n, c, ph, pw) is an element in the pooled output
//        int n = index;
//    }
//}


//void  RRSVM_updateOutput_cuda_laucher(const float *input, const float *s, float *output, long *indices,
//    const long batchSize, const long nInputPlane, const long inputHeight, const long inputWidth,
//    const long outputHeight, const outputWidth,
//    int kW, int kH,
//    int dW, int dH,
//    int padW, int padH,
//    int dilationW, int dilationH, cudaStream_t stream){
//
//    const int kThreadsPerBlock = 1024;
//    const int output_size =  nInputPlane * outputHeight * outputWidth;
//    cudaError_t err;
//
//
//    RRSVM_updateOutput_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
//
//    );
//
//    err = cudaGetLastError();
//    if(cudaSuccess != err)
//    {
//        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
//        exit( -1 );
//    }
//
//
//    }

#ifdef __cplusplus
}
#endif