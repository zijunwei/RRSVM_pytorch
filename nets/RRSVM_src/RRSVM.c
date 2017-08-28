#include <TH/TH.h>
#define real float


static void c_forward_single_batch( real *input_p, real *output_p, THIndex_t *ind_p, long nslices,
          long iwidth,
          long iheight,
          long owidth,
          long oheight,
          int kW,
          int kH)
          {
long k
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    real *ip = input_p   + k*iwidth*iheight;
    THLongTensor *indp = ind_p   + k*iwidth*iheight;
    real *sp = s + k*kW*kH
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        long hstart = i * kW;
        long wstart = j * kH;
        long hend = fminf(hstart + (kH), iheight);
        long wend = fminf(wstart + (kW), iwidth);

        /* local pointers */
        real *op = output_p  + k*owidth*oheight + i*owidth + j;

        long tcntr = 0;
        long x,y;

        THTensor * tmpVal = THTensor_(new)();
        THTensor_(resize1d)(tmpVal, kH * kW);
        real *tmp__data = THTensor_(data)(tmpVal);

        THLongTensor * tmpIndices = THLongTensor_new();
        THLongTensor_resize1d(tmpIndices, kH * kW);
        long *tmpi__data = THLongTensor_data(tmpIndices);


        for(y = hstart; y < hend; y += 1)
        {
          for(x = wstart; x < wend; x += 1)
          {
            tcntr = y*iwidth + x;

            tmp__data[tcntr] = *(ip + tcntr);
            tmpi__data[tcntr] = tcntr

          }
        }
        THTensor_(quicksortdescend)(tmp__data, tmpi__data, kH*kW, 1);

        real s_output = 0;
        for(y = hstart; y < hend; y += 1)
        {
          for(x = wstart; x < wend; x += 1)
          {
            tcntr = y*iwidth + x;
            s_output += *(sp + tcntr) * tmp__data[tcntr]
            *(indp + tcntr) = tmpi__data[tcntr] + TH_INDEX_BASE

          }
        }

        /* set output to local max */
        *op = s_output;
      }
    }
  }

  THTensor_(free)(tmpVal);
  THLongTensor_free(tmpIndices);
}

int c_forward(THFloatTensor *output, THFloatTensor *input, THFloatTensor *s, THLongTensor *indices)
{

  real * input_data;
  real * s_data;
  THLongTensor * indices_data;

  real * output_data;

  int kW = 3;
  int kH = 3;


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
   int nPlane = input->size[1];




  // Resize output
  THTensor_(resize4d)(output, batchSize, nPlane, outputHeight, outputWidth);
  THTensor_(zero)(output);
  THIndexTensor_(resize4d)(indices, batchSize, nPlane, inputHeight, inputWidth)

  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);
  s_data = THTensor_(data)(s);
  indices_data = THIndexTensor_(data)(indices);

    long p
    #pragma omp parallel for private(p)
        for (p = 0; p < batchSize; p++)
        {
          c_forward_single_batch
         (input_data+p*nPlane*inputWidth*inputHeight,
          output_data+p*nPlane*outputWidth*outputHeight,
          indices_data+p*nPlane*outputWidth*outputHeight,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH);
        }

  THTensor_(free)(input);
  THTensor_(free)(s);


}

int c_backward_grad_input(THFloatTensor *grad_output, THFloatTensor *grad_input, THFloatTensor *input, THFloatTensor *s){


}

int c_backward_grad_params(THFloatTensor *grad_output, THFloatTensor *grad_s, THFloatTensor *input, THFloatTensor *s){


}

