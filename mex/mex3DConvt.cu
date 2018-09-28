#include <mex.h>
#include <gpu/mxGPUArray.h>
#include "cuda/cudamx_cudnn.cuh"

#define MAX(a,b) ((a)>(b))? (a) : (b)
#define ERROR(msg,...) mexErrMsgIdAndTxt("mex3DConvt:err", msg, ##__VA_ARGS__)

void DeConvForward(const CudaMxArray5D& in, const CudaMxArray5D& filter,
                const CudaMxArray2D& bias, const int* pad,
                const int* stride, mxArray **out_ptr) {
  // check input validity
  int c1 = in.channel();
  int c2 = filter.channel();
  int h = in.height();
  int w = in.width();
  int d = in.depth();

  int fh = filter.height();
  int fw = filter.width();
  int fd = filter.depth();

  if (c1 != filter.channel2()) {
    ERROR("(# input channels: %d) != (# filter channels: %d).",
      c1, filter.channel2());
  }

  if (c2 != bias.height()) {
    ERROR("(# out channels: %d) != (# bias channels: %d).",
      c2, bias.height());
  }

  cudnnHandle_t handle;
  CHECKCUDNN(cudnnCreate(&handle));
  cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc ;

  CHECKCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECKCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  CHECKCUDNN(cudnnCreateTensorDescriptor(&bias_desc)) ;
  CHECKCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  CHECKCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

  int in_dim[] = {1, c1, d, w, h};
  int in_stride[] = {c1*d*w*h, d*w*h, w*h, h, 1};
  int filter_dim[] = {c1, c2, fd, fw, fh};

  int out_dim[5];
  int oh = out_dim[4] = h*stride[2] - 2*pad[2] + fh - 1;
  int ow = out_dim[3] = w*stride[1] - 2*pad[1] + fw - 1;
  int od = out_dim[2] = d*stride[0] - 2*pad[0] + fd - 1;
  out_dim[1] = c2;
  out_dim[0] = 1;
  int out_stride[] = {c2*od*ow*oh, od*ow*oh, ow*oh, oh, 1};

  int bias_dim[] ={1, c2, 1, 1, 1};
  int bias_stride[] = {c2, 1, 1, 1};
  int dilate[] = {1,1,1};

  CHECKCUDNN(cudnnSetTensorNdDescriptor(input_desc,
    CUDNN_DATA_FLOAT, 5, in_dim, in_stride));
  CHECKCUDNN(cudnnSetTensorNdDescriptor(bias_desc,
    CUDNN_DATA_FLOAT, 5, bias_dim, bias_stride));
  CHECKCUDNN(cudnnSetFilterNdDescriptor(filter_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filter_dim));
  CHECKCUDNN(cudnnSetConvolutionNdDescriptor(conv_desc,
    3, pad, stride, dilate, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CHECKCUDNN(cudnnSetTensorNdDescriptor(output_desc,
    CUDNN_DATA_FLOAT, 5, out_dim, out_stride));

  cudnnConvolutionBwdDataAlgo_t conv_data_algo;
  CHECKCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(handle,
    filter_desc, input_desc, conv_desc, output_desc,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv_data_algo));

  size_t workspace_size = 0;
  CHECKCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
    filter_desc, input_desc, conv_desc, output_desc, conv_data_algo,
    &workspace_size));

  CudaMxArray5D out(oh,ow,od,c2,1);
  out.Wrap(out_ptr);

  void *d_workspace;
  cudaMalloc(&d_workspace, workspace_size);
  
  float alpha = 1, beta = 0;
  CHECKCUDNN(cudnnConvolutionBackwardData(handle,
    &alpha, filter_desc, filter.data(), input_desc, in.data(),
    conv_desc, conv_data_algo, d_workspace, workspace_size, &beta,
    output_desc, out.data()));

  beta = 1.0;
  CHECKCUDNN(cudnnAddTensor(handle,
    &alpha, bias_desc, bias.data(), &beta, output_desc,
    out.data()));

  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
  out.Destroy();
}

void DeConvBackward(const CudaMxArray5D& in, const CudaMxArray5D& filter,
                const CudaMxArray2D& bias, const CudaMxArray5D& der_out,
                const int* pad, const int* stride,
                mxArray **der_in_ptr, mxArray **der_filter_ptr,
                mxArray **der_bias_ptr) {

  // check input validity
  int c1 = in.channel();
  int c2 = filter.channel();
  int h = in.height();
  int w = in.width();
  int d = in.depth();
  int oh = der_out.height();
  int ow = der_out.width();
  int od = der_out.depth();

  int fh = filter.height();
  int fw = filter.width();
  int fd = filter.depth();

  if (c1 != filter.channel2()) {
    ERROR("(# input channels: %d) != (# filter channels: %d).",
      c1, filter.channel2());
  }

  if (c2 != bias.height()) {
    ERROR("(# out channels: %d) != (# bias channels: %d).",
      c2, bias.height());
  }

  cudnnHandle_t handle;
  CHECKCUDNN(cudnnCreate(&handle));
  cudnnTensorDescriptor_t der_out_desc, data_desc, bias_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc ;

  CHECKCUDNN(cudnnCreateTensorDescriptor(&der_out_desc));
  CHECKCUDNN(cudnnCreateTensorDescriptor(&data_desc));
  CHECKCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
  CHECKCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  CHECKCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

  int data_dim[] = {1, c1, d, w, h};
  int data_stride[] = {c1*d*w*h, d*w*h, w*h, h, 1};
  int der_out_dim[] = {1, c2, od, ow, oh};
  int der_out_stride[] = {c2*od*ow*oh, od*ow*oh, ow*oh, oh, 1};

  int filter_dim[] = {c1, c2, fd, fw, fh};
  int bias_dim[] ={1, c2, 1, 1, 1};
  int bias_stride[] = {c2, 1, 1, 1};

  CHECKCUDNN(cudnnSetTensorNdDescriptor(data_desc,
    CUDNN_DATA_FLOAT, 5, data_dim, data_stride));
  CHECKCUDNN(cudnnSetTensorNdDescriptor(der_out_desc,
    CUDNN_DATA_FLOAT, 5, der_out_dim, der_out_stride));
  CHECKCUDNN(cudnnSetTensorNdDescriptor(bias_desc,
    CUDNN_DATA_FLOAT, 5, bias_dim, bias_stride));
  CHECKCUDNN(cudnnSetFilterNdDescriptor(filter_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filter_dim));

  int dilate[] = {1, 1, 1};
  CHECKCUDNN(cudnnSetConvolutionNdDescriptor(conv_desc,
    3, pad, stride, dilate, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));


  cudnnConvolutionBwdFilterAlgo_t conv_filter_algo;
  CHECKCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(handle,
    der_out_desc, data_desc, conv_desc, filter_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv_filter_algo));


  cudnnConvolutionFwdAlgo_t conv_algo;
  CHECKCUDNN(cudnnGetConvolutionForwardAlgorithm(handle,
    der_out_desc, filter_desc, conv_desc, data_desc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo));

  size_t workspace_filter_size = 0, workspace_data_size = 0;
  CHECKCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
    der_out_desc, data_desc, conv_desc, filter_desc, conv_filter_algo,
    &workspace_filter_size));

  CHECKCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
    der_out_desc, filter_desc, conv_desc, data_desc, conv_algo, 
    &workspace_data_size));

  size_t workspace_size = MAX(workspace_filter_size,
    workspace_data_size);

  void *d_workspace;
  cudaMalloc(&d_workspace, workspace_size);

  CudaMxArray5D der_in(h,w,d,c1,1);
  CudaMxArray5D der_filter(fh,fw,fd,c2,c1);
  CudaMxArray2D der_bias(c2,1);
  der_in.Wrap(der_in_ptr);
  der_filter.Wrap(der_filter_ptr);
  der_bias.Wrap(der_bias_ptr);
  
  
  float alpha = 1, beta = 0;
  CHECKCUDNN(cudnnConvolutionBackwardBias(handle,
    &alpha, der_out_desc, der_out.data(), &beta, bias_desc, der_bias.data()));
  
  CHECKCUDNN(cudnnConvolutionBackwardFilter(handle,
    &alpha, der_out_desc, der_out.data(), data_desc, in.data(),
    conv_desc, conv_filter_algo, d_workspace, workspace_size, &beta,
    filter_desc, der_filter.data()));

  CHECKCUDNN(cudnnConvolutionForward(handle,
    &alpha, der_out_desc, der_out.data(), filter_desc, filter.data(),
    conv_desc, conv_algo, d_workspace, workspace_size, &beta,
    data_desc, der_in.data()));

  


  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(data_desc);
  cudnnDestroyTensorDescriptor(der_out_desc);
  cudnnDestroyTensorDescriptor(bias_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
  der_in.Destroy();
  der_filter.Destroy();
  der_bias.Destroy();
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxInitGPU();
  
  if (nrhs == 5) { // FORWARD
    CudaMxArray5D in(prhs[0]);
    CudaMxArray5D filter(prhs[1]);
    CudaMxArray2D bias(prhs[2]);

    const int *pad = (int*)mxGetData(prhs[3]);
    const int *stride = (int*)mxGetData(prhs[4]);

    int pad_rev[] = {pad[2], pad[1], pad[0]};
    int stride_rev[] = {stride[2], stride[1], stride[0]};

    DeConvForward(in, filter, bias, pad_rev, stride_rev, &plhs[0]);

    in.Destroy();
    filter.Destroy();
    bias.Destroy();
  } else if (nrhs == 6) { // BACKWARD
    CudaMxArray5D in(prhs[0]);
    CudaMxArray5D filter(prhs[1]);
    CudaMxArray2D bias(prhs[2]);
    CudaMxArray5D der_out(prhs[3]);

    const int *pad = (int*)mxGetData(prhs[4]);
    const int *stride = (int*)mxGetData(prhs[5]);

    int pad_rev[] = {pad[2], pad[1], pad[0]};
    int stride_rev[] = {stride[2], stride[1], stride[0]};

    DeConvBackward(in, filter, bias, der_out, pad_rev, stride_rev, &plhs[0],
      &plhs[1], &plhs[2]); // der_in, der_filter, der_bias

    in.Destroy();
    der_out.Destroy();
    filter.Destroy();
    bias.Destroy();
  } else {
    ERROR("invalid number of input parameters.");
  }
}