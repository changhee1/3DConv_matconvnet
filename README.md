# 3DConv_matconvnet
* cuDNN wrapper (easy, simple codes)
* 3D Conv, deconv mex functions for matconvnet.
* (requires CUDA, CUDNN)

## install
* call `compile();`
* tested on Matlab R2017b, Ubuntu 14.04, GCC 4.8.4, CUDA 8.0, cuDNN 7.1

## usage:
3D Convolution
* `Y = mex3DConv(X, F, B, pad, stride)`;
* `[dX, dF, dB] = mex3DConv(X, F, B, Y, pad, stride)`;

3D Deconvolution (Transposed convolution)
* `Y = mex3DConvt(X, F, B, pad, stride)`;
* `[dX, dF, dB] = mex3DConvt(X, F, B, Y, pad, stride)`;

more details are in `example.m`
