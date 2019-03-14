# 3DConv_matconvnet
* cuDNN wrapper (easy, simple codes)
* 3D Conv, deconv mex functions for matconvnet.
* (requires CUDA, CUDNN)

## install
* call `compile();`
* tested on Matlab R2017b/2018a, Ubuntu 14.04/16.04, CUDA 8.0/9.0, cuDNN 7.1

## usage:
3D Convolution
* `Y = mex3DConv(X, F, B, pad, stride)`;
* `[dX, dF, dB] = mex3DConv(X, F, B, Y, pad, stride)`;

3D Deconvolution (Transposed convolution)
* `Y = mex3DConvt(X, F, B, pad, stride)`;
* `[dX, dF, dB] = mex3DConvt(X, F, B, Y, pad, stride)`;

more details are in `example.m`
