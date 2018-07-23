clear;
addpath(genpath('.'));
% compile();
rng(1);
in = gpuArray(single(rand([128, 96, 32 4]))); % H W D C
filter = gpuArray(single(rand([5, 5, 3, 4, 8]))); % H W D in out
bias = gpuArray(single(rand([8, 1]))); % out 1

pad = int32([2, 2, 1]); % H W D
stride = int32([2, 2, 1]); % H W D

fprintf('# 3D Convolution\n');
fprintf('data size: [%d, %d, %d ,%d]\n', size(in));
fprintf('filter size: [%d, %d, %d, %d, %d]\n', size(filter));
fprintf('pad: [%d, %d, %d]\n', pad);
fprintf('stride: [%d, %d, %d]\n', stride);
out = mex3DConv(in, filter, bias, pad, stride);
% [dIn, dFilter, dBias] = mex3DConv(in, filter, bias, out, pad, stride);
fprintf('Output data size: [%d, %d, %d, %d]\n', size(out));

% IMPORTANT to switch between # channels of IN & OUT in filter size
filter = gpuArray(single(rand([5, 5, 3, 4, 8]))); % H W D out in
bias = gpuArray(single(rand([4, 1]))); % out 1

fprintf('# 3D Deconvolution\n');
fprintf('Input data size: [%d, %d, %d ,%d]\n', size(out));
fprintf('Input filter size: [%d, %d, %d, %d, %d]\n', size(filter));
fprintf('pad: [%d, %d, %d]\n', pad);
fprintf('stride: [%d, %d, %d]\n', stride);
out2 = mex3DConvt(out, filter, bias, pad, stride);
% [dOut2, dFilter2, dBias2] = mex3DConvt(out, filter, bias, out2, pad, stride);
fprintf('Output data size: [%d, %d, %d, %d]\n', size(out2));


