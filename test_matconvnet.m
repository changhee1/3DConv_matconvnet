% compare results with matconvnet 2D conv & convt
% requires vl_nnconv, vl_nnconvt in matconvnet

clear;
addpath(genpath('.'));

ch_out = 32;

% rng(1);
in = gpuArray(single(rand([300, 150, 1])));
filter = gpuArray(single(rand([3, 3, 1, 1, ch_out])));
filter2 = permute(filter, [1 2 3 5 4]);
bias = gpuArray(single(rand([ch_out, 1])));

pad = int32([1, 1, 0]);
stride = int32([2, 2, 1]);

conv_f = mex3DConv(in, filter, bias, pad, stride);
conv_f2 = vl_nnconv(in, filter2, bias, 'pad', 1, ...
        'stride',2, 'dilate',1);
diff_conv_f = gather(abs(squeeze(conv_f)-conv_f2));
diff_conv_f = mean(diff_conv_f(:))/gather(mean(conv_f2(:)));

conv_f2 = squeeze(conv_f);
[conv_b, conv_bf, conv_bb] = mex3DConv(in, filter, bias, conv_f, ...
    pad, stride);
[conv_b2, conv_bf2, conv_bb2] = vl_nnconv(in, filter2, bias, conv_f2,...
    'pad', 1, 'stride',2, 'dilate', 1);

diff_conv_b = gather(abs(squeeze(conv_b)-squeeze(conv_b2)));
diff_conv_b = mean(diff_conv_b(:))/gather(mean(conv_b2(:)));
diff_conv_bf = gather(abs(squeeze(conv_bf)-squeeze(conv_bf2)));
diff_conv_bf = mean(diff_conv_bf(:))/gather(mean(conv_bf2(:)));
diff_conv_bb = gather(abs(squeeze(conv_bb)-squeeze(conv_bb2)));
diff_conv_bb = mean(diff_conv_bb(:))/gather(mean(conv_bb2(:)));

bias = gpuArray(single(rand([1, 1])));
convt_f = mex3DConvt(conv_f, filter, bias, pad, stride);
convt_f2 = vl_nnconvt(squeeze(conv_f), filter2, bias, ...
        'upsample', 2, 'crop', [1 0 1 0]);
 
diff_convt_f = gather(abs(convt_f-convt_f2));
diff_convt_f = mean(diff_convt_f(:))/gather(mean(convt_f2(:)));

[convt_b, convt_bf, convt_bb] = mex3DConvt(conv_f, filter, bias, convt_f, ...
    pad, stride);
[convt_b2, convt_bf2, convt_bb2] = vl_nnconvt(squeeze(conv_f), filter2, bias, convt_f,...
    'upsample', 2, 'crop', [1 0 1 0]);

diff_convt_b = gather(abs(squeeze(convt_b)-squeeze(convt_b2)));
diff_convt_b = mean(diff_convt_b(:))/gather(mean(convt_b2(:)));
diff_convt_bf = gather(abs(squeeze(convt_bf)-squeeze(convt_bf2)));
diff_convt_bf = mean(diff_convt_bf(:))/gather(mean(convt_bf2(:)));
diff_convt_bb = gather(abs(squeeze(convt_bb)-squeeze(convt_bb2)));
diff_convt_bb = mean(diff_convt_bb(:))/gather(mean(convt_bb2(:)));

fprintf('## Average diff ratio of convolution\n');
fprintf('# Y: %f %%, dX: %f %%, dF %f %%, dB %f %%\n',...
    100*diff_conv_f, 100*diff_conv_b, 100*diff_conv_bf, 100*diff_conv_bb);

fprintf('## Average diff ratio of deconvolution\n');
fprintf('# Y: %f %%, dX: %f %%, dF %f %%, dB %f %%\n',...
    100*diff_convt_f, 100*diff_convt_b, 100*diff_convt_bf, 100*diff_convt_bb);